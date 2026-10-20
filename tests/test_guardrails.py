"""Boundary checks for privacy, retry behaviour and simulated refund actions."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mock_llm import TransientLLMError
from src.part_c.classifier import call_with_retry, validate_response, ZERO_SHOT_PROMPT
from src.part_c.triage import run_pipeline
from src.part_d import refund_agent as agent
from src.part_d import tools
from src.part_d.policy_assistant import evaluate


class TriageTests(unittest.TestCase):
    def test_json_cannot_use_keyword_escape_hatch(self):
        for text in ['["billing"]', '"billing"', '{"category": null}',
                     '{"category": "shipping", "reason": "billing"}',
                     '{"category": ["billing"]}', "fraud or billing"]:
            with self.subTest(text=text):
                self.assertEqual(validate_response(text)[0], "flagged_invalid_output")
        self.assertEqual(validate_response("A billing-related problem.")[:2],
                         ("classified_keyword", "billing"))

    def test_three_retries_sleep_and_four_attempts(self):
        failures = [TransientLLMError("busy") for _ in range(3)]
        with patch("src.part_c.classifier.chat", side_effect=failures + ["ok"]) as chat, \
             patch("src.part_c.classifier.time.sleep") as sleep, \
             patch("src.part_c.classifier.random.uniform", return_value=1.0), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(call_with_retry([], "test"), "ok")
        self.assertEqual(chat.call_count, 4)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [0.05, 0.1, 0.2])

    def test_guardrails_and_zero_budget_make_no_calls(self):
        frame = pd.DataFrame([
            {"ticket_id": "empty", "customer_id": "C1", "text": "  "},
            {"ticket_id": "long", "customer_id": "C1", "text": "word " * 500},
            {"ticket_id": "budget", "customer_id": "C1", "text": "My delivery is late."},
        ])
        with patch("src.part_c.triage.call_with_retry") as call:
            results, tokens = run_pipeline(frame, ZERO_SHOT_PROMPT, token_budget=0)
        call.assert_not_called()
        self.assertEqual(tokens, 0)
        self.assertEqual(results["status"].tolist(),
                         ["skipped_empty", "flagged_long", "budget_exhausted"])

    def test_every_outgoing_message_is_redacted(self):
        from mock_llm import chat
        from src.part_c.privacy import verify_redaction
        frame = pd.DataFrame([{
            "ticket_id": "private", "customer_id": "C1",
            "text": "Refund ORD-100048; call 9876543210 or email me@example.com.",
        }])
        def inspect(messages, **kwargs):
            self.assertTrue(all(verify_redaction(m["content"]) for m in messages))
            self.assertEqual(messages[-1]["content"],
                             "Refund [ORDER_ID]; call [PHONE] or email [EMAIL]")
            return chat(messages)
        with patch("src.part_c.triage.call_with_retry", side_effect=inspect):
            run_pipeline(frame, ZERO_SHOT_PROMPT)


class PolicyTests(unittest.TestCase):
    def test_overlap_restores_grounding(self):
        normal, ablation = evaluate(10), evaluate(0)
        self.assertEqual((normal["retrieval_hits"], normal["fact_hits"], normal["fallback_hits"]),
                         (12, 12, 2))
        self.assertEqual(ablation["fact_hits"], 10)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        for module, name, value in [
            (agent, "STATE_DIR", root / "state"),
            (agent, "RUNS_PATH", root / "runs.jsonl"),
            (tools, "LEDGER_PATH", root / "ledger.jsonl"),
        ]:
            patcher = patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.message = "Please refund ORD-100048."

    def test_refund_thresholds_and_action_validation(self):
        self.assertEqual(tools.calculate_refund(1000, 45)["refund_inr"], 0)
        self.assertEqual(tools.calculate_refund(1000, 46)["refund_inr"], 1000)
        for order_id, amount in [("ORD-100025", 100), ("ORD-100145", 1999),
                                 ("ORD-100048", 999), ("ORD-100001", 1)]:
            with self.subTest(order=order_id), self.assertRaises(ValueError):
                tools.issue_refund(order_id, amount)
        self.assertFalse(tools.LEDGER_PATH.exists())
        self.assertEqual(tools.get_order("' OR 1=1 --"), {"error": "order not found"})

    def test_stop_before_tool_when_tokens_exceeded(self):
        result = agent.run_agent(self.message, "token", token_limit=0)
        self.assertEqual(result["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["llm_calls"], 1)
        self.assertFalse(tools.LEDGER_PATH.exists())

    def test_timeout_before_first_call(self):
        with patch.object(agent.mock_llm, "chat") as chat:
            result = agent.run_agent(self.message, "timeout", time_limit_seconds=0)
        chat.assert_not_called()
        self.assertEqual(result["status"], "FALLBACK_TIMEOUT")

    def test_loop_stops_after_six_calls(self):
        result = agent.run_agent("[LOOP] " + self.message, "loop")
        self.assertEqual(result["status"], "FALLBACK_MAX_ITERATIONS")
        self.assertEqual(result["llm_calls"], 6)
        self.assertEqual(len(result["steps"]), 6)
        self.assertLessEqual(result["tokens_used"], 6000)

    def test_resume_preserves_limits_and_does_not_repeat_tools(self):
        with self.assertRaises(agent.SimulatedCrash):
            agent.run_agent(self.message, "resume", run_id="resume", crash_after_tools=2)
        saved = json.loads(agent.checkpoint_path("resume").read_text(encoding="utf-8"))
        self.assertEqual(saved["tools_executed"], ["get_order", "search_policy"])
        result = agent.resume_agent("resume")
        self.assertEqual(result["status"], "REFUND")
        self.assertEqual(result["llm_calls"], 5)
        self.assertEqual([s["tool"] for s in result["steps"]],
                         ["get_order", "search_policy", "calculate_refund", "issue_refund"])
        self.assertEqual(result["steps"][-1]["observation"]["refund_inr"], 128.25)
        # Resuming a finished checkpoint returns its result, without another ledger write.
        self.assertEqual(agent.resume_agent("resume"), result)
        self.assertEqual(len(tools.LEDGER_PATH.read_text(encoding="utf-8").splitlines()), 1)
        self.assertEqual(len(agent.RUNS_PATH.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
