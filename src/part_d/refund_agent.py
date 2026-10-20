"""ReAct loop with bounded work and checkpoints after tool observations."""

import csv
import json
import re
import time
from pathlib import Path
from uuid import uuid4

import mock_llm
from src.part_d.tools import TOOL_MAP, TOOLS_SPEC

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "state"
RUNS_PATH = ROOT / "outputs/agent_runs.jsonl"
SYSTEM_PROMPT = (
    "AGENT: Use the tools to check the order and refund policy. Only completed "
    "orders delivered after 45 minutes qualify. Escalate amounts above INR 1000. "
    "All requests are within 7 days. Never claim a refund before the tool confirms it."
)


class SimulatedCrash(RuntimeError):
    """The demonstration stops only after a durable tool checkpoint."""


def checkpoint_path(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Run ids may contain only letters, numbers, underscores and hyphens.")
    return STATE_DIR / f"{run_id}.json"


def save_checkpoint(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(state["run_id"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def finish(state, status, final_answer):
    result = {
        "run_id": state["run_id"], "scenario_id": state["scenario_id"],
        "status": status, "final_answer": final_answer,
        "steps": state["steps"], "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
    }
    state["result"] = result
    save_checkpoint(state)
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    return result


def fallback(state, reason):
    return finish(
        state, f"FALLBACK_{reason}",
        f"Human review needed: {reason.lower().replace('_', ' ')} reached "
        f"after {len(state['steps'])} tools and {state['llm_calls']} model calls. "
        "Review the saved observations before taking further action.",
    )


def _continue(state, crash_after_tools=None):
    started = time.monotonic()
    elapsed_before_resume = state["elapsed_seconds"]

    while True:
        state["elapsed_seconds"] = elapsed_before_resume + time.monotonic() - started
        if state["elapsed_seconds"] >= state["time_limit_seconds"]:
            return fallback(state, "TIMEOUT")

        # A checkpoint can contain remaining calls from an assistant reply.
        # Completed observations are already in messages, so they are not replayed.
        if state["pending_tools"]:
            call = state["pending_tools"][0]
            tool = TOOL_MAP.get(call["name"])
            if tool is None:
                return fallback(state, "TOOL_ERROR")
            try:
                observation = tool(**call["arguments"])
            except (ValueError, TypeError) as error:
                observation = {"error": str(error)}
                failed = True
            else:
                failed = False
            state["messages"].append({
                "role": "tool", "tool_call_id": call["id"], "name": call["name"],
                "content": json.dumps(observation),
            })
            state["steps"].append({
                "tool": call["name"], "arguments": call["arguments"],
                "observation": observation,
            })
            state["tools_executed"].append(call["name"])
            state["step_number"] = len(state["steps"])
            state["pending_tools"].pop(0)
            state["elapsed_seconds"] = elapsed_before_resume + time.monotonic() - started
            save_checkpoint(state)
            if crash_after_tools == len(state["steps"]):
                raise SimulatedCrash(f"Saved {state['run_id']} after {len(state['steps'])} tools.")
            if failed:
                return fallback(state, "TOOL_ERROR")
            continue

        if state["llm_calls"] >= state["max_iterations"]:
            return fallback(state, "MAX_ITERATIONS")
        response = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens
        # A reply that exceeds the budget must not execute its requested tool.
        if state["tokens_used"] > state["token_limit"]:
            return fallback(state, "TOKEN_LIMIT")
        reply = response.choices[0].message
        if not reply.tool_calls:
            final_answer = reply.content or ""
            status = final_answer.split(":", 1)[0]
            state["messages"].append({"role": "assistant", "content": final_answer})
            return finish(state, status, final_answer)
        calls = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in reply.tool_calls
        ]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        state["pending_tools"] = list(calls)


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, *, run_id=None, crash_after_tools=None):
    run_id = run_id or uuid4().hex
    if checkpoint_path(run_id).exists():
        raise ValueError("Run id already exists; use resume_agent or a new run id.")
    state = {
        "run_id": run_id, "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0, "llm_calls": 0, "step_number": 0,
        "steps": [], "tools_executed": [], "pending_tools": [],
        "elapsed_seconds": 0.0,
        "max_iterations": max_iterations, "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
    }
    return _continue(state, crash_after_tools)


def resume_agent(run_id):
    state = json.loads(checkpoint_path(run_id).read_text(encoding="utf-8"))
    if "result" in state:
        return state["result"]
    return _continue(state)


def main():
    from src.part_d.tools import LEDGER_PATH

    # Regenerate only this script's named demonstration artifacts.
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNS_PATH.write_text("", encoding="utf-8")
    LEDGER_PATH.write_text("", encoding="utf-8")
    names = [f"demo-S{i}" for i in range(1, 8)]
    names += ["demo-token", "demo-timeout", "demo-resume"]
    for name in names:
        checkpoint_path(name).unlink(missing_ok=True)

    with (ROOT / "data/raw/agent_scenarios.csv").open(encoding="utf-8", newline="") as f:
        scenarios = list(csv.DictReader(f))
    results = []
    for scenario in scenarios:
        result = run_agent(scenario["message"], scenario["scenario_id"],
                           run_id=f"demo-{scenario['scenario_id']}")
        expected = scenario["expected_outcome"]
        if expected == "FALLBACK":
            expected = "FALLBACK_MAX_ITERATIONS"
        assert result["status"] == expected, result
        amount = next((
            s["observation"]["refund_inr"] for s in reversed(result["steps"])
            if s["tool"] in {"issue_refund", "escalate_to_human"}
        ), 0.0)
        assert abs(amount - float(scenario["expected_refund_inr"])) < 0.01
        print(scenario["scenario_id"], result["status"], f"INR {amount:.2f}",
              result["llm_calls"], "calls", result["tokens_used"], "tokens")
        results.append(result)
    assert results[0]["llm_calls"] == 5
    assert results[6]["llm_calls"] == 6

    first_message = scenarios[0]["message"]
    token_demo = run_agent(first_message, "S1-token", token_limit=200, run_id="demo-token")
    timeout_demo = run_agent(first_message, "S1-timeout", time_limit_seconds=0,
                             run_id="demo-timeout")
    assert token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    assert timeout_demo["status"] == "FALLBACK_TIMEOUT"
    assert timeout_demo["llm_calls"] == 0
    try:
        run_agent(first_message, "S1-resume", run_id="demo-resume", crash_after_tools=2)
    except SimulatedCrash as error:
        print(error)
    else:
        raise AssertionError("The crash demonstration did not stop.")
    resumed = resume_agent("demo-resume")
    executed = [s["tool"] for s in resumed["steps"]]
    assert executed == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["status"] == "REFUND"
    assert resumed["steps"][-1]["observation"]["refund_inr"] == 128.25
    assert len(RUNS_PATH.read_text(encoding="utf-8").splitlines()) == 10
    assert len(LEDGER_PATH.read_text(encoding="utf-8").splitlines()) == 4
    print("Token demo:", token_demo["status"], token_demo["tokens_used"], "tokens")
    print("Timeout demo:", timeout_demo["status"], timeout_demo["llm_calls"], "calls")
    print("Crash/resume tools:", " -> ".join(executed))
    print("ALL PART D AGENT CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
