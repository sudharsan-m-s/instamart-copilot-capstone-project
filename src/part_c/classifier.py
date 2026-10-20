import sys
import json
import re
import random
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mock_llm import chat, TransientLLMError

# Allowed ticket categories
ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]

# Task 3: Zero-shot prompt (strictly NO 'EXAMPLES:')
ZERO_SHOT_PROMPT = """You are an automated support ticket triage assistant. CLASSIFY the support ticket into exactly one of the four allowed categories: billing, delivery, technical, fraud.
Respond with a valid JSON object only: {"category": "...", "reason": "..."}."""

# Task 3: Few-shot prompt (strictly CONTAINS 'EXAMPLES:' with at least one fraud example)
FEW_SHOT_PROMPT = """You are an automated support ticket triage assistant. CLASSIFY the support ticket into exactly one of the four allowed categories: billing, delivery, technical, fraud.
Respond with a valid JSON object only: {"category": "...", "reason": "..."}.
EXAMPLES:
Ticket: There is an unauthorized transaction on my card linked to order [ORDER_ID]. I did not place it. Call [PHONE].
{"category": "fraud", "reason": "unauthorized card transaction"}
Ticket: The rider marked [ORDER_ID] as delivered but I never received the package. Call [PHONE].
{"category": "delivery", "reason": "undelivered package"}"""


def call_with_retry(messages: list[dict], ticket_id: str, max_retries: int = 3, base_delay: float = 0.05):
    """Retry transient failures with a logged exponential delay and jitter."""
    for n in range(max_retries + 1):
        try:
            return chat(messages)
        except TransientLLMError as e:
            if n == max_retries:
                break
            wait = base_delay * (2 ** n) * random.uniform(0.9, 1.15)
            print(f"[RETRY] Ticket {ticket_id}: retry {n}, waiting {wait:.4f}s ({e})")
            time.sleep(wait)
    raise RuntimeError(f"Permanent failure for ticket {ticket_id} after {max_retries} retries.")


def validate_response(content: str) -> tuple[str, str | None, str | None]:
    """Return status, category and reason; ambiguous replies go to human review."""
    # 1. Check for valid JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            category = parsed.get("category")
            reason = parsed.get("reason", "")
            if category in ALLOWED_CATEGORIES:
                return "classified_json", category, reason
            else:
                return "flagged_invalid_output", None, f"JSON category '{category}' outside allowed list"
        return "flagged_invalid_output", None, "JSON response must be an object"
    except json.JSONDecodeError:
        # Keyword recovery applies only to invalid JSON, never to a JSON list or string.
        words = re.findall(r"[a-z]+", (content or "").lower())
        found_categories = [c for c in ALLOWED_CATEGORIES if c in words]
        if len(found_categories) == 1:
            return "classified_keyword", found_categories[0], "keyword extraction from conversational reply"

    # 3. Fallback to human review
    return "flagged_invalid_output", None, "Unparseable or ambiguous output"
