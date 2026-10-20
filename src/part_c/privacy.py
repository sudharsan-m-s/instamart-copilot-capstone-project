"""
privacy.py - Part C Tasks 1 & 2: Input Guardrails and PII Redaction for Support Tickets.
Ensures no empty or long tickets reach LLMs and completely strips emails, phone numbers, and order IDs.
"""

import re
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mock_llm import est_tokens

# Compiled regex patterns strictly in order
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")

MAX_INPUT_TOKENS = 400


def check_input_guardrails(text: str) -> tuple[str, bool]:
    """
    Applies Task 1 input guardrails:
    - (a) Empty or whitespace-only -> status 'skipped_empty', never sent to model.
    - (b) est_tokens(text) > 400 -> status 'flagged_long', never sent to model.
    Returns: (status, is_eligible_for_model)
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return "skipped_empty", False
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long", False
    return "eligible", True


def redact_pii(text: str) -> tuple[str, dict]:
    """
    Applies Task 2 PII redaction in strict order:
    1. Email -> [EMAIL]
    2. Phone -> [PHONE]
    3. Order ID -> [ORDER_ID]
    Returns: (redacted_text, count_dict)
    """
    t, n_email = EMAIL_PATTERN.subn("[EMAIL]", text or "")
    t, n_phone = PHONE_PATTERN.subn("[PHONE]", t)
    t, n_order = ORDER_PATTERN.subn("[ORDER_ID]", t)
    counts = {
        "email": n_email,
        "phone": n_phone,
        "order_id": n_order,
        "total": n_email + n_phone + n_order,
    }
    return t, counts


def verify_redaction(text: str) -> bool:
    """Verifies that no unmasked PII remains in the text."""
    if EMAIL_PATTERN.search(text):
        return False
    if PHONE_PATTERN.search(text):
        return False
    if ORDER_PATTERN.search(text):
        return False
    return True


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TICKETS_PATH = PROJECT_ROOT / "data" / "raw" / "tickets.csv"

    print("=" * 70)
    print("TESTING PRIVACY & INPUT GUARDRAILS ON RAW TICKETS")
    print("=" * 70)

    df_tickets = pd.read_csv(TICKETS_PATH, keep_default_na=False)

    total_substitutions = {"email": 0, "phone": 0, "order_id": 0, "total": 0}
    empty_count = 0
    long_count = 0
    all_clean = True

    for _, row in df_tickets.iterrows():
        text = row["text"]
        status, eligible = check_input_guardrails(text)
        if status == "skipped_empty":
            empty_count += 1
        elif status == "flagged_long":
            long_count += 1

        redacted, counts = redact_pii(text)
        for k in total_substitutions:
            total_substitutions[k] += counts[k]

        if not verify_redaction(redacted):
            all_clean = False

    print(f"Total tickets audited : {len(df_tickets)}")
    print(f"Skipped empty tickets : {empty_count}")
    print(f"Flagged long tickets  : {long_count}")
    print(f"Substitutions -> Email: {total_substitutions['email']}, Phone: {total_substitutions['phone']}, Order ID: {total_substitutions['order_id']}")
    print(f"Total substitutions   : {total_substitutions['total']}")
    print(f"All redacted texts clean (0 PII left): {all_clean}")

    assert empty_count == 4, f"Expected 4 empty tickets, got {empty_count}"
    assert long_count == 4, f"Expected 4 long tickets, got {long_count}"
    assert total_substitutions["email"] == 55, f"Expected 55 emails, got {total_substitutions['email']}"
    assert total_substitutions["phone"] == 91, f"Expected 91 phones, got {total_substitutions['phone']}"
    assert total_substitutions["order_id"] == 132, f"Expected 132 order IDs, got {total_substitutions['order_id']}"
    assert total_substitutions["total"] == 278, f"Expected 278 total substitutions, got {total_substitutions['total']}"
    assert all_clean, "Unmasked PII detected in redacted texts!"

    print("\n[PASS] All Privacy & Guardrail checkpoints verified successfully.")
