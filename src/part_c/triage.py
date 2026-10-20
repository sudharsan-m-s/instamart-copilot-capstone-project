"""
triage.py - Part C Tasks 6 & 7: Zero-Shot vs Few-Shot Comparison, Token Budget Demo, and Production Routing.
Outputs:
- outputs/triage_budget_demo.csv
- outputs/triage_results.csv
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm
from src.part_c.privacy import check_input_guardrails, redact_pii, verify_redaction
from src.part_c.classifier import (
    ZERO_SHOT_PROMPT,
    FEW_SHOT_PROMPT,
    ALLOWED_CATEGORIES,
    call_with_retry,
    validate_response,
)

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TICKETS_PATH = DATA_DIR / "raw" / "tickets.csv"
SEGMENTS_PATH = OUTPUTS_DIR / "customer_segments.csv"


def run_pipeline(df_tickets: pd.DataFrame, prompt: str, token_budget: int | None = None):
    """
    Executes the support ticket triage pipeline over df_tickets.
    """
    mock_llm.reset()
    results = []
    total_tokens_used = 0

    for _, row in df_tickets.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        raw_text = row["text"]
        true_cat = row.get("true_category", "")

        guard_status, is_eligible = check_input_guardrails(raw_text)

        if not is_eligible:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": guard_status,
                "category": "",
                "reason": f"Input guardrail: {guard_status}",
                "true_category": true_cat,
                "tokens_used": 0,
            })
            continue

        # Check token budget before dispatching model call
        if token_budget is not None and total_tokens_used >= token_budget:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "budget_exhausted",
                "category": "",
                "reason": "Token budget exceeded",
                "true_category": true_cat,
                "tokens_used": 0,
            })
            continue

        # Redact PII
        redacted_text, _ = redact_pii(raw_text)
        assert verify_redaction(redacted_text), f"PII detected in prompt for ticket {tid}!"

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": redacted_text},
        ]
        assert all(verify_redaction(m["content"]) for m in messages), "PII in model messages"

        try:
            resp = call_with_retry(messages, ticket_id=tid)
            tokens = resp.usage.total_tokens
            total_tokens_used += tokens
            content = resp.choices[0].message.content

            val_status, pred_category, reason = validate_response(content)
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": val_status,
                "category": pred_category or "",
                "reason": reason or "",
                "true_category": true_cat,
                "tokens_used": tokens,
            })
        except RuntimeError as e:
            results.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "flagged_invalid_output",
                "category": "",
                "reason": str(e),
                "true_category": true_cat,
                "tokens_used": 0,
            })

    return pd.DataFrame(results), total_tokens_used


def route_tickets(classified, segments):
    """Apply human-review, fraud, premium and standard routing in that order."""
    lookup = segments.set_index("customer_id")
    rows = []
    for _, row in classified.iterrows():
        cid = row["customer_id"]
        known = cid in lookup.index
        premium = known and str(lookup.loc[cid, "is_premium"]).lower() == "true"
        segment = str(lookup.loc[cid, "segment_name"]) if known else ""
        valid = row["status"] in {"classified_json", "classified_keyword"}
        if not valid:
            route, priority = "human_review", "NORMAL"
        elif row["category"] == "fraud":
            route, priority = "security_team", "HIGH"
        elif premium:
            route, priority = "premium_desk", "HIGH"
        else:
            route, priority = "support_queue", "NORMAL"
        rows.append({
            "ticket_id": row["ticket_id"], "customer_id": cid,
            "status": row["status"], "category": row["category"] if valid else "",
            "segment_name": segment, "priority": priority, "route": route,
            "tokens_used": row["tokens_used"],
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("PART C: SUPPORT TICKET TRIAGE WITH GUARDRAILS")
    print("=" * 70)

    df_tickets = pd.read_csv(TICKETS_PATH, keep_default_na=False)
    # A missing customer is non-premium; a missing Part B artifact is a setup error.
    df_segments = pd.read_csv(SEGMENTS_PATH)
    print(f"Loaded {len(df_tickets)} tickets from {TICKETS_PATH}")

    # ==========================================================================
    # Task 6: Compare Zero-Shot and Few-Shot Runs
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TASK 6: ZERO-SHOT VS FEW-SHOT COMPARISON (NO TOKEN BUDGET)")
    print("=" * 70)

    print("\n--- Running Zero-Shot Pipeline ---")
    df_zero, tokens_zero = run_pipeline(df_tickets, ZERO_SHOT_PROMPT)

    print("\n--- Running Few-Shot Pipeline ---")
    df_few, tokens_few = run_pipeline(df_tickets, FEW_SHOT_PROMPT)

    # Filter classified tickets
    classified_statuses = ["classified_json", "classified_keyword"]
    zero_class = df_zero[df_zero["status"].isin(classified_statuses)]
    few_class = df_few[df_few["status"].isin(classified_statuses)]

    zero_correct = (zero_class["category"] == zero_class["true_category"]).sum()
    few_correct = (few_class["category"] == few_class["true_category"]).sum()

    zero_acc = (zero_correct / len(zero_class)) * 100
    few_acc = (few_correct / len(few_class)) * 100

    print("\n" + "-" * 70)
    print(f"{'Metric':<30} | {'Zero-Shot':<16} | {'Few-Shot':<16}")
    print("-" * 70)
    print(f"{'Classified Tickets':<30} | {len(zero_class):<16} | {len(few_class):<16}")
    print(f"{'Accuracy (%)':<30} | {zero_acc:<16.1f} | {few_acc:<16.1f}")
    print(f"{'Total Tokens Used':<30} | {tokens_zero:<16} | {tokens_few:<16}")
    print(f"{'Tokens per Classified Ticket':<30} | {tokens_zero / len(zero_class):<16.1f} | {tokens_few / len(few_class):<16.1f}")
    print("-" * 70)

    # Zero-Shot Confusion Matrix
    print("\n--- Zero-Shot Confusion Matrix (Rows: True, Columns: Pred) ---")
    labels = ["billing", "delivery", "technical", "fraud"]
    cm_zero = confusion_matrix(zero_class["true_category"], zero_class["category"], labels=labels)
    cm_df = pd.DataFrame(cm_zero, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    print(cm_df)

    # Status counts verification (135 json, 12 keyword, 9 invalid)
    status_counts_few = df_few["status"].value_counts().to_dict()
    print("\nStatus Counts Breakdown (Few-Shot):", status_counts_few)
    assert status_counts_few.get("classified_json", 0) == 135, f"Expected 135 json, got {status_counts_few.get('classified_json')}"
    assert status_counts_few.get("classified_keyword", 0) == 12, f"Expected 12 keyword, got {status_counts_few.get('classified_keyword')}"
    assert status_counts_few.get("flagged_invalid_output", 0) == 9, f"Expected 9 invalid, got {status_counts_few.get('flagged_invalid_output')}"
    assert status_counts_few.get("skipped_empty", 0) == 4, f"Expected 4 empty, got {status_counts_few.get('skipped_empty')}"
    assert status_counts_few.get("flagged_long", 0) == 4, f"Expected 4 long, got {status_counts_few.get('flagged_long')}"
    assert round(zero_acc, 1) == 85.7, f"Expected zero-shot accuracy 85.7%, got {zero_acc:.1f}%"
    assert round(few_acc, 1) == 100.0, f"Expected few-shot accuracy 100%, got {few_acc:.1f}%"
    print("[PASS] Status counts (135/12/9) and accuracies (85.7% / 100%) match reference checkpoints.")

    # ==========================================================================
    # Task 7a: Token Budget Demo
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TASK 7A: TOKEN BUDGET DEMO (TOKEN_BUDGET = 6000)")
    print("=" * 70)

    df_budget, tokens_budget = run_pipeline(df_tickets, ZERO_SHOT_PROMPT, token_budget=6000)
    exhausted_count = (df_budget["status"] == "budget_exhausted").sum()
    print(f"Total tokens used before exhaustion: {tokens_budget}")
    print(f"Tickets ending in 'budget_exhausted': {exhausted_count}")

    budget_demo_path = OUTPUTS_DIR / "triage_budget_demo.csv"
    route_tickets(df_budget, df_segments).to_csv(budget_demo_path, index=False)
    print(f"Saved budget demo results to: {budget_demo_path}")
    assert exhausted_count > 0, "No tickets were stopped by the token budget!"

    # ==========================================================================
    # Task 7b: Production Routing
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TASK 7B: PRODUCTION ROUTING & OUTPUT GENERATION (TOKEN_BUDGET = 100000)")
    print("=" * 70)

    # Run few-shot production pipeline
    df_prod, _ = run_pipeline(df_tickets, FEW_SHOT_PROMPT, token_budget=100000)
    df_triage_final = route_tickets(df_prod, df_segments)
    final_output_path = OUTPUTS_DIR / "triage_results.csv"
    df_triage_final.to_csv(final_output_path, index=False)

    print(f"\nSaved production triage results to: {final_output_path}")
    print(f"Total tickets routed: {len(df_triage_final)}")
    print("\nRouting Breakdown:")
    print(df_triage_final["route"].value_counts())
    print("\nPriority Breakdown:")
    print(df_triage_final["priority"].value_counts())

    assert len(df_triage_final) == 164, f"Expected 164 tickets, got {len(df_triage_final)}"
    print("\n[PASS] All Part C Acceptance Criteria passed successfully.")


if __name__ == "__main__":
    main()
