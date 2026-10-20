"""Answer policy questions using only the retrieved policy context."""

import csv
import json
from collections import Counter
from pathlib import Path

import mock_llm
from src.part_d.retrieval import PolicyIndex, retrieve

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = (
    "POLICY ASSISTANT: Answer only from the supplied context. "
    "If it does not contain the answer, say I could not find that information "
    "in the policy documents."
)


def answer(question, index=None):
    chunks = index.retrieve(question) if index is not None else retrieve(question)
    context = "\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    response = mock_llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"},
    ])
    return {
        "text": response.choices[0].message.content,
        "sources": [c["source"] for c in chunks],
    }


def evaluate(overlap):
    index = PolicyIndex(overlap=overlap)
    with (ROOT / "data/raw/policy_eval.csv").open(encoding="utf-8", newline="") as f:
        questions = list(csv.DictReader(f))
    rows = []
    for row in questions:
        result = answer(row["question"], index)
        answerable = row["expected_doc"] != "NONE"
        rows.append({
            **row, **result, "answerable": answerable,
            "retrieval_hit": answerable and row["expected_doc"] in result["sources"],
            "fact_hit": answerable and row["expected_fact"] in result["text"],
            "fallback_hit": not answerable and result["text"].startswith("I could not find"),
        })
    return {
        "overlap": overlap,
        "chunk_counts": dict(Counter(c["source"] for c in index.chunks)),
        "retrieval_hits": sum(r["retrieval_hit"] for r in rows),
        "fact_hits": sum(r["fact_hit"] for r in rows),
        "fallback_hits": sum(r["fallback_hit"] for r in rows),
        "questions": rows,
    }


def main():
    reports = [evaluate(10), evaluate(0)]
    normal, no_overlap = reports
    assert sum(normal["chunk_counts"].values()) == 15
    assert (normal["retrieval_hits"], normal["fact_hits"], normal["fallback_hits"]) == (12, 12, 2)
    assert no_overlap["fact_hits"] < 12
    assert all(c["source"] == "refund_policy.txt" for c in retrieve("refund policy late delivery"))
    output = ROOT / "outputs/policy_evaluation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    for report in reports:
        print(f"Overlap {report['overlap']}: {report['chunk_counts']}")
        print(f"Retrieval {report['retrieval_hits']}/12; facts {report['fact_hits']}/12; "
              f"fallbacks {report['fallback_hits']}/2")
        for row in report["questions"]:
            if row["answerable"] and not row["fact_hit"]:
                print(f"Miss: {row['question']}\nAnswer: {row['text']}")


if __name__ == "__main__":
    main()
