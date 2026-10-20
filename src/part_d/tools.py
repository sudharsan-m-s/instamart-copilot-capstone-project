"""Refund tools. Every ledger entry is simulated; no money is transferred."""

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.part_d.retrieval import retrieve

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data/instamart.db"
LEDGER_PATH = ROOT / "outputs/refund_ledger.jsonl"


def get_order(order_id):
    """Read one order from the clean database without modifying it."""
    with sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ?", (order_id,),
        ).fetchone()
    return dict(row) if row else {"error": "order not found"}


def search_policy(query):
    """Return the three policy chunks closest to a query."""
    return {"results": retrieve(query)}


def calculate_refund(order_amount, delivery_minutes):
    """Calculate the late-delivery amount for an already verified completed order."""
    if not math.isfinite(order_amount) or order_amount < 0:
        raise ValueError("Order amount must be finite and non-negative.")
    if delivery_minutes is None or not math.isfinite(delivery_minutes) or delivery_minutes < 0:
        raise ValueError("A valid delivery duration is required.")
    return {"refund_inr": round(order_amount, 2) if delivery_minutes > 45 else 0.0}


def _record_action(tool, order_id, refund_inr):
    # Check the database again at the write boundary: model arguments are not authority.
    order = get_order(order_id)
    if "error" in order or order["status"] != "completed":
        raise ValueError("Only a completed order can receive a refund.")
    expected = calculate_refund(order["order_amount"], order["delivery_minutes"])["refund_inr"]
    if not math.isfinite(refund_inr) or expected <= 0 or abs(refund_inr - expected) > 0.001:
        raise ValueError("Requested amount does not match the refund rule.")
    if (tool == "issue_refund") != (expected <= 1000):
        raise ValueError("Refunds above INR 1000 require human approval.")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool, "order_id": order_id, "refund_inr": expected,
    }
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": expected}


def issue_refund(order_id, refund_inr):
    """Append an eligible automatic refund to the simulated ledger."""
    return _record_action("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    """Append a refund above INR 1000 for human review without issuing it."""
    return _record_action("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def _spec(name, properties):
    return {
        "name": name, "description": TOOL_MAP[name].__doc__,
        "parameters": {
            "type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False,
        },
    }


TOOLS_SPEC = [
    _spec("get_order", {"order_id": {"type": "string"}}),
    _spec("search_policy", {"query": {"type": "string"}}),
    _spec("calculate_refund", {
        "order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"},
    }),
    _spec("issue_refund", {
        "order_id": {"type": "string"}, "refund_inr": {"type": "number"},
    }),
    _spec("escalate_to_human", {
        "order_id": {"type": "string"}, "refund_inr": {"type": "number"},
    }),
]
