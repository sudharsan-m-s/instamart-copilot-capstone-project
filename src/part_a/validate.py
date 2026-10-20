from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = DATA_DIR / "clean"
DB_PATH = DATA_DIR / "instamart.db"
SQL_PATH = PROJECT_ROOT / "sql" / "queries.sql"

print("=" * 60)
print("RUNNING PART A VALIDATION CHECKPOINTS")
print("=" * 60)

# 1. Customer Checkpoints (CSV)
customers = pd.read_csv(CLEAN_DIR / "customers.csv")

assert len(customers) == 1200, f"Expected 1200 customers, got {len(customers)}"
assert customers["gender"].value_counts().get("female", 0) == 610, "Female count != 610"
assert customers["gender"].value_counts().get("male", 0) == 560, "Male count != 560"
assert customers["gender"].value_counts().get("unknown", 0) == 30, "Unknown count != 30"
assert customers["age"].isna().sum() == 0, "Missing ages found"
assert customers["age"].min() >= 18, "Minimum age < 18"
assert customers["age"].max() <= 60, "Maximum age > 60"

expected_cities = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184,
}
for city, expected_count in expected_cities.items():
    actual_count = customers["city"].value_counts().get(city, 0)
    assert actual_count == expected_count, f"City {city}: expected {expected_count}, got {actual_count}"

print("[PASS] Clean customers checkpoints passed (1200 rows, gender, age, cities).")

# 2. Order Checkpoints (CSV)
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

assert len(orders) == 12309, f"Expected 12309 orders, got {len(orders)}"
assert (orders["status"] == "completed").sum() == 11110, "Completed orders != 11110"
assert (orders["status"] == "cancelled").sum() == 1199, "Cancelled orders != 1199"
assert not (orders["status"] == "pending").any(), "Pending orders found"
assert orders["order_amount"].isna().sum() == 0, "Missing order amounts found"

completed = orders["status"] == "completed"
cancelled = orders["status"] == "cancelled"

assert orders.loc[completed, "delivery_minutes"].isna().sum() == 0, "Missing delivery minutes in completed orders"
assert orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199, "Non-missing delivery minutes in cancelled orders"

completed_revenue = round(orders.loc[completed, "order_amount"].sum(), 2)
assert completed_revenue == 2998668.80, f"Expected revenue 2998668.80, got {completed_revenue}"

print(f"[PASS] Clean orders checkpoints passed (12309 rows, status counts, delivery times, INR {completed_revenue:,.2f}).")

# 3. SQLite Database Checkpoints
assert DB_PATH.exists(), f"Database not found at {DB_PATH}"

with sqlite3.connect(DB_PATH) as con:
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    db_tables = {row[0] for row in cursor.fetchall()}
    assert {"customers", "products", "orders"}.issubset(db_tables), f"Missing tables in DB: {db_tables}"

    db_cust_count = cursor.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    db_prod_count = cursor.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    db_ord_count = cursor.execute("SELECT COUNT(*) FROM orders").fetchone()[0]

    assert db_cust_count == 1200, f"DB customers count != 1200, got {db_cust_count}"
    assert db_prod_count == 40, f"DB products count != 40, got {db_prod_count}"
    assert db_ord_count == 12309, f"DB orders count != 12309, got {db_ord_count}"

print("[PASS] SQLite database tables and row counts verified.")

# 4. SQL Queries Execution & Checkpoints
assert SQL_PATH.exists(), f"SQL file not found at {SQL_PATH}"

with open(SQL_PATH, "r", encoding="utf-8") as file:
    sql_text = file.read()

# Parse individual queries separated by semicolon
raw_chunks = [c.strip() for c in sql_text.split(";") if c.strip()]
queries = []
for chunk in raw_chunks:
    code_lines = [l for l in chunk.splitlines() if not l.strip().startswith("--") and l.strip()]
    if code_lines:
        queries.append("\n".join(code_lines))

assert len(queries) == 7, f"Expected 7 queries in {SQL_PATH}, found {len(queries)}"

with sqlite3.connect(DB_PATH) as con:
    # Q1: WHERE then HAVING -> 30 customers
    df_q1 = pd.read_sql_query(queries[0], con)
    assert len(df_q1) == 30, f"Q1 expected 30 customers, got {len(df_q1)}"

    # Q2: Three-table join -> 30 rows, revenue 2998668.80, Bengaluru / Electronics = 233909.45
    df_q2 = pd.read_sql_query(queries[1], con)
    assert len(df_q2) == 30, f"Q2 expected 30 rows, got {len(df_q2)}"
    q2_total = round(df_q2["revenue"].sum(), 2)
    assert q2_total == 2998668.80, f"Q2 expected revenue 2998668.80, got {q2_total}"
    beng_elec = df_q2.loc[(df_q2["city"] == "Bengaluru") & (df_q2["category"] == "Electronics"), "revenue"].iloc[0]
    assert round(beng_elec, 2) == 233909.45, f"Q2 Bengaluru / Electronics expected 233909.45, got {beng_elec}"

    # Q3: Scalar subquery -> 11 products
    df_q3 = pd.read_sql_query(queries[2], con)
    assert len(df_q3) == 11, f"Q3 expected 11 products, got {len(df_q3)}"

    # Q4: LEFT JOIN -> 25 customers
    df_q4 = pd.read_sql_query(queries[3], con)
    assert len(df_q4) == 25, f"Q4 expected 25 customers, got {len(df_q4)}"

    # Q5: CTE -> 550 customers
    df_q5 = pd.read_sql_query(queries[4], con)
    assert len(df_q5) == 550, f"Q5 expected 550 customers, got {len(df_q5)}"

    # Q6: Ranking window functions -> top customer per city, 8 rows rank != row_number
    df_q6 = pd.read_sql_query(queries[5], con)
    diff_rank_row = (df_q6["rank_spend"] != df_q6["row_num_spend"]).sum()
    assert diff_rank_row == 8, f"Q6 expected 8 rank!=row_number rows, got {diff_rank_row}"

    # Q7: Cumulative revenue -> 183 daily rows, final cumulative revenue INR 2,998,668.80
    df_q7 = pd.read_sql_query(queries[6], con)
    assert len(df_q7) == 183, f"Q7 expected 183 rows, got {len(df_q7)}"
    last_cum_rev = round(df_q7.iloc[-1]["cumulative_revenue"], 2)
    assert last_cum_rev == 2998668.80, f"Q7 expected last cumulative revenue 2998668.80, got {last_cum_rev}"

print("[PASS] All 7 SQL queries executed and matched reference results.")

print("ALL CHECKPOINTS PASSED")
