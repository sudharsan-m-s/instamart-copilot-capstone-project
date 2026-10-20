from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "instamart.db"
SQL_PATH = PROJECT_ROOT / "sql" / "queries.sql"

QUERY_TITLES = [
    "Q1 - WHERE then HAVING (Completed orders > INR 500, customers with sum > INR 8000)",
    "Q2 - Three-table JOIN (Revenue by city and product category)",
    "Q3 - Scalar Subquery (Products priced above average catalogue price)",
    "Q4 - LEFT JOIN (Customers with no completed orders)",
    "Q5 - Common Table Expression (Customers with completed spend above cohort average)",
    "Q6 - Ranking Window Functions (City-wise spend ranks: RANK, DENSE_RANK, ROW_NUMBER)",
    "Q7 - Running Total (Daily completed revenue with cumulative running total)",
]

with open(SQL_PATH, "r", encoding="utf-8") as f:
    sql_text = f.read()

raw_chunks = [c.strip() for c in sql_text.split(";") if c.strip()]
queries = []
for chunk in raw_chunks:
    code_lines = [l for l in chunk.splitlines() if not l.strip().startswith("--") and l.strip()]
    if code_lines:
        queries.append("\n".join(code_lines))

print(f"EXECUTING {len(queries)} SQL QUERIES AGAINST {DB_PATH.name}")

with sqlite3.connect(DB_PATH) as con:
    for i, query in enumerate(queries):
        title = QUERY_TITLES[i] if i < len(QUERY_TITLES) else f"Query {i+1}"
        result = pd.read_sql_query(query, con)
        print(f"\n--- {title} ---")
        if len(result) > 10:
            print(result.head(5).to_string(index=False))
            print(f"... ({len(result) - 10} rows omitted) ...")
            print(result.tail(5).to_string(index=False))
        else:
            print(result.to_string(index=False))
        print(f"Rows returned: {len(result)}")

print("\n All queries executed successfully.")