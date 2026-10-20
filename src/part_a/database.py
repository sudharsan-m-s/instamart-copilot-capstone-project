from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
DB_PATH = DATA_DIR / "instamart.db"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(CLEAN_DIR / "customers.csv")
if (CLEAN_DIR / "products.csv").exists():
    products = pd.read_csv(CLEAN_DIR / "products.csv")
else:
    products = pd.read_csv(RAW_DIR / "products.csv")
    products.to_csv(CLEAN_DIR / "products.csv", index=False)

orders = pd.read_csv(CLEAN_DIR / "orders.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
products.to_csv(CLEAN_DIR / "products.csv", index=False)
orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

with sqlite3.connect(DB_PATH) as con:
    customers.to_sql("customers", con, if_exists="replace", index=False)
    products.to_sql("products", con, if_exists="replace", index=False)
    orders.to_sql("orders", con, if_exists="replace", index=False)

    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

print("Database created/updated successfully at:", DB_PATH)
print("Tables in database:", tables)
print(f"Row counts -> customers: {len(customers)}, products: {len(products)}, orders: {len(orders)}")
