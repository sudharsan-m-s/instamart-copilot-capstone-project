from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

df_customers = pd.read_csv(RAW_DIR / "customers.csv")
df_products = pd.read_csv(RAW_DIR / "products.csv")
df_orders = pd.read_csv(RAW_DIR / "orders.csv")
df_tickets = pd.read_csv(RAW_DIR / "tickets.csv", keep_default_na=False)

tables = {
    "Customers": df_customers,
    "Products": df_products,
    "Orders": df_orders,
    "Tickets": df_tickets,
}

print("RAW DATA OVERVIEW & MISSING VALUES")


for name, df in tables.items():
    print(f"\n--- {name} ---")
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    missing_summary = pd.DataFrame({
        "Count": df.isnull().sum(),
        "Percentage": (df.isnull().mean() * 100).round(2),
    })
    print(missing_summary, "\n")

# Customer Audit
print("=" * 60)
print("CUSTOMER AUDIT")
print("=" * 60)
blank_age = df_customers["age"].isna().sum()
ages_num = pd.to_numeric(df_customers["age"], errors="coerce")
ages_outside = ((ages_num < 10) | (ages_num > 100)).sum()
distinct_cities = df_customers["city"].nunique()
nonblank_genders = df_customers["gender"].dropna()
nonblank_genders = nonblank_genders[nonblank_genders.str.strip() != ""]
distinct_genders = nonblank_genders.nunique()
blank_genders = df_customers["gender"].isna().sum() + (df_customers["gender"].str.strip() == "").sum()

print("Customers with blank age:", blank_age)
print("Ages outside 10-100:", ages_outside)
print("Distinct raw city spellings:", distinct_cities)
print("Distinct nonblank gender spellings:", distinct_genders)
print("Blank genders:", blank_genders)

# Order Audit
print("\n" + "=" * 60)
print("ORDER AUDIT")
print("=" * 60)
total_order_rows = len(df_orders)
exact_duplicates = df_orders.duplicated().sum()
pending_orders = (df_orders["status"] == "pending").sum()
distinct_orders = df_orders["order_id"].nunique()
non_numeric_amounts = pd.to_numeric(df_orders["order_amount"], errors="coerce").isna().sum()
negative_delivery = (pd.to_numeric(df_orders["delivery_minutes"], errors="coerce") < 0).sum()
blank_delivery = df_orders["delivery_minutes"].isna().sum()

print("Total rows:", total_order_rows)
print("Exact duplicates:", exact_duplicates)
print("Pending orders:", pending_orders)
print("Distinct order IDs:", distinct_orders)
print("Non-numeric order amounts:", non_numeric_amounts)
print("Negative delivery times:", negative_delivery)
print("Blank delivery times:", blank_delivery)
