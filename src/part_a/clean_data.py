from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

df_customers = pd.read_csv(RAW_DIR / "customers.csv")
df_products = pd.read_csv(RAW_DIR / "products.csv")
df_orders = pd.read_csv(RAW_DIR / "orders.csv")

# PART A - Task 3: Clean customers
print("=" * 60)
print("CLEANING CUSTOMERS")
print("=" * 60)

city_changes = (df_customers["city"] != df_customers["city"].str.strip().str.title()).sum()
original_gender = df_customers["gender"].copy()
# (a) Standardize city: strip whitespace and title-case
df_customers["city"] = df_customers["city"].str.strip().str.title()

# (b) Standardize gender: strip whitespace, lowercase, map abbreviations, map blanks/other to 'unknown'
df_customers["gender"] = df_customers["gender"].str.strip().str.lower()
df_customers["gender"] = df_customers["gender"].replace({"m": "male", "f": "female"})
df_customers["gender"] = df_customers["gender"].where(
    df_customers["gender"].isin(["male", "female"]), "unknown"
)

# (c) Handle invalid and missing ages
df_customers["age"] = pd.to_numeric(df_customers["age"], errors="coerce")
df_customers.loc[(df_customers["age"] < 10) | (df_customers["age"] > 100), "age"] = None
df_customers["age"] = df_customers["age"].fillna(
    df_customers.groupby("city")["age"].transform("median")
)

# (d) Convert signup_date to datetime
df_customers["signup_date"] = pd.to_datetime(df_customers["signup_date"])

# Save cleaned customers
df_customers.to_csv(CLEAN_DIR / "customers.csv", index=False)

print(f"Total cleaned customer rows: {len(df_customers)}")
print("\nCity distribution:")
print(df_customers["city"].value_counts())
print("\nGender distribution:")
print(df_customers["gender"].value_counts())
print("\nMissing ages:", df_customers["age"].isna().sum())
print(f"Age range: {df_customers['age'].min():.0f} to {df_customers['age'].max():.0f}")
print("City values changed:", city_changes)
print("Gender values changed:", (original_gender != df_customers["gender"]).sum())


# PART A - Task 4: Clean orders
print("\n" + "=" * 60)
print("CLEANING ORDERS")
print("=" * 60)

# 1. Convert order_ts to datetime
df_orders["order_ts"] = pd.to_datetime(df_orders["order_ts"])

# 2. Drop exact duplicate rows
exact_dupes_count = df_orders.duplicated().sum()
df_orders = df_orders.drop_duplicates()

# 3. Keep latest record per order_id
df_orders = df_orders.sort_values("order_ts")
df_orders = df_orders.drop_duplicates("order_id", keep="last")

# 4. Impute missing / unparseable order_amount from products catalog
df_orders = df_orders.merge(
    df_products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
)

df_orders["order_amount"] = pd.to_numeric(df_orders["order_amount"], errors="coerce")
missing_amounts_before = df_orders["order_amount"].isna().sum()
imputed_amounts = (
    df_orders["price_inr"] * df_orders["quantity"] * (1 - df_orders["discount_pct"] / 100)
).round(2)
df_orders["order_amount"] = df_orders["order_amount"].fillna(imputed_amounts)

# 5. Handle delivery times
df_orders["delivery_minutes"] = pd.to_numeric(df_orders["delivery_minutes"], errors="coerce")
negative_durations = (df_orders["delivery_minutes"] < 0).sum()
df_orders.loc[df_orders["delivery_minutes"] < 0, "delivery_minutes"] = None

# Completed orders: impute missing delivery times with category median
completed_mask = df_orders["status"].eq("completed")
missing_completed_durations = df_orders.loc[completed_mask, "delivery_minutes"].isna().sum()
category_medians = (
    df_orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)
df_orders.loc[completed_mask, "delivery_minutes"] = (
    df_orders.loc[completed_mask, "delivery_minutes"].fillna(category_medians)
)

# Cancelled orders: leave delivery_minutes as None
df_orders.loc[df_orders["status"].eq("cancelled"), "delivery_minutes"] = None

# Drop join helper columns
orders_clean = df_orders.drop(columns=["price_inr", "category"])

# Save cleaned orders and copy products to clean directory
orders_clean.to_csv(CLEAN_DIR / "orders.csv", index=False)
df_products.to_csv(CLEAN_DIR / "products.csv", index=False)

print(f"Total cleaned order rows: {len(orders_clean)}")
print("Exact duplicates removed:", exact_dupes_count)
print("Amounts imputed after deduplication:", missing_amounts_before)
print("Negative durations after deduplication:", negative_durations)
print("Completed durations imputed:", missing_completed_durations)
print("\nOrder status counts:")
print(orders_clean["status"].value_counts())
print("\nMissing order amounts:", orders_clean["order_amount"].isna().sum())
print("\nMissing delivery minutes by status:")
print(orders_clean.groupby("status")["delivery_minutes"].apply(lambda x: x.isna().sum()))

completed_revenue = round(
    orders_clean.loc[orders_clean["status"] == "completed", "order_amount"].sum(), 2
)
print(f"\nCompleted order revenue: INR {completed_revenue:,.2f}")
print(f"Cleaned datasets written to: {CLEAN_DIR}")
