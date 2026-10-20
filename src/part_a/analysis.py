from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "instamart.db"

assert DB_PATH.exists(), f"Database not found at {DB_PATH}"

with sqlite3.connect(DB_PATH) as con:
    orders = pd.read_sql_query("SELECT * FROM orders", con)
    customers = pd.read_sql_query("SELECT * FROM customers", con)
    products = pd.read_sql_query("SELECT * FROM products", con)

completed = orders[orders["status"] == "completed"].copy()

data = completed.merge(customers[["customer_id", "city"]], on="customer_id")
data = data.merge(products[["product_id", "category"]], on="product_id")

# 1. Pandas Pivot Table Cross-Check (Rebuild Q2)
pivot = pd.pivot_table(
    data,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    margins_name="Grand Total",
    fill_value=0,
)

print("1. PANDAS PIVOT TABLE CROSS-CHECK (Revenue by City and Category)")
print(pivot.round(2))

sql_total = round(completed["order_amount"].sum(), 2)
pandas_total = round(pivot.loc["Grand Total", "Grand Total"], 2)

print(f"\nSQL completed revenue : INR {sql_total:,.2f}")
print(f"Pandas grand total    : INR {pandas_total:,.2f}")
assert sql_total == pandas_total, f"Mismatch: SQL {sql_total} != Pandas {pandas_total}"

# 2. City-Wise Aggregations
city_summary = data.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    mean_order_amount=("order_amount", "mean"),
    order_count=("order_amount", "count"),
)

print("2. CITY-WISE SUMMARY")
print(city_summary.round(2))

# 3. Descriptive Statistics & Outliers for Completed Order Amount
amount = completed["order_amount"].dropna()

mean_val = amount.mean()
median_val = amount.median()
skewness_val = amount.skew()
q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers_below = (amount < lower_fence).sum()
outliers_above = (amount > upper_fence).sum()

print("3. DESCRIPTIVE STATISTICS & OUTLIER DETECTION (order_amount)")
print(f"Mean order amount : INR {mean_val:.2f}")
print(f"Median order amount : INR {median_val:.2f}")
print(f"Skewness: {skewness_val:.2f} (right-skewed)")
print(f"Q1 (25th percentile): INR {q1:.2f}")
print(f"Q2 (50th percentile/Median) : INR {median_val:.2f}")
print(f"Q3 (75th percentile)        : INR {q3:.2f}")
print(f"IQR (Interquartile range)   : INR {iqr:.2f}")
print(f"Lower outlier fence         : INR {lower_fence:.2f}")
print(f"Upper outlier fence         : INR {upper_fence:.2f}")
print(f"Outliers below lower fence  : {outliers_below}")
print(f"Outliers above upper fence  : {outliers_above}")

# 4. Correlation Matrix
numeric_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr_matrix = completed[numeric_cols].corr()

print("4. CORRELATION MATRIX (Completed Orders)")
print(corr_matrix.round(3))


upper_tri = corr_matrix.where(
    pd.DataFrame(
        [[i < j for j in range(len(numeric_cols))] for i in range(len(numeric_cols))],
        index=numeric_cols,
        columns=numeric_cols,
    )
)
strongest_pair = upper_tri.stack().abs().idxmax()
strongest_val = corr_matrix.loc[strongest_pair[0], strongest_pair[1]]

print(
    f"\nStrongest absolute correlation: {strongest_pair[0]} and {strongest_pair[1]} "
    f"= {strongest_val:.3f}"
)


# 5. Interpretation (under 200 words)

interpretation = (
    "The dataset contains 11,110 completed quick-commerce orders across six metro cities. "
    "Order amounts exhibit a heavily right-skewed distribution (skewness 4.61), where the mean "
    "(INR 269.91) exceeds the median (INR 135.00) due to high-value purchases in categories such as "
    "Electronics. The middle fifty percent of orders fall between Q1 (INR 64.80) and Q3 (INR 297.00). "
    "Using the 1.5xIQR criterion, exactly 897 orders lie above the upper fence of INR 645.30 as "
    "legitimate high-basket outliers, with zero outliers below the lower fence. Linear relationships "
    "across variables remain negligible: the strongest absolute correlation is between order amount "
    "and discount percentage (-0.095), followed by order amount and quantity (+0.087). A correlation "
    "of this minimal magnitude does not establish causation."
)

word_count = len(interpretation.split())
print(f"5. BUSINESS INTERPRETATION ({word_count} words)")
print(interpretation)
