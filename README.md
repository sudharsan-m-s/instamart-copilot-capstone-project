# Instamart Customer Intelligence & Support Copilot

An enterprise data foundation, customer intelligence, and automated support triage copilot for Swiggy Instamart quick-commerce operations across six metropolitan cities (Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, and Pune).

---

## Overview

Swiggy Instamart delivers groceries, dairy, snacks, personal care, and electronics within minutes. As operational scale expands, data trust issues (dirty raw exports, duplicates, unparseable values, missing durations) and operational friction (customer churn, unstructured support tickets) emerge. 

This repository implements the **Customer Intelligence & Support Copilot** across four modular, interconnected phases:
- **Part A — Data Foundation (Clean, Store, Query)**: Audits dirty raw data, applies deterministic cleaning rules, persists verified relational tables in SQLite (`data/instamart.db`), and verifies business KPIs via SQL and pandas.
- **Part B — Customer Segments and Lapse Prediction**: Behavioral segmentation (RFM / K-Means) and churn/lapse risk modeling.
- **Part C — Support Ticket Triage with Guardrails**: PII redaction and automated ticket routing.
- **Part D — Policy RAG Assistant and Refund Agent**: Tool-augmented agent for automated policy question-answering and refund processing.

---

## Reproduce

Follow these numbered steps to reproduce the Part A data pipeline and analytical validations from a clean clone:

```bash
# 1. Clone repository and navigate to root
git clone https://github.com/sudharsan-m-s/instamart-copilot-capstone-project.git
cd instamart-copilot-capstone-project

# 2. Create and activate a Python virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate deterministic raw synthetic data
python generate_data.py

# 5. Audit raw data quality and plant anomalies
python src/part_a/audit.py

# 6. Clean customers and orders data
python src/part_a/clean_data.py

# 7. Create and populate SQLite database
python src/part_a/database.py

# 8. Run validation assertions (prints 'ALL CHECKPOINTS PASSED')
python src/part_a/validate.py

# 9. Execute all seven analytical SQL queries
python src/part_a/queries.py

# 10. Run pandas cross-check and exploratory data analysis
python src/part_a/analysis.py
```

---

## Part A — Data Foundation: Clean, Store, Query

### 1. Raw Data Quality Audit

The raw dataset exports were audited for missing values, erroneous formats, and structural anomalies:

| Audit Checkpoint | Raw Orders (`orders.csv`) | Raw Customers (`customers.csv`) | Reference Expected | Status |
| :--- | :--- | :--- | :--- | :--- |
| Total Rows | 13,084 | 1,200 | 13,084 / 1,200 | Verified |
| Exact Duplicate Rows | 119 | 0 | 119 | Verified |
| Pending Orders | 656 | N/A | 656 | Verified |
| Distinct Order IDs | 12,309 | N/A | 12,309 | Verified |
| Non-numeric Order Amounts | 120 | N/A | 120 | Verified |
| Negative Delivery Times | 106 | N/A | 106 | Verified |
| Blank Delivery Times | 2,541 | N/A | 2,541 | Verified |
| Blank Ages | N/A | 61 | 61 | Verified |
| Ages Outside [10, 100] | N/A | 6 | 6 | Verified |
| Distinct Raw City Spellings | N/A | 24 | 24 | Verified |
| Distinct Non-blank Genders | N/A | 10 | 10 | Verified |
| Blank Genders | N/A | 30 | 30 | Verified |

---

### 2. Data Cleaning Rules & Transformation Counts

#### A. Customers Cleaning (`clean_data.py`)
1. **City Standardisation**: Stripped leading/trailing whitespace and converted to title case. Standardised 24 inconsistent city spellings down to exactly 6 canonical cities:
   - Pune: 227
   - Delhi: 210
   - Bengaluru: 197
   - Chennai: 191
   - Hyderabad: 191
   - Mumbai: 184
2. **Gender Normalisation**: Stripped whitespace, lowered casing, mapped single-letter abbreviations (`m` -> `male`, `f` -> `female`), and assigned missing/unrecognized values to `unknown`.
   - Results: Female: 610, Male: 560, Unknown: 30.
3. **Age Validation & Imputation**: Flagged 6 impossible age values outside [10, 100] and 61 blank values (total 67 values). Imputed using the median customer age of the corresponding city (`groupby("city")["age"].transform("median")`).
   - Clean result: 0 missing ages; customer age range strictly bounded between 18 and 60.
4. **Signup Date Parsing**: Converted `signup_date` strings to ISO-8601 datetime format.

#### B. Orders Cleaning (`clean_data.py`)
1. **Timestamp Conversion**: Parsed `order_ts` into datetime objects.
2. **Exact Duplicate Removal**: Dropped exactly 119 redundant duplicate rows.
3. **Status Deduplication**: Resolved orders appearing multiple times across status stages (e.g., initial `pending` and subsequent `completed`/`cancelled`) by sorting by `order_ts` and retaining only the latest record per `order_id`. Eliminated 656 stale `pending` records, leaving exactly 12,309 unique orders (11,110 completed, 1,199 cancelled).
4. **Order Amount Imputation**: Converted `order_amount` to numeric, casting corrupted text (`N/A`, `error`) to null (120 records). Filled missing amounts using catalog prices:  
   `order_amount = round(price_inr * quantity * (1 - discount_pct / 100), 2)`  
   Zero missing order amounts remain.
5. **Delivery Duration Imputation**:
   - Replaced 106 negative delivery minutes with null.
   - For completed orders, filled missing delivery minutes with the median delivery duration of the respective product category.
   - For cancelled orders, kept all 1,199 delivery durations as null (as cancelled orders were never delivered).
   - Total completed revenue: **INR 2,998,668.80**.

---

### 3. Database Schema & Storage

The clean tables were persisted to SQLite at `data/instamart.db`:
- `customers` (1,200 rows): `customer_id`, `city`, `signup_date`, `age`, `gender`
- `products` (40 rows): `product_id`, `product_name`, `category`, `price_inr`
- `orders` (12,309 rows): `order_id`, `customer_id`, `product_id`, `quantity`, `order_ts`, `status`, `discount_pct`, `order_amount`, `delivery_minutes`

All timestamps in `orders` are stored as formatted text strings (`YYYY-MM-DD HH:MM:SS`). Running `validate.py` verifies all tables and constraints, returning:
```
============================================================
ALL CHECKPOINTS PASSED
============================================================
```

---

### 4. SQL Analysis Findings (`sql/queries.sql`)

| Query | SQL Technique | Description | Result / Checkpoint |
| :--- | :--- | :--- | :--- |
| **Q1** | `WHERE` then `HAVING` | High-value completed orders (> INR 500) grouped by customer with total spend > INR 8,000 | **30 customers** identified |
| **Q2** | 3-Table `JOIN` | Revenue broken down by customer city and product category | **30 rows**, Total revenue: **INR 2,998,668.80**; Largest cell: **Bengaluru / Electronics (INR 233,909.45)** |
| **Q3** | Scalar Subquery | Products priced above catalogue average price | **11 products** |
| **Q4** | `LEFT JOIN` | Customers who registered but have zero completed orders | **25 customers** |
| **Q5** | Common Table Expression (`CTE`) | Customers whose completed spend exceeds cohort average | **550 customers** (cohort average spend: **INR 2,552.06**) |
| **Q6** | Window Functions | City-partitioned spend ranking with `RANK()`, `DENSE_RANK()`, `ROW_NUMBER()` | Top per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10). Exactly **8 rows** differ between `RANK()` and `ROW_NUMBER()`. |
| **Q7** | Running Total | Daily revenue with cumulative running total over 183 days | **183 daily rows**, final cumulative revenue: **INR 2,998,668.80** |

> **Explanation of Q6 Tie Discrepancy**: When customers within the same city have identical spend, `RANK()` assigns them identical rank values while skipping subsequent ranks, whereas `ROW_NUMBER()` arbitrarily breaks ties to assign consecutive distinct integers.

---

### 5. Pandas Cross-Check & Exploratory Data Analysis (`analysis.py`)

#### A. Cross-Check with SQL
- Q2 rebuild via `pd.pivot_table(..., margins=True)` yields a Grand Total of **INR 2,998,668.80**, matching SQL exactly.
- City-wise summary of completed orders:
  - **Bengaluru**: Sum INR 523,643.40 | Mean INR 276.47 | Count 1,894
  - **Chennai**: Sum INR 463,629.15 | Mean INR 272.72 | Count 1,700
  - **Delhi**: Sum INR 502,829.80 | Mean INR 243.62 | Count 2,064
  - **Hyderabad**: Sum INR 528,589.90 | Mean INR 284.95 | Count 1,855
  - **Mumbai**: Sum INR 479,058.50 | Mean INR 297.00 | Count 1,613
  - **Pune**: Sum INR 500,918.05 | Mean INR 252.48 | Count 1,984

#### B. Distribution Statistics (Order Amount)
- **Mean**: INR 269.91
- **Median**: INR 135.00
- **Skewness**: 4.61 (heavily right-skewed)
- **Quartiles**: Q1 = INR 64.80, Q2 = INR 135.00, Q3 = INR 297.00
- **IQR**: INR 232.20
- **Tukey Fences**: Lower fence = INR -283.50, Upper fence = INR 645.30
- **Outliers**: Exactly 897 orders lie above the upper fence; 0 below the lower fence.

#### C. Correlation Matrix
| Metric | `order_amount` | `discount_pct` | `quantity` | `delivery_minutes` |
| :--- | :--- | :--- | :--- | :--- |
| **`order_amount`** | 1.000 | -0.095 | +0.087 | -0.003 |
| **`discount_pct`** | -0.095 | 1.000 | -0.000 | -0.010 |
| **`quantity`** | +0.087 | -0.000 | 1.000 | -0.013 |
| **`delivery_minutes`** | -0.003 | -0.010 | -0.013 | 1.000 |

*Strongest absolute correlation*: `order_amount` with `discount_pct` at **-0.095**.

#### D. Business Interpretation
The dataset contains 11,110 completed quick-commerce orders across six metro cities. Order amounts exhibit a heavily right-skewed distribution (skewness 4.61), where the mean (INR 269.91) exceeds the median (INR 135.00) due to high-value purchases in categories such as Electronics. The middle fifty percent of orders fall between Q1 (INR 64.80) and Q3 (INR 297.00). Using the 1.5xIQR criterion, exactly 897 orders lie above the upper fence of INR 645.30 as legitimate high-basket outliers, with zero outliers below the lower fence. Linear relationships across variables remain negligible: the strongest absolute correlation is between order amount and discount percentage (-0.095), followed by order amount and quantity (+0.087). A correlation of this minimal magnitude does not establish causation.

---

## Part B — Customer Segments and Lapse Prediction
*(Upcoming: Customer RFM feature engineering, K-Means segmentation, and Random Forest churn/lapse classification model).*

---

## Part C — Support Ticket Triage with Guardrails
*(Upcoming: Automated ticket categorization, regex-based PII redaction, token budgeting, and rule-based escalation).*

---

## Part D — Policy RAG Assistant and Refund Agent
*(Upcoming: TF-IDF vector retrieval over store policies and ReAct-style agentic refund decision workflows).*

---

## Limitations and Next Steps
1. **Linear Assumptions**: The exploratory analysis revealed negligible linear correlations among tabular features; non-linear models (gradient boosting, random forests) are preferable for Part B lapse prediction.
2. **Cold-Start Profiles**: 25 registered customers have made no completed orders; tailored onboarding incentives should target this group.
3. **Cross-Platform Portability**: All scripts utilize `pathlib.Path` relative to project root (`Path(__file__).resolve().parents[2]`), ensuring seamless execution across Linux, macOS, and Windows environments without hardcoded paths.

