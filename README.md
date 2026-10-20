# Instamart Customer Intelligence & Support Copilot

A local Python project for cleaning order data, finding customer segments, routing support tickets and handling simulated refunds.

---

## Overview

The project uses synthetic data for six cities: Bengaluru, Chennai, Delhi, Hyderabad, Mumbai and Pune. The Instamart name provides the scenario only; none of the customers, transactions or policies are real. All amounts are in INR. Every model call uses the supplied offline `mock_llm.py`; the project needs no API key, account or paid service.

This repository implements the **Customer Intelligence & Support Copilot** across four modular, interconnected phases:
- **Part A — Data Foundation (Clean, Store, Query)**: Audits dirty raw data, applies deterministic cleaning rules, persists verified relational tables in SQLite (`data/instamart.db`), and verifies business KPIs via SQL and pandas.
- **Part B — Customer Segments and Lapse Prediction**: Behavioral segmentation (RFM / K-Means / PCA) and churn/lapse risk modeling with leakage-free preprocessing and class balancing.
- **Part C — Support Ticket Triage with Guardrails**: Input guardrails, regex-based PII redaction, resilient LLM calls with jitter, zero-shot vs few-shot categorization, token budgeting, and segment-aware routing.
- **Part D — Policy RAG Assistant and Refund Agent**: Tool-augmented agent for automated policy question-answering and refund processing.

---

## Reproduce

Run from the repository root on branch `main`. The tested interpreter is Python 3.13.5; use Python 3.13 for the pinned environment in `requirements.txt`. SQLite is bundled with Python. The requirements were recorded from `pip freeze`, keeping the four direct libraries (pandas, numpy, scikit-learn and joblib) and their runtime dependencies. No plotting library is used.

1. Create a virtual environment.

```text
python -m venv .venv
```

2. Activate it. In Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

   In Linux/macOS, use `source .venv/bin/activate`. If PowerShell activation is restricted, use `.\.venv\Scripts\python.exe` in place of `python` below.

3. Install the pinned dependencies. Only this installation step needs package downloads.

```text
python -m pip install -r requirements.txt
```

4. Rebuild Parts A–D and run the validation tests.

```text
python -m src.reproduce
```

The runner executes the following in order, stopping on the first failure. These are also the individual commands for inspecting one stage:

```text
python generate_data.py
python -m src.part_a.audit
python -m src.part_a.clean_data
python -m src.part_a.database
python -m src.part_a.validate
python -m src.part_a.queries
python -m src.part_a.analysis
python -m src.part_b.segmentation
python -m src.part_b.lapse_model
python -m src.part_b.evaluation
python -m src.part_c.privacy
python -m src.part_c.triage
python -m src.part_d.policy_assistant
python -m src.part_d.refund_agent
python -m unittest discover -s tests -v
```

The generator's final line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The two supplied Python files are unchanged from the brief. A successful full run ends with `ALL FOUR PARTS REPRODUCED SUCCESSFULLY`. Console logs are saved under `outputs/logs/` and ignored by Git; the required results, model and checkpoints are kept in the project. Retry waits use bounded jitter; ledger timestamps and elapsed times naturally change between runs.

Running the Part D demonstration regenerates `agent_runs.jsonl`, `refund_ledger.jsonl` and the ten `state/demo-*.json` checkpoints. Do not use that demonstration command to preserve an independent ledger. Calling `run_agent(...)` directly appends results and uses a new run id.

Repository map: `src/part_a/` handles cleaning and SQL checks, `src/part_b/` handles segmentation and prediction, `src/part_c/` handles ticket privacy and routing, and `src/part_d/` contains retrieval, the policy assistant, tools and agent. `tests/` covers guardrails and recovery. All findings are reported below as text; no chart or image is required.

---

## Part A — Data Foundation: Clean, Store, Query

### 1. Raw Data Quality Audit

The raw dataset exports were audited for missing values, erroneous formats, and structural anomalies:

Shapes are customers **(1,200, 5)**, products **(40, 4)** and orders **(13,084, 9)**. The table below reports every raw column using pandas' default CSV missing-value interpretation. In particular, the literal `N/A` is read as missing; `error` remains text until numeric conversion.

| Table | Column | Inferred dtype | Missing | Missing % |
| :--- | :--- | :--- | ---: | ---: |
| customers | customer_id | str | 0 | 0.00 |
| customers | city | str | 0 | 0.00 |
| customers | signup_date | str | 0 | 0.00 |
| customers | age | float64 | 61 | 5.08 |
| customers | gender | str | 30 | 2.50 |
| products | product_id | str | 0 | 0.00 |
| products | product_name | str | 0 | 0.00 |
| products | category | str | 0 | 0.00 |
| products | price_inr | int64 | 0 | 0.00 |
| orders | order_id | str | 0 | 0.00 |
| orders | customer_id | str | 0 | 0.00 |
| orders | product_id | str | 0 | 0.00 |
| orders | quantity | int64 | 0 | 0.00 |
| orders | order_ts | str | 0 | 0.00 |
| orders | status | str | 0 | 0.00 |
| orders | discount_pct | int64 | 0 | 0.00 |
| orders | order_amount | str | 53 | 0.41 |
| orders | delivery_minutes | float64 | 2,541 | 19.42 |

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
1. **City Standardisation**: Stripped leading/trailing whitespace and converted to title case, changing 476 values. Standardised 24 inconsistent city spellings down to exactly 6 canonical cities:
   - Pune: 227
   - Delhi: 210
   - Bengaluru: 197
   - Chennai: 191
   - Hyderabad: 191
   - Mumbai: 184
2. **Gender Normalisation**: Stripped whitespace, lowered casing, mapped single-letter abbreviations (`m` -> `male`, `f` -> `female`), and assigned missing/unrecognized values to `unknown`. This changes 950 entries, including the 30 blanks.
   - Results: Female: 610, Male: 560, Unknown: 30.
3. **Age Validation & Imputation**: Flagged 6 impossible age values outside [10, 100] and 61 blank values (total 67 values). Imputed using the median customer age of the corresponding city (`groupby("city")["age"].transform("median")`).
   - Clean result: 0 missing ages; customer age range strictly bounded between 18 and 60.
4. **Signup Date Parsing**: Converted all 1,200 `signup_date` strings to datetimes; none failed parsing.

#### B. Orders Cleaning (`clean_data.py`)
1. **Timestamp Conversion**: Parsed all 13,084 `order_ts` strings into datetime objects; none failed parsing.
2. **Exact Duplicate Removal**: Dropped exactly 119 redundant duplicate rows.
3. **Status Deduplication**: Resolved orders appearing multiple times across status stages (e.g., initial `pending` and subsequent `completed`/`cancelled`) by sorting by `order_ts` and retaining only the latest record per `order_id`. Eliminated 656 stale `pending` records, leaving exactly 12,309 unique orders (11,110 completed, 1,199 cancelled).
4. **Order Amount Imputation**: After deduplication, converted `order_amount` to numeric and repaired **112** unparseable amounts (the raw audit counted 120, including duplicates). Filled missing amounts using catalog prices:
   `order_amount = round(price_inr * quantity * (1 - discount_pct / 100), 2)`
   Zero missing order amounts remain.
5. **Delivery Duration Imputation**:
   - Replaced **105** negative delivery minutes with null after deduplication (106 in the raw audit).
   - For completed orders, filled **774** missing delivery minutes, including those negatives, with the median delivery duration of the respective product category.
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
ALL CHECKPOINTS PASSED
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

### 1. RFM Table Construction (`segmentation.py`)
- **Snapshot Date**: `2025-10-01 00:00:00`
- **Cohort**: 1,175 customers with >= 1 completed orders
- **Recency**: Whole days between snapshot and latest completed order (`median = 19.0`, `min = 0`, `max = 180`)
- **Frequency**: Distinct completed order count (`mean = 9.46`, `max = 33`)
- **Monetary**: Total completed spend (`mean = INR 2,552.06`, `max = INR 14,766.55`)

### 2. K-Means Clustering Evaluation & Elbow Analysis
Standardised RFM features with `StandardScaler` and evaluated K in [2, 8] with `KMeans(n_init=10, random_state=42)`:

| K | Inertia | Silhouette Score | Business Feasibility & Decision |
| :---: | :---: | :---: | :--- |
| 2 | 1820.8 | 0.465 | Under-segmented (blends premium and casual customers) |
| 3 | 1268.2 | 0.445 | Moderate cohesion, misses at-risk distinction |
| **4** | **743.1** | **0.526** | **Optimal**: Peak silhouette score, distinct elbow, and clean business interpretation |
| 5 | 617.4 | 0.480 | Declining silhouette; over-fragments loyal segment |
| 6 | 503.2 | 0.482 | Marginal gain |
| 7 | 408.5 | 0.416 | Severe degradation in silhouette |
| 8 | 366.5 | 0.412 | Excessive fragmentation |

> **Inertia Explanation**: Inertia measures total within-cluster sum of squared distances. As K increases, cluster boundaries shrink closer to data points by definition, reaching zero when K=N. Therefore, decreasing inertia alone cannot identify the optimal K; the peak silhouette score at K=4 (0.526) confirms the strongest cluster separation.

### 3. Segment Profiles, Actionable Names, and PCA Projection
Output saved to `outputs/customer_segments.csv`:

| Segment ID | Segment Name | Is Premium | Size | Mean Recency (days) | Mean Frequency | Mean Monetary (INR) | Recommended Action |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | Occasional Shoppers | False | 436 | 28.9 | 3.8 | 1,057.62 | Deliver personalized category coupon nudges |
| **1** | **Premium Spenders** | **True** | **138** | **24.1** | **6.8** | **7,129.70** | **VIP priority delivery and dedicated premium desk routing** |
| **2** | Frequent Shoppers | False | 419 | 8.8 | 19.4 | 3,410.02 | Automated recurring daily essentials subscriptions |
| **3** | At-Risk / Lapsed | False | 182 | 109.0 | 2.2 | 685.99 | High-incentive win-back reactivation discounts |

- **PCA Explained Variance Ratio**: Component 1 = **0.634**, Component 2 = **0.218**, Total = **0.852** (85.2% variance captured in 2D).

Cluster numbers are labels only and have no ordering. K=4 is also small enough to support four distinct campaign actions; K=8 lowers inertia without giving equally useful additional groups. Lower inertia with more centroids is expected because additional centres can reduce within-cluster distances, so it cannot select K on its own.

### 4. Lapse Modeling Without Data Leakage (`lapse_model.py`)
- **Cutoff Date**: `2025-09-01 00:00:00`
- **Cohort**: 1,156 customers with completed orders before cutoff
- **Target Label (`lapsed`)**: 1 if 0 completed orders between 2025-09-01 and 2025-09-30, else 0.
  - Lapsed: **443** (38.32%) | Non-lapsed: **713** (61.68%)
- **Split**: 80/20 stratified (`random_state=42`): Train = 924 rows, Test = 232 rows.
- **Majority Baseline**: Predicting 0 (not lapsed) achieves **61.6% accuracy** and **0.000 F1**.
- **Preprocessing Guardrails**:
  - `OneHotEncoder(drop='first')` and `StandardScaler` sit inside each model's `Pipeline` for `GridSearchCV`. They fit only the training portion of each fold; the final refit uses the 924 training rows. No encoder, scaler or resampler sees the test rows.
  - **Learned scaler mean of `frequency`**: 8.1147 vs **Full-cohort mean**: 8.0285. *They differ because the training set excludes the 232 test samples to guarantee zero leakage.*
  - Verified 0 missing feature values.

Features are recency, frequency, monetary, mean discount, cancellation rate (cancelled / all final orders), mean delivery duration, most frequent product category, city, age, gender and tenure in days. All order aggregation uses dates strictly before 2025-09-01; no September orders enter feature aggregation, and segment ids/names are not model features. The label uses September completed orders only. Ties in favourite category are resolved using the first sorted mode.

### 5. Model Tuning & Test Set Performance

| Model | Hyperparameter Search Grid | Best Parameters | 5-Fold CV F1 | Test Precision | Test Recall | Test F1 (Lapsed Class) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | C in {0.01, 0.1, 1, 10, 100} | C = 100 | 0.651 | 0.644 | 0.652 | 0.648 |
| **Random Forest** | max_depth in {3, 5, 8}, min_samples_leaf in {5, 10, 20} | max_depth=8, min_samples_leaf=10 | **0.681** | **0.659** | **0.652** | **0.655** |

- **Random Forest Confusion Matrix on Test Set**:
  - TN: 113, FP: 30
  - FN: 31, TP: 58
- **Logistic Regression Confusion Matrix on Test Set**: TN 111, FP 32, FN 31, TP 58. Matrix rows are true classes [0, 1] and columns are predicted classes [0, 1] for both models.

The random forest is selected because its cross-validated F1 is higher, with test F1 0.655 confirming acceptable performance after tuning. Both models miss 31 lapsing customers, but the forest incorrectly targets 30 loyal customers versus 32 for logistic regression. Missing a lapsing customer can lose repeat business, while targeting a loyal customer spends an unnecessary offer. The balanced logistic model below finds more lapsing customers but also creates more false positives; campaign costs would determine whether that trade-off is preferable.

### 6. Training Class-Balance Experiment
Oversampled the minority lapsed class in the training split up to 570 instances (`random_state=42`):
- **Unbalanced Logistic Regression**: Precision = 0.644, Recall = 0.652, F1 = 0.648
- **Balanced Logistic Regression**: Precision = 0.589, **Recall = 0.854** (+20.2 percentage points), F1 = 0.697
- **Retention Impact**: Recall increased most dramatically, enabling the retention team to intercept ~85.4% of churning customers rather than missing over a third of them.

### 7. Feature Importances & Model Persistence
- **Top 5 Features (Random Forest)**:
  1. `frequency`: 0.2796 (Habitual ordering frequency is the strongest retention signal)
  2. `monetary`: 0.2712 (Cumulative investment in the platform reflects customer stickiness)
  3. `recency`: 0.1151 (Lapse latency indicates declining engagement)
  4. `avg_delivery_minutes`: 0.0613 (Delivery experience helps separate predicted outcomes)
  5. `cancellation_rate`: 0.0523 (Past cancellation behaviour contributes to predictions)
- **Logistic Regression Coefficients**:
  - `recency` (+0.0396): Greater inactivity is associated with higher lapse log-odds, holding other features fixed.
  - `frequency` (-1.6982): More completed orders are associated with lower lapse log-odds.
  - `monetary` (+0.2846): Higher spend is associated with higher lapse log-odds after accounting for frequency and the other features.
  - Numeric coefficients apply to standardised features. Neither coefficients nor forest importance establish causation; importance alone does not show a direction.
- **Model Persistence**: Saved to `outputs/lapse_model.joblib`. Reloading and testing verified 100% identical test predictions.

### 8. High-Risk Premium Customer Retention List
Top 10 at-risk customers from the Premium segment saved to `outputs/retention_list.csv`:

| Customer ID | Predicted Lapse Probability | Actually Lapsed |
| :---: | :---: | :---: |
| C0862 | 0.4910 | 0 |
| C1112 | 0.4841 | 0 |
| C0060 | 0.4798 | 0 |
| C0802 | 0.4796 | 0 |
| C1008 | 0.4778 | 0 |
| C1000 | 0.4624 | 0 |
| C0525 | 0.4597 | 1 |
| C0133 | 0.4562 | 0 |
| C0267 | 0.4525 | 0 |
| C0167 | 0.4522 | 0 |

> **Leakage Statement**: Exactly 1 of the top 10 truly lapsed. The `is_premium` flag is derived from whole-window RFM (including September transactions), making this retention list an illustrative post-hoc demonstration rather than a strict leakage-free evaluation.

---

## Part C — Support Ticket Triage with Guardrails

### 1. Privacy & Input Guardrails (`privacy.py`)
Executed across all 164 raw support tickets (`data/raw/tickets.csv`):
- **Input Guardrails**:
  - `skipped_empty`: **4 tickets** (empty or whitespace only, never sent to LLM)
  - `flagged_long`: **4 tickets** (estimate = 734 tokens > 400 `MAX_INPUT_TOKENS`, never sent to LLM)
- **PII Redaction**:
  - Emails masked: **55** (`[\w.+-]+@[\w-]+\.[\w.]+` -> `[EMAIL]`)
  - Phone numbers masked: **91** (`\b[6-9]\d{9}\b` -> `[PHONE]`)
  - Order IDs masked: **132** (`ORD-\d{6}` -> `[ORDER_ID]`)
  - Total substitutions: **278** | Unmasked PII remaining: **0** (verified by assertion)

### 2. LLM Classifier & Resilient Retries (`classifier.py`)
- **Prompts**:
  - **Zero-Shot**: Strictly specifies JSON output with allowed categories (`billing`, `delivery`, `technical`, `fraud`) without any `EXAMPLES:`.
  - **Few-Shot**: Includes labeled demonstrations with an explicit fraud report.
- **Resilience**:
  - Caught exactly **19 simulated `TransientLLMError` failures** during each full run.
  - Successfully recovered 100% of failures using exponential backoff with jitter (wait times strictly within 0.045s to 0.0575s). Zero unhandled errors.
- **Output Parsing & Fallback**:
  - **135** `classified_json` (valid JSON category)
  - **12** `classified_keyword` (chatty output resolved via lowercase regex word extraction)
  - **9** `flagged_invalid_output` (routed to human review; no guessing)

### 3. Zero-Shot vs Few-Shot Comparison (`triage.py`)

| Metric | Zero-Shot Pipeline | Few-Shot Pipeline | Delta / Business Impact |
| :--- | :---: | :---: | :--- |
| **Classified Tickets** | 147 | 147 | Identical coverage |
| **Accuracy (%)** | **85.7%** (126 / 147) | **100.0%** (147 / 147) | **+14.3% accuracy boost** |
| **Total Tokens Consumed** | 11,584 | 21,100 | +9,516 tokens (+82%) |
| **Tokens per Classified Ticket** | 78.8 | 143.5 | +64.7 tokens/ticket |

#### Zero-Shot Confusion Matrix:
| True / Pred | Pred Billing | Pred Delivery | Pred Technical | Pred Fraud |
| :--- | :---: | :---: | :---: | :---: |
| **True Billing** | 38 | 0 | 0 | 0 |
| **True Delivery** | 0 | 36 | 0 | 0 |
| **True Technical** | **10** | 0 | 29 | 0 |
| **True Fraud** | **11** | 0 | 0 | 23 |

Zero-shot prompting misclassifies 11 fraud tickets and 10 technical tickets as billing issues because of overlapping keywords. Few-shot prompting uses an extra 9,516 tokens overall, or 64.7 per classified ticket, and corrects all 21 errors in this synthetic run. This extra context is worthwhile when missing fraud is costly, although the result reflects the mock's fixed keyword rules and does not establish real-model performance. For a limited budget, zero-shot is cheaper but its 85.7% accuracy needs to be weighed against those missed fraud cases.

### 4. Token Budget Demonstration
- Evaluated zero-shot pipeline with `TOKEN_BUDGET = 6000`:
  - Stopped safely after 6,029 tokens.
  - Exactly **75 tickets** were halted with status `budget_exhausted` and gracefully routed to human review.
  - Results saved to `outputs/triage_budget_demo.csv`.

### 5. Production Routing Results
Generated from the 100,000-token few-shot run and persisted to `outputs/triage_results.csv` (164 tickets):

| Route Target | Assigned Priority | Ticket Count | Description |
| :--- | :---: | :---: | :--- |
| `support_queue` | NORMAL | 102 | Standard non-premium billing, delivery, and technical issues |
| `security_team` | HIGH | 34 | All confirmed fraud tickets |
| `human_review` | NORMAL | 17 | Guardrail exceptions (4 empty, 4 long, 9 invalid output) |
| `premium_desk` | HIGH | 11 | High-value VIP customer tickets mapped via Part B segments |
| **Total** | | **164** | Fully routed pipeline |

---

## Part D — Policy RAG Assistant and Refund Agent

### 1. Policy chunks and retrieval

`retrieval.py` reads the five policy files in filename order, splits on whitespace and makes 40-word chunks with 10-word overlap. Starts are 0, 30, 60 and so on; it stops as soon as the chunk reaches the document end.

| Source | Chunks with overlap 10 | Chunks with overlap 0 |
| :--- | ---: | ---: |
| refund_policy.txt | 4 | 3 |
| delivery_policy.txt | 3 | 3 |
| escalation_policy.txt | 3 | 2 |
| payment_policy.txt | 3 | 3 |
| privacy_policy.txt | 2 | 2 |
| Total | 15 | 13 |

`TfidfVectorizer()` fits the chunk texts once, then queries use the same vectorizer and cosine similarity; `retrieve(query, k=3)` returns source, score and text. For `refund policy late delivery`, all three retrieved chunks come from `refund_policy.txt`.

TF measures how often a term appears in a chunk; IDF gives more weight to terms appearing in fewer chunks. TF-IDF therefore rewards shared distinctive words, but may miss two phrases with the same meaning and different vocabulary.

### 2. Grounded answers and evaluation

`policy_assistant.answer(question)` supplies the three retrieved chunks to the offline mock with a `POLICY ASSISTANT` system instruction to use only context and return the fallback when information is absent. The user message is `CONTEXT:` followed by source-prefixed chunks, then `QUESTION:` followed by the question. Its result includes the answer text and all retrieved source names, including repeated filenames.

| Evaluation on policy_eval.csv | Overlap 10 | Overlap 0 |
| :--- | ---: | ---: |
| Expected document in top three, answerable questions | 12/12 | 12/12 |
| Expected fact in answer, case-sensitive substring | 12/12 | 10/12 |
| Unsupported questions answered with `I could not find...` | 2/2 | 2/2 |

With zero overlap, “What delivery time makes a completed order eligible for a full refund?” returns “receives a full refund of the order amount.” and loses `more than 45 minutes`. “How soon is money returned when a payment fails but is deducted?” also loses `48 hours`. In both cases an answer-bearing sentence crosses a 40-word boundary; finding the correct document alone does not keep its complete fact together. The 10-word overlap preserves the relevant sentence in an available chunk. Per-question text, sources and scores of the evaluation checks are recorded in `outputs/policy_evaluation.json` (the file records pass/fail checks; retrieval scores are returned by `retrieve`).

### 3. Tools and refund decisions

`tools.py` provides a `TOOLS_SPEC` list containing each tool's name, description and JSON-style parameter specification, and `TOOL_MAP` maps each name to its Python function.

| Tool | Inputs | Behaviour |
| :--- | :--- | :--- |
| get_order | order_id | Parameterised, read-only SQLite lookup; returns the four requested order fields or `order not found` |
| search_policy | query | Returns `retrieve(query)` under `results` |
| calculate_refund | order_amount, delivery_minutes | Returns the full amount only above 45 minutes; otherwise zero |
| issue_refund | order_id, refund_inr | Revalidates the completed order and amount, then appends a simulated action up to INR 1000 |
| escalate_to_human | order_id, refund_inr | Revalidates the order, then logs amounts above INR 1000 for approval without issuing them |

Cancelled orders never reach refund calculation in the normal agent flow. Write tools also check eligibility against the database, so forged model arguments cannot bypass the completed-status rule or the INR 1000 limit. A missing/invalid duration is rejected. Requests are assumed to be within the fictional seven-day window as specified by the brief.

### 4. ReAct scenario results

`run_agent` starts with an `AGENT` system message and the user request. Each assistant tool call is recorded with its id, name and arguments; the matching observation is appended as a JSON tool message before the next model call. Replies without tool calls supply the final status and answer.

| Scenario | Status | Refund/request amount (INR) | Model calls | Tokens |
| :--- | :--- | ---: | ---: | ---: |
| S1 | REFUND | 128.25 | 5 | 1,087 |
| S2 | REFUND | 135.00 | 5 | 1,087 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,074 |
| S4 | DENIED | 0.00 | 4 | 757 |
| S5 | DENIED | 0.00 | 3 | 451 |
| S6 | NOT_FOUND | 0.00 | 2 | 148 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,228 |

S3's amount is an approval request, not an issued refund. S4 is on time, S5 is cancelled, and S6 is absent from the database. S7 deliberately loops on policy searches and finishes all six calls under the default 6,000-token limit.

`outputs/agent_runs.jsonl` has ten lines: the seven scenarios, the token and timeout demonstrations, and the completed crash/resume run. Each line includes scenario id, run id, final status/answer, calls, tokens and every tool's arguments and observation. `outputs/refund_ledger.jsonl` has four timestamped entries: S1 and S2 refunds, S3 escalation, and the separate S1 crash/resume demonstration's refund. These are local simulations.

### 5. Safe stopping and resume

| Demonstration | Result | Calls | Tokens | Completed tools |
| :--- | :--- | ---: | ---: | :--- |
| S1 with token_limit=200 | FALLBACK_TOKEN_LIMIT | 3 | 468 | get_order, search_policy |
| S1 with time_limit_seconds=0 | FALLBACK_TIMEOUT | 0 | 0 | none |
| S7 with defaults | FALLBACK_MAX_ITERATIONS | 6 | 3,228 | six search_policy calls |

Token usage is added after every reply. When the total exceeds the limit, the newly requested tool is not executed; the token demonstration therefore never executes `calculate_refund`. Time is measured with `time.monotonic()` and checked at the top of each iteration, including before the first call. Every fallback returns the work done so far and a short human-handoff summary.

After each tool observation, an atomic file replacement saves `state/<run_id>.json` with the complete messages, token and call totals, step number, completed tools, pending calls and limits. The S1 demonstration raises `SimulatedCrash` after the second saved observation and calls `resume_agent("demo-resume")`. The execution sequence across both portions is:

```text
Before crash: get_order -> search_policy
After resume: calculate_refund -> issue_refund
Final result: REFUND, INR 128.25; 5 total model calls; 1,087 tokens.
```

There are exactly four executions, with no repeated completed tool. Resuming a finished checkpoint returns the stored result without appending another ledger action or run record. The time budget includes active processing time saved before the crash and the resumed work, excluding downtime.

### 6. ReAct versus plan-and-execute

ReAct uses five model calls and 1,087 tokens for S1, three calls and 451 tokens for cancelled S5, and two calls and 148 tokens for missing S6. It chooses the next action after each observation, so missing orders and cancellations stop naturally without executing the remaining refund steps. The growing conversation also means later calls repeatedly include earlier observations.

A plan-and-execute design could ask once for the tool sequence, execute it with deterministic checks, and replan only on unexpected results. Its model-call count could be lower, but token savings depend on the actual planning prompt and any replanning; that implementation was not benchmarked here. It must still branch or stop on NOT_FOUND, cancellation and approval thresholds rather than blindly executing its plan.

For this fixed refund workflow, I would choose deterministic execution of a short validated plan, with human review for exceptions. The rules are explicit and do not need repeated model decisions. ReAct is useful here as an observable teaching implementation, and would be more useful if later observations required genuinely different investigation paths.

---

## Limitations and Next Steps

1. All data and policies are fictional. The mock classifies using keywords and selects policy sentences; its perfect few-shot accuracy is not evidence of real-model accuracy. No real refund is issued.
2. The Premium flag is a post-hoc whole-window label that includes September. The retention list is illustrative and is **not a leakage-free evaluation**. A deployed version should calculate its segment flag at the prediction cutoff.
3. Feature aggregation excludes September and preprocessing/resampling never use test rows. Part A nevertheless imputes ages and delivery times using the full clean dataset, as required by the brief. A stricter prospective study should fit those cleaning statistics only on the available history as well.
4. The 25 customers without completed orders have no RFM segment. Ticket routing treats a customer absent from the segment file as non-premium.
5. TF-IDF may miss paraphrases, and three chunks can omit a relevant exception. The current grounding checks use only 14 supplied questions; expand them before considering real support use.
6. Regex redaction covers the specified synthetic email, phone and order-id formats only. Output validation checks category/schema or an unambiguous keyword, not factual correctness. Seventeen tickets go to humans in the final run.
7. Checkpoints avoid replay after the demonstrated crash, which occurs after a saved observation. They do not guarantee exactly-once refunds if a process dies between a ledger append and checkpoint persistence. A real system would need transactional storage and idempotency keys, plus authentication and concurrent-run controls. The demo ledger reset is for reproducibility only.
8. Timeout checks occur between operations; they cannot interrupt a blocking external tool. The supplied tools and mock are local. A network-backed implementation would also need per-call timeouts.
9. Seeds fix data generation, clustering, train/test splits, forests and resampling. Retry jitter and timestamps vary. Python 3.13.5 on Windows was used for verification; other platforms have not been independently tested.

Submission branch: [main](https://github.com/sudharsan-m-s/instamart-copilot-capstone-project/tree/main).
