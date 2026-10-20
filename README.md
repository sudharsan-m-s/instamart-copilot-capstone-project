# Instamart Customer Intelligence & Support Copilot

## Overview

This project cleans order data, groups customers, predicts lapse, routes support tickets and handles simulated refunds. All data and policies are fictional. Amounts are in INR, and all model calls use the supplied offline `mock_llm.py`.

| Folder | Contents |
| --- | --- |
| `src/part_a/` | Cleaning, database and SQL checks |
| `src/part_b/` | Customer segments and lapse models |
| `src/part_c/` | Ticket privacy checks and routing |
| `src/part_d/` | Policy retrieval and refund agent |
| `data/`, `sql/` | Raw/clean data, policies, SQLite database and queries |
| `outputs/`, `state/` | Results, saved model and agent checkpoints |
| `tests/` | Guardrail and recovery tests |

The supplied `generate_data.py` and `mock_llm.py` are unchanged.

## Reproduce

Run these numbered commands from the project root on `main`. Tested with Python 3.13.5 on Windows. `requirements.txt` pins the installed libraries and their dependencies; SQLite comes with Python.

1. Create the environment: `python -m venv .venv`
2. Activate it: `.\.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on Linux/macOS.
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Run all parts and tests: `python -m src.reproduce`

If PowerShell blocks activation, replace `python` in steps 3–4 with `.\.venv\Scripts\python.exe`. Only dependency installation needs network access.

The runner calls these stages in order and stops if one fails:

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

The generator prints `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The full run ends with `ALL FOUR PARTS REPRODUCED SUCCESSFULLY`. Logs in `outputs/logs/` are ignored by Git. The Part D demo replaces its run log, refund ledger and ten `demo-*` checkpoints on each run; direct calls to `run_agent` append results under new run ids.

## Part A

### Raw data

Shapes: customers (1,200, 5), products (40, 4), orders (13,084, 9). Pandas reads `N/A` as missing, while `error` stays text until numeric conversion.

| Table | Column | dtype | Missing | Missing % |
| --- | --- | --- | ---: | ---: |
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

| Raw audit | Count |
| --- | ---: |
| Order rows / distinct order ids | 13,084 / 12,309 |
| Exact duplicates / pending rows | 119 / 656 |
| Non-numeric amounts | 120 |
| Negative / blank delivery times | 106 / 2,541 |
| Blank / invalid ages (outside 10–100) | 61 / 6 |
| City spellings / non-blank gender spellings | 24 / 10 |
| Blank genders | 30 |

### Cleaning

- Strip and title-case cities: 476 values changed. Final counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191 and Mumbai 184.
- Normalise gender spellings and map blanks to `unknown`: 950 values changed. Final counts: female 610, male 560, unknown 30.
- Replace six invalid ages with missing, then fill all 67 missing ages using each city's median with `groupby().transform()`. No ages remain missing; the range is 18–60.
- Parse 1,200 signup dates and 13,084 order timestamps; no parsing failures.
- Remove 119 exact duplicate orders, then keep the latest timestamp per order id, removing 656 earlier pending records.
- After deduplication, repair 112 amounts with `round(price_inr * quantity * (1 - discount_pct / 100), 2)`.
- Replace 105 remaining negative durations with missing. Fill 774 completed-order durations using completed orders' category medians. Leave all 1,199 cancelled-order durations missing.

The cleaned data contains 12,309 unique orders: 11,110 completed and 1,199 cancelled, with no pending rows. No amounts or completed-order durations are missing. Completed revenue is INR 2,998,668.80.

Clean CSVs are saved in `data/clean/`. SQLite has `customers` (1,200 rows), `products` (40) and `orders` (12,309). Order timestamps use `YYYY-MM-DD HH:MM:SS`. The validation script prints `ALL CHECKPOINTS PASSED` after its assertions.

### SQL and pandas results

All seven queries are in `sql/queries.sql` and run through `pd.read_sql_query`.

| Query | Result |
| --- | --- |
| Q1: WHERE amount > 500, HAVING spend > 8000 | 30 customers |
| Q2: three-table revenue join | 30 city/category rows, total INR 2,998,668.80; largest: Bengaluru/Electronics, INR 233,909.45 |
| Q3: price above catalogue average | 11 products |
| Q4: LEFT JOIN, no completed orders | 25 customers |
| Q5: two CTEs, spend above customer average | 550 customers; average INR 2,552.06 |
| Q6: rank customers within city | 8 rows have different RANK and ROW_NUMBER values |
| Q7: daily running revenue | 183 rows; final total INR 2,998,668.80 |

Q6 leaders: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10). Tied spend shares a RANK, while ROW_NUMBER assigns a separate position to each row.

The pandas pivot table uses `margins=True`; its total matches SQL at INR 2,998,668.80. Named aggregation gives:

| City | Sum (INR) | Mean (INR) | Count |
| --- | ---: | ---: | ---: |
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order amounts: mean INR 269.91, median INR 135.00, skewness 4.61; Q1/Q2/Q3 are INR 64.80/135.00/297.00. IQR is INR 232.20; fences are INR -283.50 and 645.30. There are zero outliers below the lower fence and 897 above the upper fence.

| Correlation | Amount | Discount | Quantity | Delivery minutes |
| --- | ---: | ---: | ---: | ---: |
| Amount | 1.000 | -0.095 | 0.087 | -0.003 |
| Discount | -0.095 | 1.000 | -0.000 | -0.010 |
| Quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| Delivery minutes | -0.003 | -0.010 | -0.013 | 1.000 |

The data describes grocery and other quick-commerce orders across six cities. Spending is right-skewed: a small number of large purchases pull the mean above the median. The 897 high-value outliers need not be errors; the planted anomalies are duplicates, invalid ages, amounts and durations. The largest absolute correlation is amount versus discount (-0.095), followed by amount versus quantity (0.087). These are weak relationships and do not establish causation.

## Part B

### Segments

RFM uses completed orders and snapshot `2025-10-01 00:00:00`. There are 1,175 customers. Recency has median 19 days and range 0–180; frequency has mean 9.46 and maximum 33; monetary has mean INR 2,552.06 and maximum INR 14,766.55.

The three RFM features use `StandardScaler`; K-means uses `n_init=10, random_state=42`.

| K | Inertia | Silhouette |
| ---: | ---: | ---: |
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 has the highest silhouette here, a clear reduction in inertia and four useful customer groups. More centres normally lower inertia because observations can be closer to a centre, so K=8 having lower inertia than K=2 is not enough to choose it. Extra groups must also be useful to the business.

| ID | Segment | Size | Mean recency | Mean frequency | Mean monetary (INR) | Action |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Category coupons |
| 1 | Premium Spenders | 138 | 24.12 | 6.77 | 7,129.70 | Dedicated support |
| 2 | Frequent Shoppers | 419 | 8.77 | 19.40 | 3,410.02 | Repeat-order rewards |
| 3 | At-Risk / Lapsed | 182 | 109.04 | 2.20 | 685.99 | Win-back offer |

Cluster ids are unordered labels. Only the group with the highest mean monetary value is Premium. PCA variance ratios are 0.634 and 0.218, totalling 0.852. Assignments are saved in `outputs/customer_segments.csv`.

### Lapse prediction

Cutoff: `2025-09-01`. A customer is lapsed if they made no completed order in September. The pre-cutoff cohort has 1,156 customers, including 443 lapsed (38.32%). The stratified 80/20 split with seed 42 gives 924 training and 232 test rows. Always predicting not-lapsed gives 61.6% accuracy and zero lapsed-class F1.

Features are recency, frequency, monetary, average discount, cancellation rate, average delivery duration, favourite category, city, age, gender and tenure. Order aggregation uses only dates before September; segment labels are not features. Favourite-category ties use the first sorted mode. No feature values are missing.

One-hot encoding (`drop='first'`) and numeric scaling are inside each cross-validation pipeline. No encoder, scaler or resampler sees test rows. The training frequency mean is 8.1147 versus 8.0285 for the full cohort because the training subset excludes the test customers.

Both searches use five-fold CV and F1 scoring. Logistic regression uses `max_iter=1000` and C in {0.01, 0.1, 1, 10, 100}. The forest uses 100 trees, seed 42, depths {3, 5, 8} and minimum leaf sizes {5, 10, 20}, giving 45 fits.

| Model | Best settings | CV F1 | Test precision | Test recall | Test F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic regression | C=100 | 0.651 | 0.644 | 0.652 | 0.648 |
| Random forest | depth=8, leaf=10 | 0.681 | 0.659 | 0.652 | 0.655 |

Confusion matrices, with true rows and predicted columns ordered [0, 1]: logistic `[[111, 32], [31, 58]]`; forest `[[113, 30], [31, 58]]`.

The forest is selected for its higher CV F1; its test F1 is 0.655. Both models miss 31 lapsing customers, but the forest targets two fewer loyal customers by mistake. Missing a lapsing customer can lose future business, while wrongly targeting a loyal customer wastes an offer. The choice should also consider the cost of those offers.

Training-only oversampling keeps every original row and duplicates lapsed rows with replacement (seed 42), reaching 570 rows per class. Refitting the best logistic model gives precision 0.589, recall 0.854 and F1 0.697, versus 0.644/0.652/0.648 before balancing. Recall moves most (+20.2 percentage points): more lapsing customers are found, with more unnecessary offers.

| Forest feature | Importance | Reading |
| --- | ---: | --- |
| frequency | 0.2796 | Ordering frequency is the largest signal |
| monetary | 0.2712 | Total spending helps distinguish customers |
| recency | 0.1151 | Time since last order matters |
| avg_delivery_minutes | 0.0613 | Delivery history adds information |
| cancellation_rate | 0.0523 | Cancellation history adds information |

Logistic coefficients for standardised features: recency +0.0396 (higher lapse log-odds), frequency -1.6982 (lower lapse log-odds), monetary +0.2846 (higher lapse log-odds), holding other features fixed. These associations are not causal effects.

The selected model and preprocessor are saved in `outputs/lapse_model.joblib`; reloading reproduces test predictions exactly. `outputs/retention_list.csv` ranks ten Premium test customers by lapse probability; one truly lapsed. The Premium flag includes September, so this retention list is illustrative and not a leakage-free evaluation.

## Part C

Tickets are read with `keep_default_na=False`. Four empty tickets are skipped and four long tickets (734 estimated tokens each) exceed the 400-token input limit; none triggers a model call.

Redaction runs in order: email `[\w.+-]+@[\w-]+\.[\w.]+` to `[EMAIL]`; phone `\b[6-9]\d{9}\b` to `[PHONE]`; order id `ORD-\d{6}` to `[ORDER_ID]`. Across all 164 tickets there are 55 email, 91 phone and 132 order-id replacements: 278 total, with zero matches remaining. Assertions check all outgoing messages; the user message is exactly the redacted text.

Both system prompts contain CLASSIFY, the four categories and the required JSON form. The few-shot prompt adds `EXAMPLES:` with fraud and delivery examples; the zero-shot prompt has none. Each full run resets the mock and recovers all 19 transient failures. Up to three retries wait `0.05 * 2**n * random.uniform(0.9, 1.15)` seconds. Retry logs include ticket id, retry number and wait; this mock's waits are all retry 0, between 0.045 and 0.0575 seconds.

Output validation accepts 135 JSON categories and 12 unambiguous keyword replies, using `re.findall(r'[a-z]+', text.lower())`; nine invalid replies go to humans. Valid JSON with an invalid category is never rescued by keyword matching.

| Metric | Zero-shot | Few-shot |
| --- | ---: | ---: |
| Classified tickets | 147 | 147 |
| Correct | 126 | 147 |
| Accuracy | 85.7% | 100.0% |
| Total tokens | 11,584 | 21,100 |
| Tokens per classified ticket | 78.8 | 143.5 |

Zero-shot confusion table:

| True / predicted | Billing | Delivery | Technical | Fraud |
| --- | ---: | ---: | ---: | ---: |
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

Few-shot uses 9,516 extra tokens, or 64.7 more per classified ticket, and corrects 21 mistakes. Eleven of those errors are fraud tickets sent to billing, so the extra context is useful when missed fraud is costly. These results describe the supplied keyword mock, not a real model's reliability.

The 6,000-token zero-shot budget demo stops new calls once usage reaches the budget: total usage is 6,029, and 75 tickets are `budget_exhausted`. The final few-shot run has a 100,000-token budget. Results are in `triage_budget_demo.csv` and `triage_results.csv` under `outputs/`.

| Route | Priority | Tickets |
| --- | --- | ---: |
| human_review | NORMAL | 17 |
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |

Routing first sends skipped, flagged or exhausted tickets to humans, then fraud to security, then Premium customers to the premium desk, and the remainder to support. Missing segment rows count as non-premium. The final file has 164 rows; 147 are rule-validated classifications and 17 require human review.

## Part D

### Policy assistant

Policies are split into 40-word chunks with ten-word overlap, stopping when a chunk reaches the document end.

| Policy | Chunks with overlap 10 | Chunks with overlap 0 |
| --- | ---: | ---: |
| refund | 4 | 3 |
| delivery | 3 | 3 |
| escalation | 3 | 2 |
| payment | 3 | 3 |
| privacy | 2 | 2 |
| Total | 15 | 13 |

`TfidfVectorizer()` and cosine similarity rank chunks. TF measures term frequency in a chunk; IDF gives more weight to terms found in fewer chunks. TF-IDF can miss paraphrases with different words.

`retrieve(query, k=3)` returns source, score and text. For `refund policy late delivery`, all three results come from `refund_policy.txt`. `answer(question)` sends source-prefixed context and the question to the mock under a POLICY ASSISTANT instruction to use only that context, then returns text and sources.

| Evaluation | Overlap 10 | Overlap 0 |
| --- | ---: | ---: |
| Expected document retrieved | 12/12 | 12/12 |
| Expected fact in answer | 12/12 | 10/12 |
| Unsupported-question fallback | 2/2 | 2/2 |

Without overlap, “What delivery time makes a completed order eligible for a full refund?” loses `more than 45 minutes`. The failed-payment timing question also loses `48 hours`. The relevant sentences cross chunk boundaries; overlap keeps the facts together. Per-question results are in `outputs/policy_evaluation.json`.

### Refund agent

`TOOLS_SPEC` holds names, descriptions and parameter specifications; `TOOL_MAP` maps names to functions.

| Tool | Purpose |
| --- | --- |
| get_order(order_id) | Read the four order fields from SQLite, or return not found |
| search_policy(query) | Return the same policy retrieval results |
| calculate_refund(order_amount, delivery_minutes) | Full amount above 45 minutes; otherwise zero |
| issue_refund(order_id, refund_inr) | Validate and log a simulated refund up to INR 1000 |
| escalate_to_human(order_id, refund_inr) | Log amounts above INR 1000 for approval, without issuing |

Only completed orders qualify. Write tools recheck the database, duration and amount. Requests are assumed to be within seven days. The ReAct loop records each assistant call and JSON tool observation; a reply without tool calls supplies the final answer.

| Scenario | Status | Refund/request (INR) | Model calls | Tokens |
| --- | --- | ---: | ---: | ---: |
| S1 | REFUND | 128.25 | 5 | 1,087 |
| S2 | REFUND | 135.00 | 5 | 1,087 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,074 |
| S4 | DENIED | 0.00 | 4 | 757 |
| S5 | DENIED | 0.00 | 3 | 451 |
| S6 | NOT_FOUND | 0.00 | 2 | 148 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,228 |

S3 is an approval request, S4 is on time, S5 is cancelled and S6 does not exist. S7 loops deliberately and completes all six calls below the default 6,000-token limit.

| Limit demo | Status | Calls | Tokens | Tools completed |
| --- | --- | ---: | ---: | --- |
| S1, token_limit=200 | FALLBACK_TOKEN_LIMIT | 3 | 468 | get_order, search_policy |
| S1, time_limit_seconds=0 | FALLBACK_TIMEOUT | 0 | 0 | none |

Tokens are checked after each reply and before executing its requested tool. `time.monotonic()` is checked at each iteration's start. Fallbacks return completed steps and a human-handoff summary.

A checkpoint is saved after every tool observation, containing the run/scenario ids, full messages, tokens, call count, step number and executed tools. S1 crashes after the second saved observation, then `resume_agent` continues:

```text
Before crash: get_order -> search_policy
After resume: calculate_refund -> issue_refund
Final: REFUND INR 128.25, five total calls, 1,087 tokens
```

There are exactly four tool executions, none repeated. A completed checkpoint can be resumed without another ledger entry. `outputs/agent_runs.jsonl` contains ten runs: seven scenarios, two limit demos and the resumed run. The ledger has four actions: S1, S2, S3 and the separate resume demonstration. `state/` contains ten checkpoints.

ReAct uses five calls/1,087 tokens for S1, three/451 for cancelled S5 and two/148 for missing S6. It reacts to each observation, so unexpected missing or cancelled orders stop naturally, but later calls repeat the conversation context. Plan-and-execute could request a tool sequence once and replan only on errors, reducing calls; its token use was not measured here. It would still need explicit checks to stop on NOT_FOUND or cancellation. For these fixed refund rules, a short validated plan with deterministic checks would be simpler; ReAct is useful for demonstrating tool use and recovery.

## Limitations and Next Steps

- Synthetic data and the keyword mock cannot establish real customer behaviour or real-model accuracy. No money is transferred.
- The Premium flag includes September. The retention list is illustrative and not a leakage-free evaluation; calculate segments at the cutoff for actual scoring.
- Feature aggregation and model preprocessing respect the split, but Part A imputes ages/durations from the full dataset as required. A prospective study should fit cleaning statistics on available history too.
- Twenty-five customers have no completed orders and no segment. Redaction covers only the specified formats; classification validation does not prove correctness.
- TF-IDF can miss paraphrases, and the policy evaluation has only 14 questions. Add more policy exceptions and real test cases before deployment.
- Checkpoints avoid repeats after the demonstrated saved-observation crash, but a crash between ledger writing and checkpoint saving can still duplicate an action. Real refunds need transactional storage, idempotency and access controls.
- Timeout checks cannot interrupt a blocking tool. Retry jitter, timestamps and elapsed times vary; seeds fix the data and models. Only Windows/Python 3.13.5 was tested.

Submission: [main branch](https://github.com/sudharsan-m-s/instamart-copilot-capstone-project/tree/main).
