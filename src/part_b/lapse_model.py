"""Train lapse models, compare results and save the model and retention list."""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Establish project root and paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ORDERS_PATH = DATA_DIR / "clean" / "orders.csv"
CUSTOMERS_PATH = DATA_DIR / "clean" / "customers.csv"
PRODUCTS_PATH = DATA_DIR / "clean" / "products.csv"
SEGMENTS_PATH = OUTPUTS_DIR / "customer_segments.csv"

# Task 4: Define the Lapse Problem
print("PART B - TASK 4: DEFINE LAPSE PROBLEM & COHORT EXTRACTION")

orders = pd.read_csv(ORDERS_PATH)
customers = pd.read_csv(CUSTOMERS_PATH)
products = pd.read_csv(PRODUCTS_PATH)

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

cutoff = pd.to_datetime("2025-09-01 00:00:00")
sept_end = pd.to_datetime("2025-09-30 23:59:59")

# Historical orders strictly before cutoff
before = orders[orders["order_ts"] < cutoff].copy()
completed_before = before[before["status"] == "completed"].copy()
completed_before = completed_before.merge(products[["product_id", "category"]], on="product_id", how="left")

# Cohort: customers with at least one completed order before cutoff
cohort_cids = sorted(completed_before["customer_id"].unique())
print(f"Total customers in pre-cutoff cohort: {len(cohort_cids)}")
assert len(cohort_cids) == 1156, f"Expected 1156 cohort customers, got {len(cohort_cids)}"

# Ground-truth lapse label: 1 if NO completed order in Sept 2025, else 0
sept_completed = orders[
    (orders["order_ts"] >= cutoff) &
    (orders["order_ts"] <= sept_end) &
    (orders["status"] == "completed")
]
active_sept_customers = set(sept_completed["customer_id"].unique())

# Feature extraction using ONLY pre-cutoff data
# 1. Mode favorite category
fav_cat = completed_before.groupby("customer_id")["category"].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
)

# 2. Cancellation rate (cancelled / total completed + cancelled before cutoff)
all_orders_before = before.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    cancelled_orders=("status", lambda s: (s == "cancelled").sum()),
)
cancellation_rate = (
    all_orders_before["cancelled_orders"] / all_orders_before["total_orders"]
).fillna(0)

# 3. Customer order aggregates
agg_before = completed_before.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean"),
)

agg_before["recency"] = (cutoff - agg_before["last_order_ts"]).dt.days
agg_before["favourite_category"] = fav_cat
agg_before["cancellation_rate"] = cancellation_rate

# Merge with customer demographics
features = agg_before.reset_index().merge(customers, on="customer_id", how="inner")
features["tenure_days"] = (cutoff - features["signup_date"]).dt.days
features["lapsed"] = features["customer_id"].apply(
    lambda cid: 0 if cid in active_sept_customers else 1
)

lapsed_count = features["lapsed"].sum()
lapsed_pct = (lapsed_count / len(features)) * 100
print(f"Lapsed customers: {lapsed_count} ({lapsed_pct:.2f}%)")
print(f"Non-lapsed customers: {len(features) - lapsed_count} ({100 - lapsed_pct:.2f}%)")

assert lapsed_count == 443, f"Expected 443 lapsed, got {lapsed_count}"
assert round(lapsed_pct, 2) == 38.32, f"Expected 38.32% lapsed, got {lapsed_pct:.2f}"

num_cols = [
    "recency", "frequency", "monetary", "avg_discount",
    "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"
]
cat_cols = ["city", "gender", "favourite_category"]

# Train/test split (80/20, stratified, seed 42)
X = features[["customer_id"] + num_cols + cat_cols]
y = features["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training set: {len(X_train)} rows | Test set: {len(X_test)} rows")
assert len(X_train) == 924, f"Expected 924 train rows, got {len(X_train)}"
assert len(X_test) == 232, f"Expected 232 test rows, got {len(X_test)}"

# Majority-class baseline: always predict 0 (not lapsed)
y_pred_baseline = np.zeros(len(y_test))
base_acc = accuracy_score(y_test, y_pred_baseline) * 100
base_f1 = f1_score(y_test, y_pred_baseline, zero_division=0)
print(f"\nBaseline (Always Not Lapsed) -> Accuracy: {base_acc:.1f}%, Lapsed-class F1: {base_f1:.3f}")
assert round(base_acc, 1) == 61.6, f"Expected 61.6% baseline accuracy, got {base_acc}"

# Task 5: Preprocessing Without Leakage
print("PART B - TASK 5: PREPROCESSING WITHOUT DATA LEAKAGE")

# Check missing values in features
missing_counts = features[num_cols + cat_cols].isnull().sum().sum()
assert missing_counts == 0, f"Found {missing_counts} missing values in feature set!"
print("Missing values in feature set: 0 (Confirmed)")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
    ]
)

# Fit strictly on X_train
preprocessor.fit(X_train[num_cols + cat_cols])

# Compare learned scaler mean of frequency vs full-cohort mean
freq_idx = num_cols.index("frequency")
scaler_freq_mean = preprocessor.named_transformers_["num"].mean_[freq_idx]
full_cohort_freq_mean = features["frequency"].mean()

print(f"Learned mean of frequency from training scaler : {scaler_freq_mean:.4f}")
print(f"Full-cohort mean of frequency                  : {full_cohort_freq_mean:.4f}")
print("Explanation: The learned scaler mean is computed exclusively on the 924 training rows to prevent")
print("information leakage from the 232 test rows, so minor random sampling variance creates a slight difference.")

X_train_proc = preprocessor.transform(X_train[num_cols + cat_cols])
X_test_proc = preprocessor.transform(X_test[num_cols + cat_cols])

# Task 6: Train, Tune, and Evaluate Models
print("PART B - TASK 6: MODEL TUNING & EVALUATION (5-FOLD CV)")

# 1. Logistic Regression Tuning
lr_grid = GridSearchCV(
    Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1",
)
lr_grid.fit(X_train[num_cols + cat_cols], y_train)

# 2. Random Forest Tuning (45 fits: 9 combos * 5 folds)
rf_grid = GridSearchCV(
    Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]),
    param_grid={"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
    cv=5,
    scoring="f1",
)
rf_grid.fit(X_train[num_cols + cat_cols], y_train)

print(f"Logistic Regression -> Best C: {lr_grid.best_params_['model__C']} | CV F1: {lr_grid.best_score_:.3f}")
print(f"Random Forest       -> Best max_depth: {rf_grid.best_params_['model__max_depth']}, min_samples_leaf: {rf_grid.best_params_['model__min_samples_leaf']} | CV F1: {rf_grid.best_score_:.3f}")

# Evaluate on untouched test set
y_pred_lr = lr_grid.predict(X_test[num_cols + cat_cols])
y_pred_rf = rf_grid.predict(X_test[num_cols + cat_cols])

lr_f1 = f1_score(y_test, y_pred_lr)
lr_prec = precision_score(y_test, y_pred_lr)
lr_rec = recall_score(y_test, y_pred_lr)

rf_f1 = f1_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf)
rf_rec = recall_score(y_test, y_pred_rf)

print("\n--- Test Set Performance ---")
print(f"Logistic Regression: Precision={lr_prec:.3f}, Recall={lr_rec:.3f}, F1={lr_f1:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

print(f"\nRandom Forest:       Precision={rf_prec:.3f}, Recall={rf_rec:.3f}, F1={rf_f1:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

assert rf_f1 >= 0.55, f"Random Forest F1 ({rf_f1:.3f}) fell below minimum required threshold of 0.55!"

# Justification of model choice
print("\nModel Selection Justification:")
print(f"Random Forest is selected with CV F1 {rf_grid.best_score_:.3f} and test F1 {rf_f1:.3f}")

# Task 7: Class-Balance Experiment
print("PART B - TASK 7: CLASS-BALANCE OVERSAMPLING EXPERIMENT")

train_df = X_train.copy()
train_df["lapsed"] = y_train

majority_train = train_df[train_df["lapsed"] == 0]
minority_train = train_df[train_df["lapsed"] == 1]
n_needed = len(majority_train) - len(minority_train)

oversampled_minority = minority_train.sample(n=n_needed, replace=True, random_state=42)
balanced_train_df = pd.concat([train_df, oversampled_minority])

print("Pre-balance training class counts :", dict(y_train.value_counts()))
print("Post-balance training class counts:", dict(balanced_train_df["lapsed"].value_counts()))
assert len(balanced_train_df) == 1140, f"Expected 1140 balanced training rows, got {len(balanced_train_df)}"

X_bal_proc = preprocessor.transform(balanced_train_df[num_cols + cat_cols])
y_bal = balanced_train_df["lapsed"]

# Fit Logistic Regression on balanced training data
best_c = lr_grid.best_params_["model__C"]
lr_balanced = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
lr_balanced.fit(X_bal_proc, y_bal)

y_pred_bal = lr_balanced.predict(X_test_proc)
bal_f1 = f1_score(y_test, y_pred_bal)
bal_prec = precision_score(y_test, y_pred_bal)
bal_rec = recall_score(y_test, y_pred_bal)

print(f"\nUnbalanced Logistic Regression -> Precision: {lr_prec:.3f}, Recall: {lr_rec:.3f}, F1: {lr_f1:.3f}")
print(f"Balanced Logistic Regression   -> Precision: {bal_prec:.3f}, Recall: {bal_rec:.3f}, F1: {bal_f1:.3f}")
print(f"\nRecall changed by {(bal_rec - lr_rec) * 100:.1f} percentage points; "
      f"precision changed by {(bal_prec - lr_prec) * 100:.1f} points.")
print("Higher recall finds more lapsing customers, but lower precision spends offers on more loyal customers.")

# Task 8: Interpret, Persist, and Create Retention List
print("PART B - TASK 8: FEATURE IMPORTANCE, PERSISTENCE, & RETENTION LIST")

# Feature Names from ColumnTransformer
cat_encoder = preprocessor.named_transformers_["cat"]
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + cat_feature_names

# 1. Random Forest Top 5 Features
best_rf = rf_grid.best_estimator_.named_steps["model"]
rf_importances = pd.Series(best_rf.feature_importances_, index=all_feature_names).sort_values(ascending=False)

print("\n--- Top 5 Most Important Features (Random Forest) ---")
for feat, imp in rf_importances.head(5).items():
    print(f"  {feat:<22}: {imp:.4f}")

# 2. Logistic Regression Coefficients (Top 3 Interpretations)
best_lr = lr_grid.best_estimator_.named_steps["model"]
lr_coefs = pd.Series(best_lr.coef_[0], index=all_feature_names)
print("\n--- Logistic Regression Feature Coefficients (Selected 3) ---")
for feature in ["recency", "frequency", "monetary"]:
    direction = "higher" if lr_coefs[feature] > 0 else "lower"
    print(f"  {feature} ({lr_coefs[feature]:+.4f}): Associated with {direction} lapse log-odds, "
          "holding other features fixed; this is not a causal effect.")

# 3. Persist Model and Test Exact Reloadability
model_bundle = {
    "preprocessor": preprocessor,
    "model": best_rf,
    "feature_names": all_feature_names,
}
model_path = OUTPUTS_DIR / "lapse_model.joblib"
joblib.dump(model_bundle, model_path)
print(f"\nSaved production model bundle to: {model_path}")

# Reload model
loaded_bundle = joblib.load(model_path)
loaded_prep = loaded_bundle["preprocessor"]
loaded_model = loaded_bundle["model"]
loaded_preds = loaded_model.predict(loaded_prep.transform(X_test[num_cols + cat_cols]))

assert np.array_equal(y_pred_rf, loaded_preds), "Reloaded model predictions do NOT match original!"
print("[PASS] Reloaded model reproduces original predictions with 100% precision.")

# 4. Generate outputs/retention_list.csv
# Read customer segments from task 3 to identify Premium customers in test set
segments_df = pd.read_csv(SEGMENTS_PATH)
test_merged = X_test.copy()
test_merged["actually_lapsed"] = y_test.values
test_merged = test_merged.merge(segments_df[["customer_id", "is_premium", "segment_name"]], on="customer_id", how="left")

test_probs = best_rf.predict_proba(X_test_proc)[:, 1]
test_merged["lapse_probability"] = test_probs

premium_ranked = test_merged[test_merged["is_premium"] == True].sort_values(
    "lapse_probability", ascending=False
)
top_10_retention = premium_ranked.head(10)[["customer_id", "lapse_probability", "actually_lapsed"]]
top_10_retention["lapse_probability"] = top_10_retention["lapse_probability"].round(4)

retention_path = OUTPUTS_DIR / "retention_list.csv"
top_10_retention.to_csv(retention_path, index=False)
print(f"\nSaved top 10 Premium retention list to: {retention_path}")
print(top_10_retention.to_string(index=False))

n_truly_lapsed = top_10_retention["actually_lapsed"].sum()
print(f"\nOf the top 10 high-risk Premium customers, exactly {n_truly_lapsed} truly lapsed.")
print("Note on Leakage: The Premium flag is derived from whole-window RFM (including September data), making this")
print("retention list an illustrative post-hoc demonstration rather than a strict leakage-free evaluation.")

# Persist measured values so the README can be checked against a new run.
report = {
    "cohort": len(X), "lapsed": int(y.sum()),
    "train_size": len(X_train), "test_size": len(X_test),
    "train_frequency_mean": float(scaler_freq_mean),
    "cohort_frequency_mean": float(full_cohort_freq_mean),
    "logistic": {
        "best_params": lr_grid.best_params_, "cv_f1": lr_grid.best_score_,
        "precision": lr_prec, "recall": lr_rec, "f1": lr_f1,
        "confusion": confusion_matrix(y_test, y_pred_lr).tolist(),
        "coefficients": {k: float(lr_coefs[k]) for k in ["recency", "frequency", "monetary"]},
    },
    "forest": {
        "best_params": rf_grid.best_params_, "cv_f1": rf_grid.best_score_,
        "precision": rf_prec, "recall": rf_rec, "f1": rf_f1,
        "confusion": confusion_matrix(y_test, y_pred_rf).tolist(),
        "top_features": {k: float(v) for k, v in rf_importances.head(5).items()},
    },
    "balanced_logistic": {"precision": bal_prec, "recall": bal_rec, "f1": bal_f1},
    "retention_truly_lapsed": int(n_truly_lapsed),
}
(OUTPUTS_DIR / "model_metrics.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
