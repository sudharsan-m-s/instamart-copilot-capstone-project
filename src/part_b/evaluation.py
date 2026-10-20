from pathlib import Path
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

print("RUNNING PART B VALIDATION CHECKPOINTS")

# 1. Customer Segments Validation
seg_path = OUTPUTS_DIR / "customer_segments.csv"
assert seg_path.exists(), f"Missing file: {seg_path}"

df_seg = pd.read_csv(seg_path)
expected_cols = {"customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"}
assert expected_cols.issubset(set(df_seg.columns)), f"Missing columns in {seg_path}: {expected_cols - set(df_seg.columns)}"
assert len(df_seg) == 1175, f"Expected 1175 segments rows, got {len(df_seg)}"

premium_count = df_seg["is_premium"].sum()
assert premium_count == 138, f"Expected 138 premium customers, got {premium_count}"

# Verify premium is strictly highest monetary segment
monetary_means = df_seg.groupby("segment_name")["monetary"].mean()
assert monetary_means.idxmax() == df_seg[df_seg["is_premium"]]["segment_name"].iloc[0], "Premium is not highest monetary!"

print("[PASS] customer_segments.csv verified (1175 rows, 138 Premium, valid schema).")

# 2. Lapse Model Persistence Validation
model_path = OUTPUTS_DIR / "lapse_model.joblib"
assert model_path.exists(), f"Missing file: {model_path}"

bundle = joblib.load(model_path)
assert "preprocessor" in bundle and "model" in bundle, "Invalid bundle structure in lapse_model.joblib"
print("[PASS] lapse_model.joblib successfully reloaded.")

# 3. Retention List Validation
ret_path = OUTPUTS_DIR / "retention_list.csv"
assert ret_path.exists(), f"Missing file: {ret_path}"

df_ret = pd.read_csv(ret_path)
assert set(df_ret.columns) == {"customer_id", "lapse_probability", "actually_lapsed"}, "Invalid columns in retention_list.csv"
assert len(df_ret) == 10, f"Expected 10 rows in retention list, got {len(df_ret)}"
assert df_ret["lapse_probability"].is_monotonic_decreasing, "Retention list is not sorted descending by lapse probability!"

print("[PASS] retention_list.csv verified (10 customers, sorted descending).")

print("\n" )
print("ALL PART B CHECKPOINTS PASSED")