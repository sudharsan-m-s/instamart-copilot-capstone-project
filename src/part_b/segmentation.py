from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Establish project root and paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ORDERS_PATH = DATA_DIR / "clean" / "orders.csv"

# Task 1: Build the RFM Table
print("PART B - TASK 1: BUILD RFM TABLE")

orders = pd.read_csv(ORDERS_PATH)
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Filter for completed orders only
completed = orders[orders["status"] == "completed"].copy()
snapshot = pd.to_datetime("2025-10-01 00:00:00")

# Compute RFM metrics per customer
rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()

rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm["monetary"] = rfm["monetary"].round(2)

print(f"Total customers in RFM cohort: {len(rfm)}")
print(f"Recency   -> Median: {rfm['recency'].median():.1f}, Min: {rfm['recency'].min()}, Max: {rfm['recency'].max()}")
print(f"Frequency -> Mean: {rfm['frequency'].mean():.2f}, Max: {rfm['frequency'].max()}")
print(f"Monetary  -> Mean: INR {rfm['monetary'].mean():.2f}, Max: INR {rfm['monetary'].max():.2f}")

# Assertions matching references
assert len(rfm) == 1175, f"Expected 1175 customers, got {len(rfm)}"
assert rfm["recency"].median() == 19.0, f"Expected recency median 19, got {rfm['recency'].median()}"
assert rfm["recency"].min() == 0, f"Expected recency min 0, got {rfm['recency'].min()}"
assert rfm["recency"].max() == 180, f"Expected recency max 180, got {rfm['recency'].max()}"
assert round(rfm["frequency"].mean(), 2) == 9.46, f"Expected frequency mean 9.46, got {rfm['frequency'].mean():.2f}"
assert rfm["frequency"].max() == 33, f"Expected frequency max 33, got {rfm['frequency'].max()}"
assert round(rfm["monetary"].mean(), 2) == 2552.06, f"Expected monetary mean 2552.06, got {rfm['monetary'].mean():.2f}"
assert round(rfm["monetary"].max(), 2) == 14766.55, f"Expected monetary max 14766.55, got {rfm['monetary'].max():.2f}"
print("[PASS] RFM table matches all reference checkpoints.")

# Task 2: Cluster with K-Means
print("PART B - TASK 2: K-MEANS CLUSTERING & ELBOW / SILHOUETTE ANALYSIS")

rfm_features = ["recency", "frequency", "monetary"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[rfm_features])

kmeans_results = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    kmeans_results.append({
        "K": k,
        "Inertia": round(km.inertia_, 1),
        "Silhouette": round(sil, 3),
    })

df_eval = pd.DataFrame(kmeans_results)
print(df_eval.to_string(index=False))

# Explanation for K choice
print("\nUsing K=4: highest silhouette score and four useful customer groups.")

# Task 3: Profile, Name and Project the Segments (K=4)
print("PART B - TASK 3: SEGMENT PROFILING, NAMING, AND PCA PROJECTION")

km4 = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["cluster"] = km4.fit_predict(X_scaled)

# Identify the segment with highest mean monetary value as Premium
monetary_by_cluster = rfm.groupby("cluster")["monetary"].mean()
premium_cluster_id = monetary_by_cluster.idxmax()

# Name clusters by their mean behaviour, rather than assuming an id order.
freq_by_cluster = rfm.groupby("cluster")["frequency"].mean()
rec_by_cluster = rfm.groupby("cluster")["recency"].mean()

frequent_cluster_id = freq_by_cluster.idxmax()
at_risk_cluster_id = rec_by_cluster.idxmax()
occasional_cluster_id = [c for c in range(4) if c not in [premium_cluster_id, frequent_cluster_id, at_risk_cluster_id]][0]

cluster_name_map = {
    premium_cluster_id: "Premium Spenders",
    frequent_cluster_id: "Frequent Shoppers",
    at_risk_cluster_id: "At-Risk / Lapsed",
    occasional_cluster_id: "Occasional Shoppers",
}

rfm["segment_id"] = rfm["cluster"]
rfm["segment_name"] = rfm["cluster"].map(cluster_name_map)
rfm["is_premium"] = rfm["cluster"] == premium_cluster_id

profile_summary = rfm.groupby(["segment_id", "segment_name", "is_premium"]).agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
).reset_index()

print("\n--- Segment Profiles (K=4) ---")
print(profile_summary.round(2).to_string(index=False))

# PCA Projection
pca = PCA(n_components=2)
pca.fit(X_scaled)
var_ratios = pca.explained_variance_ratio_
print(f"\nPCA Explained Variance Ratios: PC1 = {var_ratios[0]:.3f}, PC2 = {var_ratios[1]:.3f}, Total = {var_ratios.sum():.3f}")

# Save outputs/customer_segments.csv
output_cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
out_file = OUTPUTS_DIR / "customer_segments.csv"
rfm[output_cols].to_csv(out_file, index=False)
print(f"\nSaved customer segments to: {out_file}")
print(f"Total rows written: {len(rfm)}")
