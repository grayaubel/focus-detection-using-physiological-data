# 💡 Focus Detection Using Ensemble SVC per Cluster

# Step 1: Load & Prepare Data
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score
from itertools import combinations
from collections import defaultdict

# Load data
df = pd.read_csv("all_data.csv")

# Step 2: Compute Per-Subject Baseline Features
baseline_df = df[df['label'] == 1]
baseline_means = baseline_df.groupby('subject')[[
    'HR', 'EDA_mean', 'RESP_rate', 'RESP_regularity', 'net_acc_mean'
]].mean()
baseline_means.columns = [f"{col}_baseline_sub" for col in baseline_means.columns]
df = df.merge(baseline_means, left_on='subject', right_index=True, how='left')

for col in ['HR', 'EDA_mean', 'RESP_rate', 'RESP_regularity', 'net_acc_mean']:
    df[f"{col}_diff_from_baseline"] = df[col] - df[f"{col}_baseline_sub"]

# Step 3: Create Interaction Features
interactions = [
    ('HR', 'EDA_mean'),
    ('HR_diff_from_baseline', 'EDA_mean_diff_from_baseline'),
    ('RESP_rate_diff_from_baseline', 'net_acc_mean_diff_from_baseline'),
    ('HR', 'RESP_regularity'),
    ('EDA_mean', 'RESP_rate'),
    ('HR_diff_from_baseline', 'RESP_regularity_diff_from_baseline'),
]

for f1, f2 in interactions:
    df[f"{f1}_x_{f2}"] = df[f1] * df[f2]

# Step 4: Feature Selection with Permutation Importance
candidate_features = [col for col in df.columns if "_diff_from_baseline" in col or "_x_" in col or col in [
    'HR', 'EDA_mean', 'RESP_rate', 'RESP_regularity', 'net_acc_mean']]

train_mask = ~df['subject'].isin([15, 16])
X_train = df.loc[train_mask, candidate_features]
y_train = df.loc[train_mask, 'focus_label']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

result = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    'feature': candidate_features,
    'importance': result.importances_mean
}).sort_values(by='importance', ascending=False)

# Top 15 features
top_15_features = importance_df.head(15)['feature'].tolist()

# Step 5: Cluster Subjects Based on Baseline
cluster_data = baseline_means.copy()
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data['cluster'] = kmeans.fit_predict(cluster_data)
df['cluster'] = df['subject'].map(cluster_data['cluster'].to_dict())

# Step 6: Train SVC per Cluster (Ensemble)
cluster_models = {}
for cluster_id in df['cluster'].unique():
    cluster_df = df[df['cluster'] == cluster_id]
    X = cluster_df[top_15_features]
    y = cluster_df['focus_label']

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=1.0, gamma='scale', probability=True))
    ])
    model.fit(X, y)
    cluster_models[cluster_id] = model

# Step 7: Predict (Runtime Inference Logic)
def predict_with_cluster(sample_row, cluster_model_map, kmeans_model, top_features):
    """
    sample_row: DataFrame with one row
    cluster_model_map: dict {cluster_id: trained model}
    kmeans_model: trained KMeans used for subject clustering
    top_features: list of features for model input
    """
    cluster_input = sample_row[[f"{col}_baseline_sub" for col in ['HR', 'EDA_mean', 'RESP_rate', 'RESP_regularity', 'net_acc_mean']]].values.reshape(1, -1)
    cluster_id = kmeans_model.predict(cluster_input)[0]
    X = sample_row[top_features].values.reshape(1, -1)
    model = cluster_model_map[cluster_id]
    return model.predict(X)[0], cluster_id
