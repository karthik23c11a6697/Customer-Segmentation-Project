"""
Customer Segmentation Project
Internship Task 2 - Python K-Means Clustering
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. GENERATE / LOAD DATASET ─────────────────────────────────────────────────
np.random.seed(42)
n = 500

ages     = np.concatenate([np.random.normal(25,4,120), np.random.normal(35,5,150),
                           np.random.normal(45,6,130), np.random.normal(58,7,100)])
incomes  = np.concatenate([np.random.normal(30000,5000,120), np.random.normal(55000,8000,150),
                           np.random.normal(75000,10000,130), np.random.normal(95000,12000,100)])
spend    = np.concatenate([np.random.normal(20,8,120), np.random.normal(45,12,150),
                           np.random.normal(65,15,130), np.random.normal(80,18,100)])
freq     = np.concatenate([np.random.normal(3,1,120), np.random.normal(6,2,150),
                           np.random.normal(9,2,130), np.random.normal(12,3,100)])
tenure   = np.concatenate([np.random.normal(1,0.5,120), np.random.normal(3,1,150),
                           np.random.normal(5,1.5,130), np.random.normal(8,2,100)])
idx = np.random.permutation(n)

df = pd.DataFrame({
    'Customer_ID':   [f'C{1000+i}' for i in range(n)],
    'Age':           np.clip(ages[idx], 18, 75).astype(int),
    'Annual_Income': np.clip(incomes[idx], 15000, 150000).astype(int),
    'Monthly_Spend': np.clip(spend[idx], 5, 200).round(2),
    'Purchase_Freq': np.clip(freq[idx], 1, 20).round(1),
    'Tenure_Years':  np.clip(tenure[idx], 0.1, 12).round(1),
    'Gender':        np.random.choice(['Male','Female'], n, p=[0.48,0.52]),
    'Region':        np.random.choice(['North','South','East','West'], n),
    'Category_Pref': np.random.choice(['Electronics','Fashion','Grocery','Home','Sports'], n),
})
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 2. PREPROCESSING ───────────────────────────────────────────────────────────
features = ['Age','Annual_Income','Monthly_Spend','Purchase_Freq','Tenure_Years']
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. FIND OPTIMAL K ──────────────────────────────────────────────────────────
inertias, sil_scores = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

best_k = 4  # confirmed by elbow + silhouette

# ── 4. FINAL CLUSTERING ────────────────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = km_final.fit_predict(X_scaled)

cluster_means = df.groupby('Cluster')[['Annual_Income','Monthly_Spend','Purchase_Freq','Age']].mean()
cluster_means['score'] = cluster_means['Annual_Income']*0.5 + cluster_means['Monthly_Spend']*100
order = cluster_means['score'].rank().astype(int)
label_map = {c: ['Budget Shoppers','Occasional Buyers','Regular Customers','Premium Loyalists'][order[c]-1]
             for c in range(best_k)}
df['Segment'] = df['Cluster'].map(label_map)

print(f"\nSilhouette Score: {silhouette_score(X_scaled, km_final.labels_):.3f}")
print("\nSegment Summary:")
print(df.groupby('Segment')[features].mean().round(1))

# ── 5. PCA ─────────────────────────────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'], df['PCA2'] = X_pca[:,0], X_pca[:,1]

# ── 6. EXPORT ──────────────────────────────────────────────────────────────────
df.to_csv('customer_segmented.csv', index=False)
print("\nExported: customer_segmented.csv")
