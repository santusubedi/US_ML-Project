"""
Phase 4: Clustering Analysis - Accident Hotspots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 4: CLUSTERING ANALYSIS - ACCIDENT HOTSPOTS")
print("="*70)

# Load featured dataset
print("\nLoading featured dataset...")
df = pd.read_csv('../data/processed/accidents_featured.csv')
print(f"✓ Loaded {df.shape[0]:,} rows")

# ========================================================================
# 1. Prepare Location Data for Clustering
# ========================================================================
print("\n" + "="*70)
print("1. PREPARING LOCATION DATA")
print("="*70)

# Check if location columns exist
if 'Start_Lat' not in df.columns or 'Start_Lng' not in df.columns:
    print("ERROR: Location columns not found!")
    exit(1)

# Remove any NaN locations
df_geo = df[['Start_Lat', 'Start_Lng', 'Severity']].dropna()
print(f"✓ Using {len(df_geo):,} locations for clustering")

# Sample for faster processing (use all if < 100k)
if len(df_geo) > 100000:
    df_geo = df_geo.sample(n=100000, random_state=42)
    print(f"✓ Sampled 100,000 locations for analysis")

# Standardize coordinates
scaler = StandardScaler()
X_geo = scaler.fit_transform(df_geo[['Start_Lat', 'Start_Lng']])

# ========================================================================
# 2. K-Means Clustering
# ========================================================================
print("\n" + "="*70)
print("2. K-MEANS CLUSTERING")
print("="*70)

# Determine optimal k using elbow method
print("Finding optimal number of clusters...")
inertias = []
k_range = range(5, 21, 5)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_geo)
    inertias.append(kmeans.inertia_)
    print(f"  k={k}: inertia={kmeans.inertia_:.0f}")

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('K-Means Elbow Method', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/06_kmeans_elbow.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 06_kmeans_elbow.png")
plt.close()

# Use k=10 clusters
optimal_k = 10
print(f"\nUsing k={optimal_k} clusters")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_geo['Cluster_KMeans'] = kmeans.fit_predict(X_geo)

print(f"✓ K-Means clustering completed")
print(f"Cluster distribution:")
print(df_geo['Cluster_KMeans'].value_counts().sort_index())

# ========================================================================
# 3. DBSCAN Clustering
# ========================================================================
print("\n" + "="*70)
print("3. DBSCAN CLUSTERING")
print("="*70)

# DBSCAN with eps and min_samples tuned for geographic data
dbscan = DBSCAN(eps=0.1, min_samples=50)
df_geo['Cluster_DBSCAN'] = dbscan.fit_predict(X_geo)

n_clusters = len(set(df_geo['Cluster_DBSCAN'])) - (1 if -1 in df_geo['Cluster_DBSCAN'] else 0)
n_noise = list(df_geo['Cluster_DBSCAN']).count(-1)

print(f"✓ DBSCAN clustering completed")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise:,} ({100*n_noise/len(df_geo):.1f}%)")

# ========================================================================
# 4. Visualize Clusters
# ========================================================================
print("\n" + "="*70)
print("4. VISUALIZING CLUSTERS")
print("="*70)

# K-Means clusters
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot K-Means
scatter = axes[0].scatter(df_geo['Start_Lng'], df_geo['Start_Lat'], 
                          c=df_geo['Cluster_KMeans'], cmap='tab10', 
                          s=1, alpha=0.5)
axes[0].set_xlabel('Longitude', fontsize=12)
axes[0].set_ylabel('Latitude', fontsize=12)
axes[0].set_title(f'K-Means Clustering (k={optimal_k})', 
                  fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=axes[0], label='Cluster')

# Plot DBSCAN
scatter = axes[1].scatter(df_geo['Start_Lng'], df_geo['Start_Lat'], 
                          c=df_geo['Cluster_DBSCAN'], cmap='tab10', 
                          s=1, alpha=0.5)
axes[1].set_xlabel('Longitude', fontsize=12)
axes[1].set_ylabel('Latitude', fontsize=12)
axes[1].set_title(f'DBSCAN Clustering ({n_clusters} clusters)', 
                  fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig('../results/figures/07_geographic_clusters.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 07_geographic_clusters.png")
plt.close()

# ========================================================================
# 5. Analyze Clusters
# ========================================================================
print("\n" + "="*70)
print("5. CLUSTER ANALYSIS")
print("="*70)

# Analyze severity by cluster
cluster_severity = df_geo.groupby('Cluster_KMeans')['Severity'].agg(['mean', 'count'])
cluster_severity = cluster_severity.sort_values('mean', ascending=False)

print("\nCluster Analysis (K-Means):")
print("Cluster | Avg Severity | Accident Count")
print("-" * 45)
for idx, row in cluster_severity.iterrows():
    print(f"   {idx:2d}   |     {row['mean']:.2f}     |   {int(row['count']):,}")

# Plot cluster severity
plt.figure(figsize=(12, 6))
cluster_severity['mean'].plot(kind='bar', color='coral')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Average Severity', fontsize=12)
plt.title('Average Accident Severity by Cluster', fontsize=14, fontweight='bold')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/08_cluster_severity.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 08_cluster_severity.png")
plt.close()

# ========================================================================
# 6. Save Clustering Results
# ========================================================================
print("\n" + "="*70)
print("6. SAVING RESULTS")
print("="*70)

output_file = '../data/processed/accidents_clustered.csv'
df_geo.to_csv(output_file, index=False)
print(f"✓ Saved clustered data to: {output_file}")

# Save cluster summary
summary = cluster_severity.to_string()
with open('../results/reports/cluster_summary.txt', 'w') as f:
    f.write("ACCIDENT HOTSPOT CLUSTERING ANALYSIS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total locations analyzed: {len(df_geo):,}\n")
    f.write(f"K-Means clusters: {optimal_k}\n")
    f.write(f"DBSCAN clusters: {n_clusters}\n\n")
    f.write("Cluster Details:\n")
    f.write(summary)

print(f"✓ Saved cluster summary")

print(f"\n{'='*70}")
print("PHASE 4 COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")
