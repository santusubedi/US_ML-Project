"""
Phase 3: Feature Engineering and Selection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 3: FEATURE ENGINEERING AND SELECTION")
print("="*70)

# Load cleaned data
print("\nLoading cleaned dataset...")
df = pd.read_csv('../data/processed/accidents_cleaned.csv')
print(f"✓ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

# ========================================================================
# 1. Encode Categorical Variables
# ========================================================================
print("\n" + "="*70)
print("1. ENCODING CATEGORICAL VARIABLES")
print("="*70)

# Store original for later
df_original = df.copy()

# Label encode categorical features
label_encoders = {}
categorical_cols = ['Weather_Condition', 'City', 'State', 'Season']
categorical_cols = [col for col in categorical_cols if col in df.columns]

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"✓ Encoded {col}: {len(le.classes_)} unique values")

# Convert boolean columns to int
boolean_cols = ['Junction', 'Traffic_Signal', 'Crossing', 'Station', 'Stop']
boolean_cols = [col for col in boolean_cols if col in df.columns]

for col in boolean_cols:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype(int)
        print(f"✓ Converted {col} to int")

# ========================================================================
# 2. Feature Selection using Random Forest
# ========================================================================
print("\n" + "="*70)
print("2. FEATURE SELECTION")
print("="*70)

# Prepare features for selection
feature_cols = [col for col in df.columns if col not in [
    'Severity', 'Start_Lat', 'Start_Lng', 'Weather_Condition', 
    'City', 'State', 'Season'
]]

# Remove any remaining non-numeric columns
feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]

X = df[feature_cols].fillna(0)
y = df['Severity']

print(f"Training Random Forest for feature importance...")
print(f"Features shape: {X.shape}")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n✓ Feature importance calculated")
print(f"\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_15 = feature_importance.head(15)
plt.barh(range(len(top_15)), top_15['Importance'], color='steelblue')
plt.yticks(range(len(top_15)), top_15['Feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Most Important Features for Severity Prediction', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/04_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 04_feature_importance.png")
plt.close()

# Select top features (threshold: cumulative 90% importance)
cumsum = feature_importance['Importance'].cumsum()
n_features = (cumsum <= 0.90).sum() + 1
selected_features = feature_importance.head(n_features)['Feature'].tolist()

print(f"\n✓ Selected {len(selected_features)} features explaining 90% variance")

# ========================================================================
# 3. Create Final Feature Set
# ========================================================================
print("\n" + "="*70)
print("3. CREATING FINAL FEATURE SET")
print("="*70)

# Keep Severity, location, and selected features
final_cols = ['Severity', 'Start_Lat', 'Start_Lng'] + selected_features
final_cols = [col for col in final_cols if col in df.columns]

df_final = df[final_cols].copy()

print(f"✓ Final dataset shape: {df_final.shape}")
print(f"✓ Selected features: {len(selected_features)}")

# ========================================================================
# 4. Correlation Analysis
# ========================================================================
print("\n" + "="*70)
print("4. CORRELATION ANALYSIS")
print("="*70)

# Calculate correlation with Severity
correlations = df_final.drop(['Start_Lat', 'Start_Lng'], axis=1, errors='ignore').corrwith(df_final['Severity']).sort_values(ascending=False)
print("\nTop 10 Features Correlated with Severity:")
print(correlations.head(10).to_string())

# Plot correlation heatmap (top features only)
top_features_for_heatmap = correlations.abs().sort_values(ascending=False).head(15).index.tolist()
plt.figure(figsize=(12, 10))
correlation_matrix = df_final[top_features_for_heatmap].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap (Top 15 Features)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 05_correlation_heatmap.png")
plt.close()

# ========================================================================
# 5. Save Final Dataset
# ========================================================================
print("\n" + "="*70)
print("5. SAVING FINAL DATASET")
print("="*70)

output_file = '../data/processed/accidents_featured.csv'
df_final.to_csv(output_file, index=False)
print(f"✓ Saved to: {output_file}")

# Save feature list
with open('../data/processed/selected_features.txt', 'w') as f:
    f.write('\n'.join(selected_features))
print(f"✓ Saved selected features list")

print(f"\n{'='*70}")
print("PHASE 3 COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")
