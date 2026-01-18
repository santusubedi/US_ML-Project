"""
Phase 5: Classification Models - Accident Severity Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 5: CLASSIFICATION MODELS")
print("="*70)

# Load featured dataset
print("\nLoading featured dataset...")
df = pd.read_csv('../data/processed/accidents_featured.csv')
print(f"✓ Loaded {df.shape[0]:,} rows")

# ========================================================================
# 1. Prepare Data
# ========================================================================
print("\n" + "="*70)
print("1. PREPARING DATA FOR MODELING")
print("="*70)

# Separate features and target
X = df.drop(['Severity', 'Start_Lat', 'Start_Lng'], axis=1, errors='ignore')
y = df['Severity']

# Remove any remaining non-numeric columns
X = X.select_dtypes(include=[np.number])

print(f"Features shape: {X.shape}")
print(f"Target distribution:")
print(y.value_counts().sort_index())

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features standardized")

# ========================================================================
# 2. K-Nearest Neighbors (KNN)
# ========================================================================
print("\n" + "="*70)
print("2. TRAINING K-NEAREST NEIGHBORS (KNN)")
print("="*70)

# Find optimal k
print("Finding optimal k...")
k_values = [3, 5, 7, 9, 11, 15]
k_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    k_scores.append(scores.mean())
    print(f"  k={k:2d}: CV accuracy={scores.mean():.4f}")

# Plot k selection
plt.figure(figsize=(10, 6))
plt.plot(k_values, k_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('K Value', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('KNN: Optimal K Selection', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/figures/09_knn_k_selection.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 09_knn_k_selection.png")
plt.close()

# Train with best k
best_k = k_values[np.argmax(k_scores)]
print(f"\nTraining KNN with k={best_k}...")

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)

print(f"✓ KNN Accuracy: {knn_accuracy:.4f}")

# ========================================================================
# 3. Decision Tree
# ========================================================================
print("\n" + "="*70)
print("3. TRAINING DECISION TREE")
print("="*70)

dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print(f"✓ Decision Tree Accuracy: {dt_accuracy:.4f}")

# ========================================================================
# 4. Random Forest
# ========================================================================
print("\n" + "="*70)
print("4. TRAINING RANDOM FOREST")
print("="*70)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, 
                                   random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"✓ Random Forest Accuracy: {rf_accuracy:.4f}")

# ========================================================================
# 5. Ensemble - Stacking
# ========================================================================
print("\n" + "="*70)
print("5. TRAINING STACKING ENSEMBLE")
print("="*70)

# Base models
estimators = [
    ('knn', KNeighborsClassifier(n_neighbors=best_k)),
    ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42))
]

# Meta-learner
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

print("Training stacking ensemble (this may take a few minutes)...")
stacking_model.fit(X_train_scaled, y_train)
stacking_pred = stacking_model.predict(X_test_scaled)
stacking_accuracy = accuracy_score(y_test, stacking_pred)

print(f"✓ Stacking Ensemble Accuracy: {stacking_accuracy:.4f}")

# ========================================================================
# 6. Model Comparison
# ========================================================================
print("\n" + "="*70)
print("6. MODEL COMPARISON")
print("="*70)

results = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree', 'Random Forest', 'Stacking Ensemble'],
    'Accuracy': [knn_accuracy, dt_accuracy, rf_accuracy, stacking_accuracy]
}).sort_values('Accuracy', ascending=False)

print("\nModel Performance:")
print(results.to_string(index=False))

# Plot comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(results['Model'], results['Accuracy'], color=['steelblue', 'coral', 'green', 'purple'])
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim([0.7, 1.0])
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('../results/figures/10_model_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 10_model_comparison.png")
plt.close()

# ========================================================================
# 7. Best Model - Detailed Analysis
# ========================================================================
print("\n" + "="*70)
print("7. BEST MODEL DETAILED ANALYSIS")
print("="*70)

# Use Random Forest as best model
best_model = rf_model
best_pred = rf_pred
best_name = "Random Forest"

print(f"\nDetailed results for {best_name}:")
print("\nClassification Report:")
print(classification_report(y_test, best_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
plt.ylabel('True Severity', fontsize=12)
plt.xlabel('Predicted Severity', fontsize=12)
plt.title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/11_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 11_confusion_matrix.png")
plt.close()

# ========================================================================
# 8. Save Models
# ========================================================================
print("\n" + "="*70)
print("8. SAVING MODELS")
print("="*70)

joblib.dump(knn_model, '../results/models/knn_model.pkl')
joblib.dump(dt_model, '../results/models/dt_model.pkl')
joblib.dump(rf_model, '../results/models/rf_model.pkl')
joblib.dump(stacking_model, '../results/models/stacking_model.pkl')
joblib.dump(scaler, '../results/models/scaler.pkl')

print(f"✓ Saved all models")

# Save results summary
with open('../results/reports/model_results.txt', 'w') as f:
    f.write("ACCIDENT SEVERITY PREDICTION - MODEL RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(results.to_string(index=False))
    f.write(f"\n\nBest Model: {best_name}\n")
    f.write(f"Accuracy: {rf_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, best_pred))

print(f"✓ Saved results summary")

print(f"\n{'='*70}")
print("PHASE 5 COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")
