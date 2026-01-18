"""
Phase 1: Data Exploration and Initial Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# Create results directory
os.makedirs('../results/figures', exist_ok=True)
os.makedirs('../logs', exist_ok=True)

print("="*70)
print("PHASE 1: DATA EXPLORATION AND INITIAL ANALYSIS")
print("="*70)

# Start logging
log_file = open('../logs/exploration_log.txt', 'w')

def log_print(message):
    """Print and log message"""
    print(message)
    log_file.write(message + '\n')

log_print(f"\nStarted at: {datetime.now()}\n")

# Load dataset
log_print("Loading US Accidents dataset...")
try:
    # Try to find the CSV file
    data_files = [f for f in os.listdir('../data/raw') if f.endswith('.csv')]
    if not data_files:
        log_print("ERROR: No CSV file found in data/raw/")
        log_print("Please download the dataset from Kaggle and place it in data/raw/")
        exit(1)
    
    data_file = data_files[0]
    log_print(f"Found data file: {data_file}")
    
    # Load with sampling for faster processing (use 500k rows)
    df = pd.read_csv(f'../data/raw/{data_file}', nrows=500000)
    log_print(f"✓ Dataset loaded successfully! Shape: {df.shape}")
    
except Exception as e:
    log_print(f"ERROR loading dataset: {e}")
    exit(1)

# Basic information
log_print(f"\n{'='*70}")
log_print("DATASET OVERVIEW")
log_print(f"{'='*70}")
log_print(f"Number of rows: {df.shape[0]:,}")
log_print(f"Number of columns: {df.shape[1]}")
log_print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Column information
log_print(f"\n{'='*70}")
log_print("COLUMN INFORMATION")
log_print(f"{'='*70}")
log_print(f"\nColumns: {list(df.columns)}\n")

# Data types
log_print("\nData Types:")
log_print(str(df.dtypes))

# Missing values analysis
log_print(f"\n{'='*70}")
log_print("MISSING VALUES ANALYSIS")
log_print(f"{'='*70}")
missing = df.isnull().sum()
missing_percent = 100 * missing / len(df)
missing_table = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Percent': missing_percent
})
missing_table = missing_table[missing_table['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)
log_print(f"\n{missing_table.to_string()}")

# Severity distribution
log_print(f"\n{'='*70}")
log_print("SEVERITY DISTRIBUTION")
log_print(f"{'='*70}")
if 'Severity' in df.columns:
    severity_dist = df['Severity'].value_counts().sort_index()
    log_print(f"\n{severity_dist.to_string()}")
    
    # Plot severity distribution
    plt.figure(figsize=(10, 6))
    severity_dist.plot(kind='bar', color='steelblue')
    plt.title('Accident Severity Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Severity Level', fontsize=12)
    plt.ylabel('Number of Accidents', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/figures/01_severity_distribution.png', dpi=300, bbox_inches='tight')
    log_print("✓ Saved: 01_severity_distribution.png")
    plt.close()

# Top states
log_print(f"\n{'='*70}")
log_print("TOP 10 STATES BY ACCIDENT COUNT")
log_print(f"{'='*70}")
if 'State' in df.columns:
    top_states = df['State'].value_counts().head(10)
    log_print(f"\n{top_states.to_string()}")
    
    # Plot top states
    plt.figure(figsize=(12, 6))
    top_states.plot(kind='barh', color='coral')
    plt.title('Top 10 States by Accident Count', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Accidents', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/figures/02_top_states.png', dpi=300, bbox_inches='tight')
    log_print("✓ Saved: 02_top_states.png")
    plt.close()

# Weather conditions
log_print(f"\n{'='*70}")
log_print("TOP 10 WEATHER CONDITIONS")
log_print(f"{'='*70}")
if 'Weather_Condition' in df.columns:
    top_weather = df['Weather_Condition'].value_counts().head(10)
    log_print(f"\n{top_weather.to_string()}")

# Numerical features summary
log_print(f"\n{'='*70}")
log_print("NUMERICAL FEATURES SUMMARY")
log_print(f"{'='*70}")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
log_print(f"\n{df[numerical_cols].describe().to_string()}")

# Save processed info
log_print(f"\n{'='*70}")
log_print("SAVING EXPLORATION RESULTS")
log_print(f"{'='*70}")
df.info(buf=log_file)

# Create missing values heatmap for key columns
key_cols = ['Severity', 'Temperature(F)', 'Visibility(mi)', 'Precipitation(in)', 
            'Wind_Speed(mph)', 'Weather_Condition', 'City', 'State']
key_cols = [col for col in key_cols if col in df.columns]

plt.figure(figsize=(12, 8))
sns.heatmap(df[key_cols].isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap (Key Features)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/figures/03_missing_values_heatmap.png', dpi=300, bbox_inches='tight')
log_print("✓ Saved: 03_missing_values_heatmap.png")
plt.close()

log_print(f"\n{'='*70}")
log_print("PHASE 1 COMPLETED SUCCESSFULLY!")
log_print(f"{'='*70}")
log_print(f"Completed at: {datetime.now()}")

log_file.close()
print("\n✓ Exploration complete! Check logs/exploration_log.txt for details")
