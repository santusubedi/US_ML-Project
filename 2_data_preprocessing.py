"""
Phase 2: Data Cleaning and Preprocessing
"""
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 2: DATA CLEANING AND PREPROCESSING")
print("="*70)

# Load dataset
print("\nLoading dataset...")
data_files = [f for f in os.listdir('../data/raw') if f.endswith('.csv')]
if not data_files:
    print("ERROR: No CSV file found!")
    exit(1)

df = pd.read_csv(f'../data/raw/{data_files[0]}', nrows=500000)
print(f"✓ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

original_shape = df.shape

# ========================================================================
# 1. Handle Missing Values
# ========================================================================
print("\n" + "="*70)
print("1. HANDLING MISSING VALUES")
print("="*70)

# Drop columns with >40% missing values
missing_threshold = 0.4
cols_to_drop = []

for col in df.columns:
    missing_pct = df[col].isnull().sum() / len(df)
    if missing_pct > missing_threshold:
        cols_to_drop.append(col)

print(f"Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing:")
print(cols_to_drop)
df = df.drop(columns=cols_to_drop)

# Fill numerical missing values with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"✓ Filled {col} with median: {median_val:.2f}")

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
        df[col].fillna(mode_val, inplace=True)
        print(f"✓ Filled {col} with mode: {mode_val}")

print(f"\n✓ Missing values handled. Remaining nulls: {df.isnull().sum().sum()}")

# ========================================================================
# 2. Remove Duplicates
# ========================================================================
print("\n" + "="*70)
print("2. REMOVING DUPLICATES")
print("="*70)

duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()
duplicates_removed = duplicates_before - df.duplicated().sum()
print(f"✓ Removed {duplicates_removed:,} duplicate rows")

# ========================================================================
# 3. Parse Datetime Features
# ========================================================================
print("\n" + "="*70)
print("3. PARSING DATETIME FEATURES")
print("="*70)

if 'Start_Time' in df.columns:
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
    df['Month'] = df['Start_Time'].dt.month
    df['Year'] = df['Start_Time'].dt.year
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    print("✓ Created: Hour, DayOfWeek, Month, Year, IsWeekend")

# ========================================================================
# 4. Feature Engineering
# ========================================================================
print("\n" + "="*70)
print("4. FEATURE ENGINEERING")
print("="*70)

# Rush hour indicator (7-9 AM, 4-7 PM)
if 'Hour' in df.columns:
    df['IsRushHour'] = df['Hour'].apply(
        lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0
    )
    print("✓ Created: IsRushHour")

# Season
if 'Month' in df.columns:
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['Season'] = df['Month'].apply(get_season)
    print("✓ Created: Season")

# Weather severity index
weather_cols = ['Temperature(F)', 'Visibility(mi)', 'Precipitation(in)']
weather_cols = [col for col in weather_cols if col in df.columns]

if len(weather_cols) >= 2:
    # Normalize and create severity score
    if 'Visibility(mi)' in df.columns:
        df['Low_Visibility'] = (df['Visibility(mi)'] < 1).astype(int)
    if 'Precipitation(in)' in df.columns:
        df['High_Precipitation'] = (df['Precipitation(in)'] > 0.1).astype(int)
    print("✓ Created: Weather severity indicators")

# ========================================================================
# 5. Remove Outliers (IQR Method)
# ========================================================================
print("\n" + "="*70)
print("5. REMOVING OUTLIERS")
print("="*70)

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

outlier_cols = ['Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)']
outlier_cols = [col for col in outlier_cols if col in df.columns]

rows_before = len(df)
for col in outlier_cols:
    df = remove_outliers_iqr(df, col)

rows_removed = rows_before - len(df)
print(f"✓ Removed {rows_removed:,} outlier rows")

# ========================================================================
# 6. Select Key Features
# ========================================================================
print("\n" + "="*70)
print("6. SELECTING KEY FEATURES")
print("="*70)

# Define key features to keep
key_features = [
    'Severity', 'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Visibility(mi)',
    'Precipitation(in)', 'Wind_Speed(mph)', 'Weather_Condition', 'City', 'State',
    'Junction', 'Traffic_Signal', 'Crossing', 'Station', 'Stop',
    'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsRushHour', 'Season'
]

# Keep only existing columns
available_features = [col for col in key_features if col in df.columns]
df_clean = df[available_features].copy()

print(f"✓ Selected {len(available_features)} key features")
print(f"Features: {available_features}")

# ========================================================================
# 7. Save Cleaned Dataset
# ========================================================================
print("\n" + "="*70)
print("7. SAVING CLEANED DATASET")
print("="*70)

os.makedirs('../data/processed', exist_ok=True)
output_file = '../data/processed/accidents_cleaned.csv'
df_clean.to_csv(output_file, index=False)

print(f"\n{'='*70}")
print("PREPROCESSING SUMMARY")
print(f"{'='*70}")
print(f"Original shape: {original_shape}")
print(f"Cleaned shape: {df_clean.shape}")
print(f"Rows removed: {original_shape[0] - df_clean.shape[0]:,} ({100*(original_shape[0]-df_clean.shape[0])/original_shape[0]:.1f}%)")
print(f"Columns removed: {original_shape[1] - df_clean.shape[1]}")
print(f"\n✓ Saved to: {output_file}")

print(f"\n{'='*70}")
print("PHASE 2 COMPLETED SUCCESSFULLY!")
print(f"{'='*70}")
