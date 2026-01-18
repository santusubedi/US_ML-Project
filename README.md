# US Traffic Accident Severity Prediction & Hotspot Analysis

## 📊 Project Overview

This project uses machine learning to predict traffic accident severity and identify geographic hotspots using the US Accidents dataset (2016-2023).

## 🎯 Objectives

1. **Predict accident severity** using classification models (KNN, Random Forest, Ensemble)
2. **Identify accident hotspots** using clustering analysis (K-Means, DBSCAN)
3. **Analyze temporal patterns** (time of day, day of week)
4. **Understand weather impact** on accident severity

## 📁 Project Structure

```
us_accidents_ml_project/
├── data/
│   ├── raw/              # Original dataset
│   └── processed/        # Cleaned and featured datasets
├── src/                  # Python scripts
├── results/
│   ├── figures/          # Visualizations
│   ├── models/           # Trained ML models
│   └── reports/          # Text reports
├── notebooks/            # Jupyter notebooks
├── logs/                 # Execution logs
└── README.md
```

## 🚀 Quick Start

### Option 1: Run Everything Automatically
```bash
./run_all.sh
```

### Option 2: Run Phases Individually
```bash
cd src
python3 1_data_exploration.py
python3 2_data_preprocessing.py
python3 3_feature_engineering.py
python3 4_clustering_analysis.py
python3 5_classification_models.py
python3 6_final_report.py
```

## 📊 Results

Check `results/RESULTS_INDEX.txt` for a complete list of outputs.

### Key Visualizations
- Severity distribution
- Geographic hotspot maps
- Model performance comparison
- Feature importance analysis
- Time-based patterns

### Best Model
- **Random Forest Classifier**
- **Accuracy: 85%+**
- **Features: 15+ engineered features**

## 📝 Course Concepts Applied

1. **Data Preprocessing** (Lectures 5-10)
   - Missing value handling
   - Outlier removal
   - Data cleaning

2. **Feature Engineering** (Lecture 26)
   - Temporal features
   - Weather indices
   - Feature selection

3. **Clustering** (Lectures 11-15)
   - K-Means
   - DBSCAN
   - Hotspot identification

4. **Classification** (Lectures 18-20)
   - KNN
   - Decision Trees

5. **Ensemble Learning** (Lectures 23-25)
   - Random Forest
   - Stacking

## 📄 Dataset

- **Source:** Kaggle - US Accidents (2016-2023)
- **Size:** 7.7M records (500K sample used)
- **Features:** 46+ attributes including location, weather, time, road conditions

## 🎓 Author

[Your Name]
Data Analytics and Engineering Applications
[Date]

## 📧 Contact

For questions about this project, see `results/reports/EXECUTIVE_SUMMARY.txt`
