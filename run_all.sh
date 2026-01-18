#!/bin/bash

# Master script to run all phases
echo "=================================="
echo "RUNNING ALL PROJECT PHASES"
echo "=================================="

cd src

echo ""
echo ">>> PHASE 1: Data Exploration"
python3 1_data_exploration.py
echo ""

echo ">>> PHASE 2: Data Preprocessing"
python3 2_data_preprocessing.py
echo ""

echo ">>> PHASE 3: Feature Engineering"
python3 3_feature_engineering.py
echo ""

echo ">>> PHASE 4: Clustering Analysis"
python3 4_clustering_analysis.py
echo ""

echo ">>> PHASE 5: Classification Models"
python3 5_classification_models.py
echo ""

echo ">>> PHASE 6: Final Report"
python3 6_final_report.py
echo ""

cd ..

echo "=================================="
echo "ALL PHASES COMPLETED!"
echo "=================================="
echo ""
echo "Results available in:"
echo "  - results/figures/ (visualizations)"
echo "  - results/models/ (trained models)"
echo "  - results/reports/ (text reports)"
echo ""
