# Data Classification Using AI 🤖

> Project 2 | DecodeLabs Industrial Training | Batch 2026
> Developer: Mayar Yasser

## Overview
A machine learning classification model built using the KNN algorithm
to classify Iris flowers into 3 species based on their measurements.

## Dataset
- Name: Iris Dataset
- Samples: 150 flowers
- Classes: Setosa, Versicolor, Virginica
- Features: Sepal Length, Sepal Width, Petal Length, Petal Width

## Pipeline
- Data Loading & Feature Scaling (StandardScaler)
- Train-Test Split (80% Training / 20% Testing)
- KNN Classification (n_neighbors=5)
- Confusion Matrix & F1 Score

## How to Run
```bash
pip install scikit-learn pandas matplotlib seaborn
python classifier.py
```

## Results
- Confusion Matrix saved as confusion_matrix.png
- F1 Score: ~97%

## Tools Used
- Python
- scikit-learn
- matplotlib
- seaborn

## Developer
Mayar Yasser | DecodeLabs Batch 2026