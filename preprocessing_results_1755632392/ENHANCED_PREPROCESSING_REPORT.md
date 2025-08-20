# Enhanced Scientific Preprocessing Report

**Generated:** 2025-08-19 21:58:21
**Processing Time:** 1103.5 seconds

## Executive Summary
- **Original:** 5,108,820 × 50
- **Final:** 5,108,820 × 29
- **Features removed:** 21
- **Target classes:** 3
- **Imbalance ratio:** 43660.9:1

## Enhanced Encoding Applied

## Feature Importance Analysis
- **Methods used:** random_forest
- **Features selected:** 28
- **Top 10 features:** sequence_number, panel_position_number, station_desc, workstep_number_erp, workstep_id, line_id, recipe_revision_id, measure_step_number, measure_value, measure_fail_code

## User Features Analysis
- **Requested:** 31
- **Preserved:** 0
- **Removed user features:** 31
  - High Missing: station_diag_desc
  - Constant: workstep_number_alt, plant_desc
  - Correlation: workstep_number_mes

## Imbalance Handling
- **Method:** smote
- **Original samples:** 5,108,820
- **Final samples:** 5,108,820

## Enhanced Usage Guide
```python
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Load preprocessed data
df = pd.read_parquet('FINAL_PREPROCESSED_DATASET.parquet')
X = df.drop('book_state', axis=1)
y = df['book_state']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Recommended models for extreme imbalance
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42),
    'LogisticRegression': LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000)
}

# Evaluation for imbalanced data
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'{name} F1-Score: {f1:.4f}')
    print(classification_report(y_test, y_pred))
```
