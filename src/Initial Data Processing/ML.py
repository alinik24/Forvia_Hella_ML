import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tqdm import tqdm
import gc

# Define paths
base_path = 'C:/Users/alina/OneDrive/Desktop/RWML projects/data_hella/'
measurements_csv = os.path.join(base_path, 'station_outputs_full/non_label_measurements.csv')
bookings_parquet = os.path.join(base_path, 'bookings.parquet')
output_dir = os.path.join(base_path, 'station_outputs_full/models')
os.makedirs(output_dir, exist_ok=True)

# Select a high-volume station
station_id = '1405e64b'  # Example: station with 39.89M rows
batch_size = 100000  # Adjust based on RAM (e.g., 50000 for 8GB, 100000 for 16GB)

# Step 1: Pivot measurements for the station in chunks
print(f"Pivoting measurements for station {station_id}...")
feature_dfs = []
total_rows = 39887993  # From non_label_stations_summary.csv for 1405e64b
with tqdm(total=total_rows, desc="Processing rows", unit="rows") as pbar:
    for chunk in pd.read_csv(measurements_csv, chunksize=batch_size):
        station_chunk = chunk[chunk['station_id'] == station_id]
        if not station_chunk.empty:
            pivot_chunk = station_chunk.pivot_table(
                index='serial_number',
                columns='measurement_name',
                values='measure_value',
                aggfunc='mean'
            ).reset_index()
            feature_dfs.append(pivot_chunk)
        pbar.update(len(chunk[chunk['station_id'] == station_id]))
        del chunk, station_chunk, pivot_chunk
        gc.collect()

if not feature_dfs:
    print(f"No data for station {station_id}. Skipping.")
    exit()

# Combine chunks
features = pd.concat(feature_dfs, ignore_index=True)
features = features.groupby('serial_number').mean().reset_index()  # Aggregate duplicates
features.to_parquet(os.path.join(output_dir, f'station_{station_id}_features.parquet'))
print(f"Feature matrix saved for station {station_id}: {len(features)} serial_numbers")

# Step 2: Merge with bookings
print("Merging with bookings.parquet...")
bookings = pd.read_parquet(bookings_parquet)
data = features.merge(bookings[['serial_number', 'book_state']], on='serial_number', how='inner')

# Check data sufficiency
if len(data) < 20:
    print(f"Skipping station {station_id}: Too few samples ({len(data)})")
    exit()
if data['book_state'].nunique() < 2:
    print(f"Skipping station {station_id}: Only one class in book_state")
    exit()

# Step 3: Prepare features and labels
X = data.drop(['serial_number', 'book_state'], axis=1)
y = data['book_state']

# Handle missing values
X = X.fillna(X.mean())

# Encode categorical features (if any)
X = pd.get_dummies(X, drop_first=True)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Train Random Forest
print(f"Training Random Forest for station {station_id}...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)  # For global model
print(f"Classification Report for station {station_id}:")
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Confusion Matrix for station {station_id}:")
print(confusion_matrix(y_test, y_pred))

# Save predictions for global model
pred_df = pd.DataFrame({
    'serial_number': data.loc[y_test.index, 'serial_number'],
    f'{station_id}_proba_pass': y_pred_proba[:, 0],
    f'{station_id}_proba_suspect': y_pred_proba[:, 1],
    f'{station_id}_proba_fail': y_pred_proba[:, 2]
})
pred_df.to_parquet(os.path.join(output_dir, f'station_{station_id}_predictions.parquet'))

# Step 6: Feature Importance with SHAP
print(f"Computing SHAP values for station {station_id}...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[2], X_test, feature_names=X.columns, show=False)
plt.title(f"SHAP Summary Plot for Fail Class - Station {station_id}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'shap_summary_fail_station_{station_id}.png'))
plt.close()

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(shap_values[2]).mean(axis=0)
}).sort_values(by='importance', ascending=False).head(10)
feature_importance.to_csv(os.path.join(output_dir, f'top_features_station_{station_id}.csv'), index=False)
print(f"Top Features for station {station_id}:")
print(feature_importance)

# Step 7: Anomaly Detection
print(f"Running anomaly detection for station {station_id}...")
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X_scaled)

data['anomaly'] = anomalies
anomaly_data = data[data['anomaly'] == -1][['serial_number', 'book_state']]
anomaly_report = anomaly_data.groupby('book_state').size().reset_index(name='count')
anomaly_report.to_csv(os.path.join(output_dir, f'anomaly_report_station_{station_id}.csv'), index=False)
print(f"Anomaly Detection Report for station {station_id}:")
print(anomaly_report)

# Step 8: Save models
joblib.dump(rf_model, os.path.join(output_dir, f'predictive_model_station_{station_id}.pkl'))
joblib.dump(iso_forest, os.path.join(output_dir, f'anomaly_model_station_{station_id}.pkl'))
print(f"Models saved for station {station_id}")