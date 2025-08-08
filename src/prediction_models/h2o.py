import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from h2o.automl import H2OAutoML
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import h2o

# ============================================================
# 1. Load Dataset
# ============================================================
df = pd.read_parquet('dataset.parquet')

# Map target values & ensure categorical
df['book_state'] = df['book_state'].map({0: 'pass', 2: 'fail'})
df['book_state'] = df['book_state'].astype('category')

# Clean column names for H2O
df.columns = df.columns.str.replace(' ', '_')

# Train/test split with stratification
df_train, df_test = train_test_split(
    df, test_size=0.2, stratify=df['book_state'], random_state=666
)

# ============================================================
# 2. Initialize H2O
# ============================================================
h2o.init()

# Convert pandas → H2OFrame
h2o_frame = h2o.H2OFrame(df_train)
h2o_frame_test = h2o.H2OFrame(df_test)

# Define features/target
x = h2o_frame.columns
y = 'book_state'
x.remove(y)

# ============================================================
# 3. Run AutoML
# ============================================================
h2o_automl = H2OAutoML(
    max_runtime_secs=1800,  # 30 min
    seed=666,
    sort_metric='AUC',
    include_algos=['GBM', 'XGBoost', 'DRF', 'GLM', 'DeepLearning'],
    nfolds=5,
    balance_classes=True,
    keep_cross_validation_predictions=True,
    keep_cross_validation_models=True
)

h2o_automl.train(x=x, y=y, training_frame=h2o_frame)

# ============================================================
# 4. Leaderboard & Model Suggestion
# ============================================================
h2o_models = h2o.automl.get_leaderboard(h2o_automl, extra_columns="ALL")
print("\n=== Model Leaderboard ===")
print(h2o_models.as_data_frame().to_string(index=False))

# Extract best algorithm name
leader_id = h2o_automl.leader.model_id
leader_algo = leader_id.split('_')[0]  # crude extraction from model_id naming convention
print(f"\nSuggested Algorithm for Fine-Tuning: {leader_algo}")

# ============================================================
# 5. Predictions with Leader Model
# ============================================================
leader_model = h2o_automl.leader
y_pred = leader_model.predict(h2o_frame_test).as_data_frame()
y_actual = df_test['book_state'].reset_index(drop=True)

# Metrics
y_pred_probs = y_pred['fail']  # probability of 'fail'
roc_auc = roc_auc_score((y_actual == 'fail').astype(int), y_pred_probs)
accuracy = (y_pred['predict'] == y_actual).mean()
class_report = classification_report(y_actual, y_pred['predict'], output_dict=False)

print("\n=== Leader Model Performance ===")
print(f"Test Set Accuracy: {accuracy:.4f}")
print(f"Test Set ROC AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_actual, y_pred['predict']))

# ============================================================
# 6. Plots
# ============================================================

# Confusion Matrix
cm = confusion_matrix(y_actual, y_pred['predict'], labels=['pass', 'fail'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['pass', 'fail'],
            yticklabels=['pass', 'fail'])
plt.title('Confusion Matrix - Leader Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve((y_actual == 'fail').astype(int), y_pred_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Leader Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ============================================================
# 7. Variable Importance (if supported)
# ============================================================
if hasattr(leader_model, 'varimp_plot'):
    print("\n=== Variable Importance ===")
    leader_model.varimp_plot()
else:
    print("\nLeader model does not support variable importance.")

# ============================================================
# 8. Compare Top 5 Models
# ============================================================
print("\n=== Top 5 Models Comparison ===")
top_models = h2o_models.as_data_frame().head(5)
for model_id in top_models['model_id']:
    model = h2o.get_model(model_id)
    preds = model.predict(h2o_frame_test).as_data_frame()
    if 'predict' in preds.columns:
        acc = (preds['predict'] == y_actual).mean()
        print(f"Model: {model_id}, Test Accuracy: {acc:.4f}")
