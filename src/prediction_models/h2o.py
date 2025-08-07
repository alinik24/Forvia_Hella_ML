import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from h2o.automl import H2OAutoML
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

import h2o

# Load the dataset from Parquet
df = pd.read_parquet('dataset.parquet')

# Prepare features and target
df['book_state'] = df['book_state'].astype('category')  # Ensure book_state is categorical
df['book_state'] = df['book_state'].map({0: 'pass', 2: 'fail'})  # Map 0 to 'pass', 2 to 'fail'
df.columns = df.columns.str.replace(' ', '_')  # Clean column names for H2O

# Split train and test sets
sample_size = int(len(df) * 0.8)  # 80% train, 20% test
df_train = df.iloc[:sample_size].copy()
df_test = df.iloc[sample_size:].copy()

# Initialize H2O
h2o.init()

# Convert to H2OFrame
h2o_frame = h2o.H2OFrame(df_train)
h2o_frame_test = h2o.H2OFrame(df_test)

# Define features and target
x = h2o_frame.columns
y = 'book_state'
x.remove(y)

# Configure AutoML for binary classification
h2o_automl = H2OAutoML(
    max_runtime_secs=5 * 60,  # 5 minutes
    seed=666,
    sort_metric='logloss',  # Primary metric for binary classification
    include_algos=['DRF', 'GLM', 'XGBoost', 'GBM', 'DeepLearning', 'NaiveBayes', 'StackedEnsemble'],
    # All H2O classifiers
    nfolds=5,  # 5-fold cross-validation
    keep_cross_validation_predictions=True,  # For detailed analysis
    export_checkpoints_dir=None  # Avoid saving models to disk
)

# Train models
h2o_automl.train(x=x, y=y, training_frame=h2o_frame)

# Get leaderboard with additional metrics
h2o_models = h2o.automl.get_leaderboard(h2o_automl, extra_columns="ALL")

# Make predictions with the leader model
leader_model = h2o_automl.leader
y_pred = leader_model.predict(h2o_frame_test)

# Convert predictions to pandas for evaluation
y_pred_df = y_pred.as_data_frame()
y_actual = df_test['book_state'].reset_index(drop=True)
y_pred_labels = y_pred_df['predict']
y_pred_probs = y_pred_df['fail']  # Probability for 'fail' class

# Calculate evaluation metrics
accuracy = (y_pred_labels == y_actual).mean()
roc_auc = roc_auc_score(y_actual, y_pred_probs, labels=['pass', 'fail'])
class_report = classification_report(y_actual, y_pred_labels, output_dict=True)

# Create comparison DataFrame
h2o_compare = pd.DataFrame({
    'actual': y_actual,
    'predicted': y_pred_labels,
    'prob_fail': y_pred_probs
})

# Print detailed results
print("\n=== Model Leaderboard ===")
print(h2o_models.as_data_frame().to_string(index=False))

print("\n=== Leader Model Performance ===")
print(f"Test Set Accuracy: {accuracy:.4f}")
print(f"Test Set ROC AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_actual, y_pred_labels))

# Plot confusion matrix
cm = confusion_matrix(y_actual, y_pred_labels, labels=['pass', 'fail'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['pass', 'fail'],
            yticklabels=['pass', 'fail'])
plt.title('Confusion Matrix - Leader Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_actual, y_pred_probs, pos_label='fail')
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Leader Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Variable importance plot for leader model (if available)
if hasattr(leader_model, 'varimp_plot'):
    print("\n=== Variable Importance ===")
    leader_model.varimp_plot()

# Compare top models
top_models = h2o_models.as_data_frame().head(5)
print("\n=== Top 5 Models Comparison ===")
for model_id in top_models['model_id']:
    model = h2o.get_model(model_id)
    preds = model.predict(h2o_frame_test).as_data_frame()
    acc = (preds['predict'] == y_actual).mean()
    print(f"Model: {model_id}, Test Accuracy: {acc:.4f}")
