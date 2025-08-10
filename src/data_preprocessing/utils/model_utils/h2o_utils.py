import h2o
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from h2o.automl import H2OAutoML
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    roc_curve
)
from sklearn.model_selection import train_test_split

# --- Helper functions for project structure ---
# These would typically be in their own files as per the original script,
# but for this example, we'll keep them here for a complete, runnable file.

def save_last_run(input_path, output_path, version):
    """Saves the last used file paths and version to a config file."""
    # In a real project, this would write to a JSON or YAML file.
    # We'll just print it for demonstration.
    print("\n--- Saving Last Run Configuration ---")
    print(f"input_path: {input_path}")
    print(f"output_path: {output_path}")
    print(f"version: {version}")

def load_last_run():
    """Loads the last used file paths from a config file."""
    # In a real project, this would read from a JSON or YAML file.
    # For now, we'll return an empty dict to simulate no previous run.
    return {"input_path": None, "output_path": None, "version": None}


def run_automl_pipeline(input_path, output_path):
    """
    Executes the full H2O AutoML pipeline.

    Args:
        input_path (str): Path to the input parquet dataset.
        output_path (str): Directory to save the output plots.
    """
    # ============================================================
    # 1. Load Dataset
    # ============================================================
    try:
        print(f"Loading dataset from {input_path}...")
        df = pd.read_parquet(input_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {input_path}")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Map target values & ensure categorical
    df['book_state'] = df['book_state'].map({0: 'pass', 2: 'fail'})
    df['book_state'] = df['book_state'].astype('category')

    # Clean column names for H2O
    df.columns = df.columns.str.replace(' ', '_')

    # Train/test split with stratification
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df['book_state'], random_state=666
    )

    # Convert pandas → H2OFrame
    h2o_frame = h2o.H2OFrame(df_train)
    h2o_frame_test = h2o.H2OFrame(df_test)

    # Define features/target
    x = h2o_frame.columns
    y = 'book_state'
    x.remove(y)

    # ============================================================
    # 2. Run AutoML
    # ============================================================
    print("\n--- Running H2O AutoML ---")
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
    # 3. Leaderboard & Model Suggestion
    # ============================================================
    h2o_models = h2o.automl.get_leaderboard(h2o_automl, extra_columns="ALL")
    print("\n=== Model Leaderboard ===")
    print(h2o_models.as_data_frame().to_string(index=False))

    leader_id = h2o_automl.leader.model_id
    leader_algo = leader_id.split('_')[0]
    print(f"\nSuggested Algorithm for Fine-Tuning: {leader_algo}")

    # ============================================================
    # 4. Predictions with Leader Model
    # ============================================================
    print("\n--- Evaluating Leader Model Performance ---")
    leader_model = h2o_automl.leader
    y_pred = leader_model.predict(h2o_frame_test).as_data_frame()
    y_actual = df_test['book_state'].reset_index(drop=True)

    # Metrics
    if 'fail' in y_pred.columns:
        y_pred_probs = y_pred['fail']
        roc_auc = roc_auc_score((y_actual == 'fail').astype(int), y_pred_probs)
    else:
        print("Warning: 'fail' column not found in predictions. Skipping ROC AUC calculation.")
        roc_auc = "N/A"

    accuracy = (y_pred['predict'] == y_actual).mean()
    class_report = classification_report(y_actual, y_pred['predict'], output_dict=False)

    print(f"Test Set Accuracy: {accuracy:.4f}")
    print(f"Test Set ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(class_report)

    # ============================================================
    # 5. Plots
    # ============================================================
    print("\n--- Generating Plots ---")

    # Confusion Matrix
    cm = confusion_matrix(y_actual, y_pred['predict'], labels=['pass', 'fail'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['pass', 'fail'],
                yticklabels=['pass', 'fail'])
    plt.title('Confusion Matrix - Leader Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'))
    plt.close()
    print(f"Saved confusion matrix to {os.path.join(output_path, 'confusion_matrix.png')}")

    # ROC Curve
    if 'fail' in y_pred.columns:
        fpr, tpr, _ = roc_curve((y_actual == 'fail').astype(int), y_pred_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Leader Model')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(os.path.join(output_path, 'roc_curve.png'))
        plt.close()
        print(f"Saved ROC curve to {os.path.join(output_path, 'roc_curve.png')}")

    # Variable Importance (if supported)
    if hasattr(leader_model, 'varimp_plot'):
        print("\n=== Variable Importance ===")
        # This will display the plot in a new window, which is why we don't save it
        leader_model.varimp_plot()
    else:
        print("\nLeader model does not support variable importance plotting.")

    # ============================================================
    # 6. Compare Top 5 Models
    # ============================================================
    print("\n=== Top 5 Models Comparison ===")
    top_models = h2o_models.as_data_frame().head(5)
    for model_id in top_models['model_id']:
        model = h2o.get_model(model_id)
        preds = model.predict(h2o_frame_test).as_data_frame()
        if 'predict' in preds.columns:
            acc = (preds['predict'] == y_actual).mean()
            print(f"Model: {model_id}, Test Accuracy: {acc:.4f}")
            
