# Modified Utils Script (h2o_utils.py)
import json
import os
import socket
import time
import warnings
from datetime import datetime
from typing import Tuple

import h2o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from h2o.automl import H2OAutoML
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

CONFIG_FILE = "last_run_config.json"


def save_last_run(data_choice=None, original_raw_path=None, train_path=None, test_path=None, output_path=None,
                  target_column=None, use_validation_frame=None):
    config = {
        "data_choice": data_choice,
        "original_raw_path": original_raw_path,
        "train_path": train_path,
        "test_path": test_path,
        "output_path": output_path,
        "target_column": target_column,
        "use_validation_frame": use_validation_frame,
        "timestamp": datetime.now().isoformat()
    }
    try:
        config_path = CONFIG_FILE
        if output_path and os.path.exists(output_path):
            config_path = os.path.join(output_path, CONFIG_FILE)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print("Configuration saved successfully")
    except Exception as e:
        print(f"Could not save configuration: {e}")


def load_last_run():
    config_paths = [CONFIG_FILE, os.path.join(".", CONFIG_FILE)]

    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if 'timestamp' in config:
                    print(f"📅 Last run: {config['timestamp']}")
                return config
            except Exception as e:
                print(f"Could not load configuration from {config_path}: {e}")
    return {}


def find_free_port(start_port=54321, max_attempts=10):
    port = start_port
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"No free ports found in range {start_port} to {start_port + max_attempts - 1}")


def initialize_h2o_cluster(max_mem_gb=14, max_retries=5):
    print(f"Initializing H2O cluster with {max_mem_gb}GB memory...")

    for attempt in range(max_retries):
        try:
            try:
                h2o.cluster().shutdown()
                time.sleep(2)
            except:
                pass

            port = find_free_port()
            print(f"   🔌 Attempting to use port: {port}")

            ice_root_path = "./h2o_cache"
            if not os.path.exists(ice_root_path):
                os.makedirs(ice_root_path)

            h2o.init(
                max_mem_size_GB=max_mem_gb,
                nthreads=-1,
                min_mem_size_GB=4,
                ice_root=ice_root_path,
                strict_version_check=False,
                start_h2o=True,
                port=port,
                name=f"AutoML_Cluster_{port}"
            )

            cluster_info = h2o.cluster()
            print(f"H2O Cluster initialized:")
            print(f"   • Nodes: {cluster_info.cloud_size}")
            print(f"   • Memory: {cluster_info.cloud_healthy}")
            print(f"   • Version: {cluster_info.version}")
            print(f"   • Port: {port}")
            return

        except Exception as e:
            print(f"❌ Attempt {attempt + 1}/{max_retries} failed to initialize H2O cluster: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                raise RuntimeError("Failed to initialize H2O cluster after maximum retries")


def load_data_with_polars(file_path):
    print(f"Loading data from {file_path} using Polars...")
    try:
        pl_df = pl.read_parquet(file_path)
        df = pl_df.to_pandas()
        print("Data loaded successfully with Polars!")
        return df
    except Exception as e:
        print(f"Error loading data with Polars: {e}")
        print("Falling back to Pandas read_parquet...")
        return pd.read_parquet(file_path)


def remove_problematic_columns(train_df, test_df, target_column):
    print("🧹 Removing problematic columns...")

    columns_to_remove = []
    problematic_patterns = [
        'lot_id', 'lot_number', 'workorder_type', 'station_diag_number',
        'station_diag_desc', 'workstep_number_alt', 'plant_desc'
    ]

    protected_columns = ['station_desc', 'measure_value']

    if 'station_desc' in train_df.columns:
        unique_count = train_df['station_desc'].nunique()
        missing_ratio = train_df['station_desc'].isnull().sum() / len(train_df)
        print(f"Validating station_desc:")
        print(f"      • Unique values: {unique_count}")
        print(f"      • Missing ratio: {missing_ratio:.1%}")
        if unique_count <= 1:
            print(f"station_desc is constant, will be retained as categorical")
        elif missing_ratio > 0.5:
            print(f"station_desc has high missing rate, filling with 'UNKNOWN'")
            train_df['station_desc'] = train_df['station_desc'].fillna('UNKNOWN')
            test_df['station_desc'] = test_df['station_desc'].fillna('UNKNOWN')

    for col in train_df.columns:
        if col == target_column or col in protected_columns:
            continue

        if col in problematic_patterns:
            columns_to_remove.append(col)
            print(f"   • Removing problematic column (H2O reported): {col}")
            continue

        missing_ratio = train_df[col].isnull().sum() / len(train_df)
        if missing_ratio > 0.98:
            columns_to_remove.append(col)
            print(f"   • Removing mostly missing column: {col} ({missing_ratio:.1%} missing)")
            continue

        if train_df[col].nunique() <= 1:
            columns_to_remove.append(col)
            print(f"   • Removing constant column: {col}")

    if columns_to_remove:
        train_df = train_df.drop(columns=columns_to_remove)
        test_df = test_df.drop(columns=[col for col in columns_to_remove if col in test_df.columns])
        print(f"Removed {len(columns_to_remove)} problematic columns")
    else:
        print("No additional problematic columns found")

    return train_df, test_df


def comprehensive_schema_harmonization(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    print("Starting comprehensive schema harmonization...")

    train_df, test_df = remove_problematic_columns(train_df, test_df, target_column)

    train_harmonized = train_df.copy()
    test_harmonized = test_df.copy()

    common_columns = list(set(train_harmonized.columns) & set(test_harmonized.columns))
    if len(common_columns) != len(train_harmonized.columns) or len(common_columns) != len(test_harmonized.columns):
        train_only = set(train_harmonized.columns) - set(common_columns)
        test_only = set(test_harmonized.columns) - set(common_columns)
        print(f"Column alignment:")
        print(f"      • Common columns: {len(common_columns)}")
        if train_only:
            print(f"      • Train-only columns (will be removed): {sorted(train_only)}")
        if test_only:
            print(f"      • Test-only columns (will be removed): {sorted(test_only)}")

        train_harmonized = train_harmonized[common_columns]
        test_harmonized = test_harmonized[common_columns]

    print("Processing categorical columns for H2O compatibility...")

    categorical_cols = []
    for col in common_columns:
        if col == target_column:
            continue

        if col == 'measure_value':
            try:
                print(f"Analyzing measure_value...")
                train_harmonized[col] = pd.to_numeric(train_harmonized[col], errors='coerce')
                test_harmonized[col] = pd.to_numeric(test_harmonized[col], errors='coerce')

                combined_values = pd.concat([train_harmonized[col], test_harmonized[col]], ignore_index=True)
                stats = {
                    'min': combined_values.min(),
                    'max': combined_values.max(),
                    'mean': combined_values.mean(),
                    'median': combined_values.median(),
                    'std': combined_values.std(),
                    'nan_count': combined_values.isna().sum(),
                    'inf_count': np.isinf(combined_values).sum() if np.isinf(combined_values).any() else 0
                }
                print(f"         • Statistics: {stats}")

                fill_value = stats['median']
                if pd.isna(fill_value):
                    fill_value = 0.0

                train_harmonized[col] = train_harmonized[col].fillna(fill_value)
                test_harmonized[col] = test_harmonized[col].fillna(fill_value)
                train_harmonized[col] = train_harmonized[col].replace([np.inf, -np.inf], fill_value)
                test_harmonized[col] = test_harmonized[col].replace([np.inf, -np.inf], fill_value)

                print(f"      • Processed 'measure_value' as numeric (original values preserved)")
                continue
            except Exception as e:
                print(f"    Failed to process 'measure_value' as numeric: {e}. Treating as categorical.")

        if train_harmonized[col].dtype == 'object' or test_harmonized[col].dtype == 'object':
            train_harmonized[col] = train_harmonized[col].astype(str).fillna('MISSING')
            test_harmonized[col] = test_harmonized[col].astype(str).fillna('MISSING')

            train_harmonized[col] = train_harmonized[col].replace('nan', 'MISSING')
            test_harmonized[col] = test_harmonized[col].replace('nan', 'MISSING')

            categorical_cols.append(col)
            print(f"• Processed categorical column: {col} ({train_harmonized[col].nunique()} categories)")

    print("   Processing numeric columns...")
    numeric_cols = []
    for col in common_columns:
        if col == target_column or col in categorical_cols:
            continue

        try:
            train_harmonized[col] = pd.to_numeric(train_harmonized[col], errors='coerce')
            test_harmonized[col] = pd.to_numeric(test_harmonized[col], errors='coerce')

            combined_values = pd.concat([train_harmonized[col], test_harmonized[col]], ignore_index=True)
            fill_value = combined_values.median()
            if pd.isna(fill_value):
                fill_value = 0.0

            train_harmonized[col] = train_harmonized[col].fillna(fill_value)
            test_harmonized[col] = test_harmonized[col].fillna(fill_value)

            if np.isinf(train_harmonized[col]).any() or np.isinf(test_harmonized[col]).any():
                print(f"Found infinite values in {col}, replacing with median")
                train_harmonized[col] = train_harmonized[col].replace([np.inf, -np.inf], fill_value)
                test_harmonized[col] = test_harmonized[col].replace([np.inf, -np.inf], fill_value)

            numeric_cols.append(col)

        except Exception as e:
            print(f"Could not process numeric column {col}: {e}")
            train_harmonized[col] = train_harmonized[col].astype(str).fillna('MISSING')
            test_harmonized[col] = test_harmonized[col].astype(str).fillna('MISSING')
            categorical_cols.append(col)

    print(f"• Processed {len(numeric_cols)} numeric columns")
    print(f"• Processed {len(categorical_cols)} categorical columns")

    column_order = [target_column] + sorted([col for col in common_columns if col != target_column])

    train_harmonized = train_harmonized[column_order]
    test_harmonized = test_harmonized[column_order]

    print("   🗜️ Optimizing memory usage...")
    initial_memory = train_harmonized.memory_usage(deep=True).sum() + test_harmonized.memory_usage(deep=True).sum()

    for col in train_harmonized.select_dtypes(include=[np.number]).columns:
        if col == target_column or col == 'measure_value':
            continue

        col_min = min(train_harmonized[col].min(), test_harmonized[col].min())
        col_max = max(train_harmonized[col].max(), test_harmonized[col].max())

        if train_harmonized[col].dtype == 'float64':
            if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                train_harmonized[col] = train_harmonized[col].astype(np.float32)
                test_harmonized[col] = test_harmonized[col].astype(np.float32)
        elif train_harmonized[col].dtype == 'int64':
            if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                train_harmonized[col] = train_harmonized[col].astype(np.int32)
                test_harmonized[col] = test_harmonized[col].astype(np.int32)

    final_memory = train_harmonized.memory_usage(deep=True).sum() + test_harmonized.memory_usage(deep=True).sum()
    memory_saved = initial_memory - final_memory

    print(f"Schema harmonization completed successfully!")
    print(f"      • Total columns: {len(column_order)}")
    print(f"      • Train shape: {train_harmonized.shape}")
    print(f"      • Test shape: {test_harmonized.shape}")
    if memory_saved > 0:
        print(f"      • Memory saved: {memory_saved / 1024 ** 2:.1f} MB")

    return train_harmonized, test_harmonized


def chunked_h2o_frame_conversion(df, chunk_size=500000):
    print(f"Converting DataFrame to H2OFrame in chunks of {chunk_size} rows...")
    h2o_frame = None
    column_types = {'measure_value': 'real', 'station_desc': 'enum'}
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end]
        try:
            chunk_frame = h2o.H2OFrame(chunk, column_types=column_types)
            if h2o_frame is None:
                h2o_frame = chunk_frame
            else:
                h2o_frame = h2o_frame.rbind(chunk_frame)
            print(f"      • Processed chunk {start} to {end}")
        except Exception as e:
            print(f"Chunk {start} to {end} failed: {e}")
            raise
    return h2o_frame


def maximize_automl_configuration(training_frame, validation_frame, x, y, use_validation_frame=True):
    print(f"\nCONFIGURING COMPREHENSIVE AUTOML")
    print("=" * 50)

    max_runtime_secs = 1800  # Reduced to 30 minutes for efficiency
    max_models = 10  # Reduced for efficiency
    nfolds = 0 if use_validation_frame else 5

    print(f"Total Runtime Budget: {max_runtime_secs} seconds ({max_runtime_secs / 60:.0f} minutes)")
    print(f"AutoML Configuration:")
    print(f"   • Max Models: {max_models}")
    print(f"   • Validation: {'Validation frame' if use_validation_frame else f'{nfolds}-fold cross-validation'}")
    print(f"   • Included Algorithms: GBM, DRF, DeepLearning, StackedEnsemble")
    print(f"   • Balance Classes: Enabled with class weights")
    print(f"   • Early Stopping: Enabled")

    class_counts = training_frame[y].table().as_data_frame()
    total_samples = class_counts['Count'].sum()
    class_weights = {}
    for idx, row in class_counts.iterrows():
        class_label = str(row[0])
        weight = total_samples / (len(class_counts) * row['Count'])
        class_weights[class_label] = min(weight, 100.0)

    print(f"   • Class Weights: {class_weights}")

    automl = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=max_models,
        seed=42,
        sort_metric='logloss',
        stopping_metric='logloss',
        stopping_tolerance=0.001,
        stopping_rounds=3,
        nfolds=nfolds,
        keep_cross_validation_predictions=True,
        keep_cross_validation_models=False,
        keep_cross_validation_fold_assignment=False,
        include_algos=['GBM', 'DRF', 'DeepLearning', 'StackedEnsemble'],
        balance_classes=True,
        class_sampling_factors=[class_weights.get(str(i), 1.0) for i in range(3)],
        max_after_balance_size=3.0,
        project_name=f"MultiClass_AutoML_{int(time.time())}",
        verbosity='info',
        export_checkpoints_dir=os.path.join(os.path.dirname(training_frame.frame_id), "checkpoints")
    )

    return automl


def train_comprehensive_automl(automl, x, y, training_frame, validation_frame, leaderboard_frame=None):
    print(f"\nTRAINING COMPREHENSIVE AUTOML MODELS")
    print("=" * 60)
    start_time = time.time()
    print("Starting AutoML training...")
    print("    This may take some time depending on your runtime budget...")

    try:
        automl.train(
            x=x,
            y=y,
            training_frame=training_frame,
            validation_frame=validation_frame if validation_frame is not None else None,
            leaderboard_frame=leaderboard_frame
        )
        training_time = time.time() - start_time
        print(f"AutoML training completed in {training_time / 60:.2f} minutes")

        if automl.leader is None:
            print("❌ AutoML failed to produce a leader model. The leaderboard is empty.")
            return None, None, training_time

        leaderboard = h2o.automl.get_leaderboard(automl, extra_columns='ALL')
        print(f"Total models trained: {leaderboard.nrows}")
        return automl, leaderboard, training_time

    except Exception as e:
        training_time = time.time() - start_time
        print(f"AutoML training failed after {training_time / 60:.2f} minutes: {e}")
        import traceback
        traceback.print_exc()
        return None, None, training_time


def analyze_model_performance(automl, leaderboard, test_frame, y, output_path):
    print(f"\nCOMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)

    if leaderboard is None or leaderboard.nrows == 0:
        print("No leaderboard available for analysis")
        return {'champion_model': None, 'leaderboard_df': pd.DataFrame()}

    try:
        lb_df = leaderboard.as_data_frame()
        lb_df['algorithm'] = lb_df['model_id'].apply(lambda x: x.split('_')[0])
        algorithm_counts = lb_df['algorithm'].value_counts()
        print(f"Algorithm Distribution:")
        for algo, count in algorithm_counts.items():
            percentage = (count / len(lb_df)) * 100
            print(f"   • {algo}: {count} models ({percentage:.1f}%)")

        print(f"\nPerformance Metrics Summary:")
        if 'logloss' in lb_df.columns:
            valid_logloss = lb_df['logloss'].dropna()
            if not valid_logloss.empty:
                print(f"   • Best LogLoss: {valid_logloss.min():.6f}")
                print(f"   • Worst LogLoss: {valid_logloss.max():.6f}")
                print(f"   • Mean LogLoss: {valid_logloss.mean():.6f}")
                print(f"   • Std LogLoss: {valid_logloss.std():.6f}")

        if 'auc' in lb_df.columns:
            valid_aucs = lb_df['auc'].dropna()
            if not valid_aucs.empty:
                print(f"   • Best AUC: {valid_aucs.max():.6f}")
                print(f"   • Mean AUC: {valid_aucs.mean():.6f}")

        if 'mean_per_class_error' in lb_df.columns:
            valid_mpce = lb_df['mean_per_class_error'].dropna()
            if not valid_mpce.empty:
                print(f"   • Best Mean Per Class Error: {valid_mpce.min():.6f}")

        top_5 = lb_df.head(5)
        print(f"\nTOP 5 MODELS:")
        for i, row in top_5.iterrows():
            logloss_str = f"{row.get('logloss', 'N/A'):.6f}" if pd.notnull(row.get('logloss')) else "N/A"
            auc_str = f"{row.get('auc', 'N/A'):.6f}" if pd.notnull(row.get('auc')) else "N/A"
            print(f"   {i + 1}. {row['model_id']:<40} | LogLoss: {logloss_str} | AUC: {auc_str}")

        leader = automl.leader
        print(f"\nCHAMPION MODEL ANALYSIS:")
        print(f"   • Model ID: {leader.model_id}")
        print(f"   • Algorithm: {leader.__class__.__name__}")

        try:
            print("   🔮 Making test predictions...")
            test_predictions = leader.predict(test_frame)
            test_actual = test_frame[y].as_data_frame()[y]
            test_pred_df = test_predictions.as_data_frame()

            test_accuracy = accuracy_score(test_actual, test_pred_df['predict'])

            test_auc = None
            class_report = {}
            try:
                prob_cols = [col for col in test_pred_df.columns if col.startswith('p')]
                if len(prob_cols) >= 3:
                    classes = sorted(test_actual.unique())
                    if len(classes) == len(prob_cols):
                        y_actual_encoded = pd.Categorical(test_actual, categories=classes).codes
                        y_pred_probs = test_pred_df[prob_cols].values
                        test_auc = roc_auc_score(y_actual_encoded, y_pred_probs,
                                                 multi_class='ovr', average='weighted')

                class_report = classification_report(test_actual, test_pred_df['predict'], output_dict=True)
                print(f"Class-wise Performance:")
                for class_label in sorted(test_actual.unique()):
                    if str(class_label) in class_report:
                        metrics = class_report[str(class_label)]
                        print(
                            f"      • Class {class_label}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
            except Exception as auc_error:
                print(f"AUC or classification report failed: {auc_error}")

            print(f"• Test Accuracy: {test_accuracy:.6f}")
            if test_auc:
                print(f"• Test AUC (weighted): {test_auc:.6f}")

        except Exception as pred_error:
            print(f"Prediction analysis failed: {pred_error}")
            import traceback
            traceback.print_exc()
            test_accuracy = 0.0
            test_auc = None
            class_report = {}
            test_pred_df = pd.DataFrame()
            test_actual = pd.Series()

        try:
            lb_df.to_csv(os.path.join(output_path, "comprehensive_leaderboard.csv"), index=False)
            with open(os.path.join(output_path, "model_comparison_metrics.json"), 'w') as f:
                json.dump({
                    'test_accuracy': float(test_accuracy),
                    'test_auc': float(test_auc) if test_auc is not None else None,
                    'classification_report': class_report,
                    'model_count': len(lb_df),
                    'best_logloss': float(
                        valid_logloss.min()) if 'logloss' in lb_df.columns and not valid_logloss.empty else None
                }, f, indent=4)
            print("Results saved successfully")
        except Exception as save_error:
            print(f"Could not save results: {save_error}")

        return {
            'champion_model': leader,
            'leaderboard_df': lb_df,
            'algorithm_distribution': algorithm_counts.to_dict(),
            'performance_summary': {
                'best_logloss': float(
                    valid_logloss.min()) if 'logloss' in lb_df.columns and not valid_logloss.empty else float('nan'),
                'mean_logloss': float(
                    valid_logloss.mean()) if 'logloss' in lb_df.columns and not valid_logloss.empty else float('nan'),
                'best_auc': float(valid_aucs.max()) if 'auc' in lb_df.columns and not valid_aucs.empty else float(
                    'nan'),
                'mean_auc': float(valid_aucs.mean()) if 'auc' in lb_df.columns and not valid_aucs.empty else float(
                    'nan')
            },
            'test_metrics': {
                'accuracy': test_accuracy,
                'auc': test_auc,
                'classification_report': class_report
            },
            'test_predictions': test_pred_df,
            'test_actual': test_actual
        }

    except Exception as e:
        print(f"Model performance analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {'champion_model': None, 'leaderboard_df': pd.DataFrame()}


def create_comprehensive_visualizations(analysis_results, output_path):
    print(f"\nCREATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 50)

    try:
        lb_df = analysis_results['leaderboard_df']
        if lb_df.empty:
            print("No leaderboard data available for visualization")
            return

        if 'logloss' in lb_df.columns:
            valid_logloss = lb_df['logloss'].dropna()
            if not valid_logloss.empty:
                plt.figure(figsize=(10, 6))
                plt.hist(valid_logloss, bins=min(20, len(valid_logloss)),
                         alpha=0.7, color='lightcoral', edgecolor='black')
                plt.title('Distribution of LogLoss Scores Across Models')
                plt.xlabel('LogLoss Score (lower is better)')
                plt.ylabel('Number of Models')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, "model_logloss_distribution.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()
                print("LogLoss distribution plot created")

        if 'test_actual' in analysis_results and 'test_predictions' in analysis_results:
            test_actual = analysis_results['test_actual']
            test_predictions = analysis_results['test_predictions']

            if not test_actual.empty and not test_predictions.empty and 'predict' in test_predictions.columns:
                try:
                    cm = confusion_matrix(test_actual, test_predictions['predict'])
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
                    plt.title('Confusion Matrix - Champion Model')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_path, "confusion_matrix_champion.png"),
                                dpi=150, bbox_inches='tight')
                    plt.close()
                    print("Confusion matrix created")
                except Exception as cm_error:
                    print(f"Could not create confusion matrix: {cm_error}")

        if 'algorithm_distribution' in analysis_results:
            algo_dist = analysis_results['algorithm_distribution']
            if algo_dist:
                plt.figure(figsize=(10, 6))
                algorithms = list(algo_dist.keys())
                counts = list(algo_dist.values())

                plt.bar(algorithms, counts, color='lightgreen', alpha=0.7, edgecolor='black')
                plt.title('Algorithm Distribution in AutoML Training')
                plt.xlabel('Algorithm Type')
                plt.ylabel('Number of Models')
                plt.xticks(rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, "algorithm_distribution.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()
                print("Algorithm distribution plot created")

    except Exception as e:
        print(f"Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()


def generate_executive_summary(analysis_results, data_info, training_time, output_path):
    print("\nGENERATING EXECUTIVE SUMMARY")
    print("=" * 50)

    try:
        champion_model = analysis_results.get('champion_model')
        test_metrics = analysis_results.get('test_metrics', {})
        leaderboard_df = analysis_results.get('leaderboard_df', pd.DataFrame())

        summary = {
            'timestamp': datetime.now().isoformat(),
            'execution_summary': {
                'total_models_trained': len(leaderboard_df),
                'training_time_minutes': round(training_time / 60.0, 2),
                'successful_completion': champion_model is not None
            },
            'best_model_info': {
                'model_id': champion_model.model_id if champion_model else 'None',
                'algorithm_type': champion_model.__class__.__name__ if champion_model else 'None'
            },
            'performance_metrics': {
                'test_accuracy': round(test_metrics.get('accuracy', 0.0), 6),
                'test_auc': round(test_metrics.get('auc'), 6) if test_metrics.get('auc') is not None else None,
                'has_classification_report': bool(test_metrics.get('classification_report'))
            },
            'dataset_information': data_info,
            'technical_details': {
                'h2o_version': h2o.__version__,
                'cross_validation_folds': 5,
                'balance_classes_enabled': True,
                'included_algorithms': ['GBM', 'DRF', 'DeepLearning', 'StackedEnsemble'],
                'excluded_algorithms': ['GLM']
            }
        }

        perf_summary = analysis_results.get('performance_summary', {})
        if perf_summary:
            summary['model_performance_distribution'] = {
                'best_logloss': round(perf_summary.get('best_logloss'), 6) if not pd.isna(
                    perf_summary.get('best_logloss', float('nan'))) else None,
                'mean_logloss': round(perf_summary.get('mean_logloss'), 6) if not pd.isna(
                    perf_summary.get('mean_logloss', float('nan'))) else None,
                'best_auc': round(perf_summary.get('best_auc'), 6) if not pd.isna(
                    perf_summary.get('best_auc', float('nan'))) else None,
                'mean_auc': round(perf_summary.get('mean_auc'), 6) if not pd.isna(
                    perf_summary.get('mean_auc', float('nan'))) else None
            }

        with open(os.path.join(output_path, "automl_executive_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)
        print("Executive summary generated successfully")

        return summary

    except Exception as e:
        print(f"Executive summary generation failed: {e}")
        return None


def run_comprehensive_automl_pipeline(train_path, test_path, output_path, target_column='target',
                                      use_validation_frame=True):
    try:
        print("\n" + "=" * 70)
        print("DATA LOADING & H2O CONVERSION")
        print("=" * 70)

        train_df = load_data_with_polars(train_path)
        test_df = load_data_with_polars(test_path)

        if train_df.empty or test_df.empty:
            raise ValueError("One or both datasets are empty after loading")

        print(f"Data loaded successfully:")
        print(f"   • Train: {train_df.shape}")
        print(f"   • Test: {test_df.shape}")

        print("\nSkipping redundant H2O-specific preprocessing...")

        print("\nConverting to H2O Frames...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                h2o_train = chunked_h2o_frame_conversion(train_df)
                h2o_test = chunked_h2o_frame_conversion(test_df)
                print("H2O Frames created successfully")
                break
            except Exception as h2o_error:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {h2o_error}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print("Max retries reached for H2O Frame conversion")
                    raise

        if target_column in h2o_train.columns:
            h2o_train[target_column] = h2o_train[target_column].asfactor()
            h2o_test[target_column] = h2o_test[target_column].asfactor()
            print(f"Target '{target_column}' converted to categorical factor")

        x = h2o_train.columns.copy()
        x.remove(target_column)
        y = target_column

        print(f"   • Features: {len(x)}")
        print(f"   • Target: {y}")

        automl = maximize_automl_configuration(
            training_frame=h2o_train,
            validation_frame=h2o_test if use_validation_frame else None,
            x=x,
            y=y,
            use_validation_frame=use_validation_frame
        )

        automl_results, leaderboard, training_time = train_comprehensive_automl(
            automl,
            x,
            y,
            training_frame=h2o_train,
            validation_frame=h2o_test if use_validation_frame else None,
            leaderboard_frame=h2o_test
        )

        if not automl_results or not automl_results.leader:
            print("AutoML failed to train any models. Exiting pipeline.")
            return None

        analysis_results = analyze_model_performance(automl_results, leaderboard, h2o_test, y, output_path)

        data_info = {
            'train_samples': h2o_train.nrow,
            'test_samples': h2o_test.nrow,
            'features': len(x),
            'target_column': target_column
        }

        create_comprehensive_visualizations(analysis_results, output_path)
        executive_summary = generate_executive_summary(analysis_results, data_info, training_time, output_path)

        summary_stats = {
            'best_algorithm': analysis_results['champion_model'].__class__.__name__ if analysis_results[
                'champion_model'] else 'None',
            'final_metrics': analysis_results['test_metrics'],
            'total_models_trained': len(analysis_results['leaderboard_df']),
            'training_time_minutes': round(training_time / 60.0, 2)
        }

        return analysis_results['champion_model'], leaderboard, summary_stats

    except Exception as e:
        print(f"\nAutoML pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None
