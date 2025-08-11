# Modified Main Script (h2o.py)
import h2o
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing.utils.model_utils.h2o_utils import (
    run_comprehensive_automl_pipeline,
    save_last_run,
    load_last_run,
    initialize_h2o_cluster,
    load_data_with_polars,
    comprehensive_schema_harmonization
)

def main():
    """
    Main function to orchestrate the H2O AutoML pipeline with robust data preprocessing.
    Ensures consistent data types, proper encoding, and comprehensive error handling.
    Preserves original measure_value and protects station_desc.
    """
    try:
        # Initialize H2O cluster with optimal settings for 14GB system
        print("="*70)
        print("INITIALIZING H2O CLUSTER FOR COMPREHENSIVE AUTOML")
        print("="*70)
        initialize_h2o_cluster(max_mem_gb=14, max_retries=5)  # Increased to 14GB

        # --- Configuration Management ---
        print("\n" + "="*50)
        print("CONFIGURATION SETUP")
        print("="*50)

        reuse_last = input("Reuse last run configuration? (y/n): ").strip().lower() == "y"
        last_run = load_last_run() if reuse_last else {}
        
        data_choice = None
        original_raw_path = None
        train_path = None
        test_path = None
        output_path = None
        use_validation_frame = None
        
        def get_and_validate_path(prompt_text, file_type=None):
            path = input(f"Enter path to {prompt_text}: ").strip().strip('"')
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            if file_type and not path.lower().endswith(f".{file_type}"):
                raise ValueError(f"File must be a {file_type} file: {path}")
            return path

        if reuse_last and last_run.get('data_choice') and last_run.get('output_path'):
            data_choice = last_run['data_choice']
            output_path = last_run['output_path']
            use_validation_frame = last_run.get('use_validation_frame', False)
            
            if data_choice == "1" and last_run.get('original_raw_path') and os.path.exists(last_run['original_raw_path']):
                original_raw_path = last_run['original_raw_path']
                print(f"Using previous train_path: {original_raw_path}")
                print(f"Using previous output_path: {output_path}")
            elif data_choice == "2" and last_run.get('train_path') and last_run.get('test_path') and os.path.exists(last_run['train_path']) and os.path.exists(last_run['test_path']):
                train_path = last_run['train_path']
                test_path = last_run['test_path']
                print(f"Using previous train_path: {train_path}")
                print(f"Using previous test_path: {test_path}")
                print(f"Using previous output_path: {output_path}")
            else:
                print("❌ Previous configuration is incomplete or paths are invalid. Please set up a new run.")
                reuse_last = False
        
        if not reuse_last:
            print("\nChoose data input method:")
            print("1. Single raw data file (will be split into train/test)")
            print("2. Separate pre-split train and test files")
            data_choice = input("Enter choice (1 or 2): ").strip()

            if data_choice == "1":
                original_raw_path = get_and_validate_path("raw parquet dataset", file_type="parquet")
            elif data_choice == "2":
                train_path = get_and_validate_path("train parquet dataset", file_type="parquet")
                test_path = get_and_validate_path("test parquet dataset", file_type="parquet")
            else:
                raise ValueError("Invalid data input choice. Please enter '1' or '2'.")
            
            output_path = input("Enter path to output directory: ").strip().strip('"')
            use_validation_frame = input("Use validation frame instead of cross-validation? (y/n): ").strip().lower() == "y"
            
        output_path = os.path.join(output_path, "h2o_results")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created output directory: {output_path}")
            
        target_column = None

        # --- Enhanced Data Loading and Preprocessing ---
        print("\n" + "="*50)
        print("ENHANCED DATA LOADING & PREPROCESSING")
        print("="*50)
        print("Features: Comprehensive type checking, consistent encoding, and schema harmonization")

        def detect_target_column(df):
            print("\n🔍 DETECTING TARGET COLUMN")
            print("="*30)
            
            common_target_names = ['target', 'book_state']
            available_targets = [col for col in common_target_names if col in df.columns]
            
            print(f"📊 Dataset shape: {df.shape}")
            print(f"📝 Available columns ({len(df.columns)}): {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
            
            if available_targets:
                print(f"🎯 Found potential target columns: {available_targets}")
                if len(available_targets) == 1:
                    target_col = available_targets[0]
                    print(f"   ✅ Auto-selected target column: '{target_col}'")
                else:
                    print("   Multiple potential target columns found. Please choose:")
                    for i, col in enumerate(available_targets, 1):
                        unique_vals = df[col].nunique()
                        print(f"   {i}. '{col}' (unique values: {unique_vals})")
                    
                    choice = input("   Enter choice (1-{}): ".format(len(available_targets))).strip()
                    try:
                        choice_idx = int(choice) - 1
                        target_col = available_targets[choice_idx]
                        print(f"   ✅ Selected target column: '{target_col}'")
                    except (ValueError, IndexError):
                        target_col = available_targets[0]
                        print(f"   ⚠️ Invalid choice, defaulting to: '{target_col}'")
            else:
                print("❌ No common target column names found.")
                print("📋 All available columns:")
                for i, col in enumerate(df.columns, 1):
                    unique_vals = df[col].nunique()
                    data_type = str(df[col].dtype)
                    print(f"   {i:2d}. '{col}' ({data_type}, unique: {unique_vals})")
                
                while True:
                    choice = input(f"\nEnter target column name or number (1-{len(df.columns)}): ").strip()
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(df.columns):
                            target_col = df.columns[choice_idx]
                            break
                    except ValueError:
                        pass
                    
                    if choice in df.columns:
                        target_col = choice
                        break
                    
                    print(f"   ❌ Invalid choice '{choice}'. Please try again.")
                
                print(f"   ✅ Selected target column: '{target_col}'")
            
            unique_count = df[target_col].nunique()
            missing_count = df[target_col].isna().sum()
            
            print(f"\n📈 Target column analysis:")
            print(f"   • Column: '{target_col}'")
            print(f"   • Data type: {df[target_col].dtype}")
            print(f"   • Unique values: {unique_count}")
            print(f"   • Missing values: {missing_count} ({missing_count/len(df)*100:.2f}%)")
            
            if unique_count > 20:
                print(f"   ⚠️ High cardinality target ({unique_count} unique values)")
                print(f"   This might be a regression problem or need preprocessing.")
                
                sample_values = df[target_col].dropna().head(10).tolist()
                print(f"   📋 Sample values: {sample_values}")
                
                continue_choice = input("   Continue with this target? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    print("   ❌ Target selection aborted by user.")
                    raise ValueError("Target column selection aborted.")
            
            if unique_count <= 20:
                print(f"   📊 Value distribution:")
                value_counts = df[target_col].value_counts().head(10)
                for val, count in value_counts.items():
                    percentage = count / len(df) * 100
                    print(f"      • {val}: {count} ({percentage:.2f}%)")
                if len(value_counts) < df[target_col].nunique():
                    print(f"      • ... and {df[target_col].nunique() - len(value_counts)} more values")
            
            return target_col

        def validate_and_encode_target(df, target_column):
            if df[target_column].isna().any():
                print(f"   ⚠️ Found {df[target_column].isna().sum()} missing values in target column, filling with -1")
                df[target_column] = df[target_column].fillna(-1)
            
            unique_values = sorted(df[target_column].dropna().unique())
            print(f"   📊 Original unique values: {unique_values}")
            
            if len(unique_values) <= 10:
                print(f"   🎯 Detected classification problem with {len(unique_values)} classes")
                
                if set(unique_values).issubset({0, 1, 2}):
                    print("   ✅ Target already in correct format (0, 1, 2)")
                    df[target_column] = df[target_column].astype(int)
                else:
                    print("   🔄 Mapping target values to integers...")
                    value_mapping = {val: idx for idx, val in enumerate(unique_values)}
                    print(f"   📋 Value mapping: {value_mapping}")
                    
                    df[target_column] = df[target_column].map(value_mapping)
                    df[target_column] = df[target_column].fillna(-1).astype(int)
            else:
                print(f"   📈 Detected potential regression problem with {len(unique_values)} unique values")
                print("   🤔 Options for handling continuous target:")
                print("   1. Keep as regression problem (convert to float)")
                print("   2. Bin into categories (convert to classification)")
                
                choice = input("   Enter choice (1 or 2): ").strip()
                
                if choice == "2":
                    n_bins = int(input("   Enter number of bins (2-10): ").strip() or "3")
                    n_bins = max(2, min(10, n_bins))
                    
                    df[target_column], bin_edges = pd.cut(df[target_column], bins=n_bins, labels=False, retbins=True)
                    df[target_column] = df[target_column].fillna(-1).astype(int)
                    
                    print(f"   ✅ Binned target into {n_bins} categories")
                    print(f"   📊 Bin edges: {[f'{edge:.3f}' for edge in bin_edges]}")
                else:
                    df[target_column] = pd.to_numeric(df[target_column], errors='coerce').fillna(0.0)
                    print("   ✅ Converted to continuous regression target")
            
            if df[target_column].dtype in ['int64', 'int32', 'int8', 'int16']:
                class_counts = df[target_column].value_counts().sort_index()
                print(f"   ✅ Final target encoding:")
                for class_val, count in class_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"      • Class {class_val}: {count} samples ({percentage:.2f}%)")
            
            return df

        if data_choice == "1":
            print("📂 Loading raw dataset for train/test split...")
            raw_df = load_data_with_polars(original_raw_path)
            
            target_column = detect_target_column(raw_df)
            raw_df = validate_and_encode_target(raw_df, target_column)

            print("\n🔀 Performing stratified test split...")
            valid_targets = raw_df[target_column].isin([0, 1, 2])
            if not valid_targets.all():
                invalid_count = (~valid_targets).sum()
                print(f"   ⚠️ Removing {invalid_count} rows with invalid target values for splitting")
                raw_df = raw_df[valid_targets]
            
            test_size = 0.1
            test_indices = raw_df.groupby(target_column, group_keys=False).apply(
                lambda x: x.sample(frac=test_size, random_state=42)
            ).index
            test_df = raw_df.loc[test_indices]
            train_df = raw_df.drop(test_indices)
            
            print(f"   ✅ Split completed: Train={len(train_df)} samples, Test={len(test_df)} samples")

            print("\n🔧 Applying comprehensive schema harmonization...")
            train_df, test_df = comprehensive_schema_harmonization(train_df, test_df, target_column)
            
            temp_train_path = os.path.join(output_path, "train_split.parquet")
            temp_test_path = os.path.join(output_path, "test_split.parquet")
            train_df.to_parquet(temp_train_path, index=False)
            test_df.to_parquet(temp_test_path, index=False)
            print(f"💾 Saved train dataset to: {temp_train_path}")
            print(f"💾 Saved test dataset to: {temp_test_path}")

        else:  # data_choice == "2"
            print("📂 Loading pre-split train and test datasets...")
            train_df = load_data_with_polars(train_path)
            test_df = load_data_with_polars(test_path)
            
            target_column = detect_target_column(train_df)
            
            train_df = validate_and_encode_target(train_df, target_column)
            test_df = validate_and_encode_target(test_df, target_column)
            
            print("\n🔧 Applying comprehensive schema harmonization...")
            train_df, test_df = comprehensive_schema_harmonization(train_df, test_df, target_column)
            
            train_df.to_parquet(train_path, index=False)
            test_df.to_parquet(test_path, index=False)
            print(f"💾 Overwrote original train file with preprocessed data: {train_path}")
            print(f"💾 Overwrote original test file with preprocessed data: {test_path}")

        save_last_run(
            data_choice=data_choice,
            original_raw_path=original_raw_path,
            train_path=train_path,
            test_path=test_path,
            output_path=output_path,
            target_column=target_column,
            use_validation_frame=use_validation_frame
        )

        print("\n" + "="*50)
        print("FINAL DATASET STATISTICS")
        print("="*50)
        
        stats_train_path = os.path.join(output_path, "train_split.parquet") if data_choice == "1" else train_path
        stats_test_path = os.path.join(output_path, "test_split.parquet") if data_choice == "1" else test_path

        final_train = pd.read_parquet(stats_train_path)
        final_test = pd.read_parquet(stats_test_path)
        
        print(f"📊 Training Dataset:")
        print(f"   • Shape: {final_train.shape}")
        print(f"   • Memory Usage: {final_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        train_class_dist = final_train[target_column].value_counts().sort_index()
        print(f"   • Class Distribution:")
        for class_val, count in train_class_dist.items():
            percentage = (count / len(final_train)) * 100
            print(f"      - Class {class_val}: {count} samples ({percentage:.2f}%)")
        
        print(f"\n📊 Test Dataset:")
        print(f"   • Shape: {final_test.shape}")
        print(f"   • Memory Usage: {final_test.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        test_class_dist = final_test[target_column].value_counts().sort_index()
        print(f"   • Class Distribution:")
        for class_val, count in test_class_dist.items():
            percentage = (count / len(final_test)) * 100
            print(f"      - Class {class_val}: {count} samples ({percentage:.2f}%)")

        print(f"\n🔍 Schema Consistency Check:")
        schema_mismatches = []
        for col in final_train.columns:
            if col in final_test.columns:
                if final_train[col].dtype != final_test[col].dtype:
                    schema_mismatches.append({
                        'column': col,
                        'train_dtype': str(final_train[col].dtype),
                        'test_dtype': str(final_test[col].dtype)
                    })
        
        if schema_mismatches:
            print(f"   ❌ Found {len(schema_mismatches)} schema mismatches:")
            for mismatch in schema_mismatches:
                print(f"      • {mismatch['column']}: train={mismatch['train_dtype']} vs test={mismatch['test_dtype']}")
            raise ValueError("Schema mismatches detected between train and test datasets!")
        else:
            print(f"   ✅ All {len(final_train.columns)} columns have consistent data types")
        
        print("✅ Configuration saved successfully")

        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE H2O AUTOML PIPELINE")
        print("="*70)
        print("Pipeline Features:")
        print("✓ Robust data type consistency ensured")
        print("✓ Comprehensive preprocessing completed")
        print("✓ Advanced algorithms with extensive tuning")
        print("✓ Stacked ensembles and meta-learning")
        print("✓ Cross-validation with balanced classes")
        print("✓ Comprehensive model evaluation and comparison")
        print("✓ Optimized for 14GB memory system")
        print("="*70)

        results = run_comprehensive_automl_pipeline(
            train_path=stats_train_path,
            test_path=stats_test_path,
            output_path=output_path,
            target_column=target_column,
            use_validation_frame=use_validation_frame
        )

        if results:
            best_model, leaderboard, summary_stats = results
            print("\n" + "="*70)
            print("🎉 COMPREHENSIVE AUTOML PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"🏆 Champion Model: {best_model.model_id}")
            print(f"🎯 Algorithm Type: {summary_stats['best_algorithm']}")
            print(f"📊 Test Accuracy: {summary_stats['final_metrics']['accuracy']:.4f}")
            auc_score = summary_stats['final_metrics'].get('auc', 'N/A')
            if auc_score != 'N/A' and auc_score is not None:
                print(f"🎪 ROC AUC: {auc_score:.4f}")
            else:
                print(f"🎪 ROC AUC: {auc_score}")
            print(f"🔢 Total Models Trained: {summary_stats['total_models_trained']}")
            print(f"📁 Results Location: {output_path}")
            print("="*70)
            print("\n📋 Generated Reports & Visualizations:")
            key_files = [
                "comprehensive_leaderboard.csv",
                "model_comparison_metrics.json",
                "confusion_matrix_champion.png",
                "model_logloss_distribution.png",
                "automl_executive_summary.json"
            ]
            for file in key_files:
                file_path = os.path.join(output_path, file)
                if os.path.exists(file_path):
                    print(f"   ✅ {file}")
                else:
                    print(f"   ⚠️ {file} (not generated)")
        else:
            print("❌ Pipeline execution failed. Check error logs above.")

    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user.")
    except Exception as e:
        print(f"\n❌ Critical error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            conn = h2o.connection()
            if conn and conn.connected:
                print("\n🔄 Shutting down H2O cluster...")
                h2o.shutdown(prompt=False)
                print("✅ H2O cluster shutdown complete.")
        except:
            print("⚠️ H2O cluster cleanup completed.")

if __name__ == "__main__":
    main()