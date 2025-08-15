"""
Optimized Imbalanced Learning Pipeline for Extreme Class Imbalance
================================================================
Designed for Windows systems with memory constraints
Based on state-of-the-art research and GitHub best practices

Key Features:
- Memory-efficient processing with chunking
- Robust preprocessing for mixed data types  
- Advanced ensemble methods for extreme imbalance
- No H2O dependency - pure scikit-learn
- Comprehensive evaluation and visualization
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from pathlib import Path
import json
import pickle
import gc
import os
from collections import Counter

# Core ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           balanced_accuracy_score, f1_score, precision_recall_fscore_support)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                             VotingClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Imbalanced learning
try:
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    from imblearn.combine import SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("Installing imbalanced-learn...")
    os.system("pip install imbalanced-learn")
    try:
        from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
        from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
        from imblearn.under_sampling import EditedNearestNeighbours
        from imblearn.combine import SMOTEENN
        from imblearn.pipeline import Pipeline as ImbPipeline
        IMBLEARN_AVAILABLE = True
    except ImportError:
        IMBLEARN_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

warnings.filterwarnings('ignore')

class OptimizedDataProcessor:
    """Memory-efficient data processor for mixed-type datasets"""
    
    def __init__(self, chunk_size=50000, verbose=True):
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.encoders = {}
        self.scaler = None
        self.feature_info = {}
        
    def analyze_features(self, df, target_col):
        """Analyze feature types and characteristics"""
        if self.verbose:
            print("🔍 Analyzing feature types...")
            
        analysis = {
            'numeric_features': [],
            'categorical_features': [],
            'constant_features': [],
            'high_cardinality': [],
            'missing_heavy': []
        }
        
        for col in df.columns:
            if col == target_col:
                continue
                
            # Check if constant
            if df[col].nunique() <= 1:
                analysis['constant_features'].append(col)
                continue
                
            # Check missing percentage
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > 0.8:
                analysis['missing_heavy'].append(col)
                continue
                
            # Determine type
            if pd.api.types.is_numeric_dtype(df[col]):
                analysis['numeric_features'].append(col)
            else:
                analysis['categorical_features'].append(col)
                # Check cardinality
                if df[col].nunique() > 100:
                    analysis['high_cardinality'].append(col)
        
        self.feature_info = analysis
        
        if self.verbose:
            print(f"   Numeric: {len(analysis['numeric_features'])}")
            print(f"   Categorical: {len(analysis['categorical_features'])}")
            print(f"   Constant: {len(analysis['constant_features'])}")
            print(f"   High cardinality: {len(analysis['high_cardinality'])}")
            print(f"   Missing heavy: {len(analysis['missing_heavy'])}")
            
        return analysis
    
    def preprocess_data(self, df, target_col, fit=True):
        """Robust preprocessing for mixed data types"""
        if self.verbose:
            print("🔧 Preprocessing data...")
            
        df_processed = df.copy()
        
        # Remove problematic features
        cols_to_drop = (self.feature_info['constant_features'] + 
                       self.feature_info['missing_heavy'] + 
                       self.feature_info['high_cardinality'])
        
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop)
            if self.verbose:
                print(f"   Dropped {len(cols_to_drop)} problematic features")
        
        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col].copy()
        
        # Handle missing values
        numeric_cols = [col for col in X.columns if col in self.feature_info['numeric_features']]
        categorical_cols = [col for col in X.columns if col in self.feature_info['categorical_features']]
        
        # Process numeric features
        if numeric_cols:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            
        # Process categorical features  
        if categorical_cols:
            for col in categorical_cols:
                X[col] = X[col].fillna('MISSING').astype(str)
                
                if fit:
                    # Fit encoder
                    le = LabelEncoder()
                    # Handle unseen categories during transform
                    unique_vals = list(X[col].unique()) + ['UNKNOWN']
                    le.fit(unique_vals)
                    self.encoders[col] = le
                
                # Transform
                if col in self.encoders:
                    # Handle unseen categories
                    X[col] = X[col].apply(lambda x: x if x in self.encoders[col].classes_ else 'UNKNOWN')
                    X[col] = self.encoders[col].transform(X[col])
        
        # Scale features
        if fit:
            self.scaler = RobustScaler()  # More robust to outliers
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        X_final = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        if self.verbose:
            print(f"✅ Preprocessing completed: {X_final.shape}")
            
        return X_final, y


class ExtremeImbalanceHandler:
    """Specialized handler for extreme class imbalance (>1000:1 ratio)"""
    
    def __init__(self, verbose=True, random_state=42):
        self.verbose = verbose
        self.random_state = random_state
        self.strategy = None
        
    def analyze_and_recommend(self, y):
        """Analyze imbalance and recommend strategy"""
        class_counts = pd.Series(y).value_counts().sort_index()
        imbalance_ratio = class_counts.max() / class_counts.min()
        minority_samples = class_counts.min()
        
        if self.verbose:
            print(f"\n⚖️ Class Imbalance Analysis:")
            print(f"   Distribution: {class_counts.to_dict()}")
            print(f"   Ratio: {imbalance_ratio:.1f}:1")
            print(f"   Minority samples: {minority_samples}")
        
        # Strategy based on research for extreme imbalance
        if imbalance_ratio > 10000:
            strategy = "ensemble_only"
            reason = "Extreme imbalance - Use ensemble methods only"
        elif imbalance_ratio > 1000:
            strategy = "conservative_smote"
            reason = "Very high imbalance - Conservative SMOTE"
        elif imbalance_ratio > 100:
            strategy = "smote_enn" 
            reason = "High imbalance - SMOTE + cleaning"
        else:
            strategy = "balanced_ensemble"
            reason = "Moderate imbalance - Balanced ensemble"
            
        self.strategy = strategy
        
        if self.verbose:
            print(f"   Recommended: {strategy}")
            print(f"   Reason: {reason}")
            
        return strategy, imbalance_ratio
    
    def apply_strategy(self, X, y, strategy=None):
        """Apply appropriate rebalancing strategy"""
        if strategy is None:
            strategy, _ = self.analyze_and_recommend(y)
            
        if not IMBLEARN_AVAILABLE:
            if self.verbose:
                print("⚠️ imbalanced-learn not available, using original data")
            return X, y
        
        try:
            if strategy == "ensemble_only":
                # For extreme imbalance, rely on specialized ensembles
                return X, y
                
            elif strategy == "conservative_smote":
                # Very conservative SMOTE - minimal oversampling
                target_ratio = min(0.1, 1000 / len(y))  # Max 10% of majority class
                sampler = SMOTE(
                    sampling_strategy={
                        cls: max(int(target_ratio * Counter(y)[Counter(y).most_common(1)[0][0]]), 
                                Counter(y)[cls]) 
                        for cls in Counter(y).keys()
                        if Counter(y)[cls] < Counter(y).most_common(1)[0][1]
                    },
                    k_neighbors=min(5, Counter(y)[min(Counter(y), key=Counter(y).get)] - 1),
                    random_state=self.random_state
                )
                
            elif strategy == "smote_enn":
                sampler = SMOTEENN(
                    smote=SMOTE(
                        sampling_strategy=0.3,  # Conservative ratio
                        random_state=self.random_state
                    ),
                    random_state=self.random_state
                )
                
            else:  # balanced_ensemble
                # Moderate resampling
                sampler = SMOTE(
                    sampling_strategy=0.5,
                    random_state=self.random_state
                )
            
            if strategy != "ensemble_only":
                if self.verbose:
                    print(f"🔄 Applying {strategy}...")
                    
                X_res, y_res = sampler.fit_resample(X, y)
                
                if self.verbose:
                    new_dist = pd.Series(y_res).value_counts().sort_index()
                    print(f"   Before: {pd.Series(y).value_counts().sort_index().to_dict()}")
                    print(f"   After: {new_dist.to_dict()}")
                    
                return X_res, y_res
            else:
                return X, y
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Resampling failed: {e}")
                print("   Using original data")
            return X, y


class AdvancedEnsembleBuilder:
    """Build specialized ensembles for extreme imbalance"""
    
    def __init__(self, random_state=42, verbose=True):
        self.random_state = random_state
        self.verbose = verbose
        self.models = {}
        self.ensemble = None
        
    def build_ensemble(self, strategy="extreme_imbalance"):
        """Build ensemble based on imbalance strategy"""
        
        if self.verbose:
            print(f"\n🤖 Building {strategy} ensemble...")
            
        base_models = []
        
        if IMBLEARN_AVAILABLE:
            # Balanced Random Forest - designed for imbalance
            base_models.append(
                ('brf', BalancedRandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight='balanced_subsample'
                ))
            )
            
            # Easy Ensemble - uses AdaBoost with balanced sampling
            base_models.append(
                ('ee', EasyEnsembleClassifier(
                    n_estimators=50,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            )
        
        # Extra Trees with balanced class weights
        base_models.append(
            ('et', ExtraTreesClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ))
        )
        
        # Logistic Regression with balanced weights
        base_models.append(
            ('lr', LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state,
                solver='liblinear'  # Better for small datasets
            ))
        )
        
        # Regular Random Forest with balanced weights
        base_models.append(
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ))
        )
        
        # Create voting ensemble
        self.ensemble = VotingClassifier(
            estimators=base_models,
            voting='soft',  # Use predicted probabilities
            n_jobs=1  # Avoid nested parallelization
        )
        
        self.models = dict(base_models)
        
        if self.verbose:
            print(f"✅ Ensemble built with {len(base_models)} models:")
            for name, _ in base_models:
                print(f"   • {name}")
                
        return self.ensemble


class ComprehensiveEvaluator:
    """Comprehensive evaluation for imbalanced classification"""
    
    def __init__(self, output_dir, verbose=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.results = {}
        
    def evaluate_model(self, model, X_test, y_test, class_names=None):
        """Comprehensive model evaluation"""
        
        if self.verbose:
            print("\n📊 Comprehensive Model Evaluation...")
            
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Core metrics
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
        
        self.results = {
            'balanced_accuracy': balanced_acc,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_metrics': {
                'precision': precision.tolist(),
                'recall': recall.tolist(), 
                'f1_score': f1.tolist(),
                'support': support.tolist()
            },
            'predictions': {
                'y_pred': y_pred.tolist(),
                'y_test': y_test.tolist()
            }
        }
        
        if y_pred_proba is not None:
            self.results['probabilities'] = y_pred_proba.tolist()
        
        # Generate visualizations
        self._create_visualizations(y_test, y_pred, y_pred_proba, conf_matrix, class_names)
        
        if self.verbose:
            print(f"✅ Evaluation completed:")
            print(f"   Balanced Accuracy: {balanced_acc:.4f}")
            print(f"   Macro F1-Score: {class_report['macro avg']['f1-score']:.4f}")
            print(f"   Weighted F1-Score: {class_report['weighted avg']['f1-score']:.4f}")
            
        return self.results
    
    def _create_visualizations(self, y_test, y_pred, y_pred_proba, conf_matrix, class_names):
        """Create comprehensive visualizations"""
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in sorted(np.unique(y_test))]
            
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class Performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['precision', 'recall', 'f1_score', 'support']
        metric_data = self.results['per_class_metrics']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = metric_data[metric]
            
            bars = ax.bar(class_names, values, color=plt.cm.Set3(np.linspace(0, 1, len(values))))
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score' if metric != 'support' else 'Count')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{val:.3f}' if metric != 'support' else f'{int(val)}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Class Distribution Comparison
        plt.figure(figsize=(12, 6))
        
        test_dist = pd.Series(y_test).value_counts().sort_index()
        pred_dist = pd.Series(y_pred).value_counts().sort_index()
        
        x = np.arange(len(class_names))
        width = 0.35
        
        plt.bar(x - width/2, test_dist.values, width, label='Actual', alpha=0.8)
        plt.bar(x + width/2, pred_dist.values, width, label='Predicted', alpha=0.8)
        
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Actual vs Predicted Class Distribution')
        plt.xticks(x, class_names)
        plt.legend()
        plt.yscale('log')  # Log scale for better visualization of imbalanced data
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"📈 Visualizations saved to: {self.output_dir}")


def main():
    """Optimized main pipeline for extreme imbalance"""
    
    print("🚀 OPTIMIZED IMBALANCED LEARNING PIPELINE")
    print("="*55)
    print("🎯 Designed for extreme class imbalance and system constraints")
    print("📚 Based on state-of-the-art research and GitHub best practices")
    print("="*55)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Get data path
        while True:
            data_path = input("\n📂 Enter dataset path (.parquet/.csv): ").strip().strip('"')
            if os.path.exists(data_path) and data_path.lower().endswith(('.parquet', '.csv')):
                break
            print("❌ Invalid file path. Please provide existing .parquet or .csv file.")
        
        # Load data efficiently
        print(f"\n📊 Loading data: {data_path}")
        if data_path.lower().endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
            
        print(f"✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Memory optimization
        if df.memory_usage(deep=True).sum() > 2e9:  # > 2GB
            print("💾 Large dataset detected, optimizing memory...")
            # Optimize dtypes
            for col in df.select_dtypes(include=['int64']):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            for col in df.select_dtypes(include=['float64']):
                df[col] = pd.to_numeric(df[col], downcast='float')
            gc.collect()
            
        # Auto-detect target column
        print("\n🎯 Detecting target column...")
        potential_targets = []
        
        for col in df.columns:
            try:
                unique_vals = df[col].nunique()
                if 2 <= unique_vals <= 20:  # Reasonable for classification
                    sample_vals = df[col].dropna().unique()[:5]
                    potential_targets.append((col, unique_vals, sample_vals))
            except:
                continue
                
        if potential_targets:
            print("🔍 Potential target columns:")
            for i, (col, n_classes, sample_vals) in enumerate(potential_targets):
                print(f"   {i+1}. {col}: {n_classes} classes {list(sample_vals)}")
                
            while True:
                try:
                    choice = int(input(f"Select target column (1-{len(potential_targets)}): "))
                    if 1 <= choice <= len(potential_targets):
                        target_column = potential_targets[choice-1][0]
                        break
                except ValueError:
                    pass
                print("❌ Invalid selection")
        else:
            print("❌ No obvious target columns found. Showing all columns:")
            for i, col in enumerate(df.columns):
                print(f"   {i+1:2d}. {col}")
            while True:
                try:
                    choice = int(input(f"Select target column (1-{len(df.columns)}): "))
                    if 1 <= choice <= len(df.columns):
                        target_column = df.columns[choice-1]
                        break
                except ValueError:
                    pass
                print("❌ Invalid selection")
        
        print(f"✅ Target selected: '{target_column}'")
        
        # Analyze target distribution (without printing classes)
        target_dist = df[target_column].value_counts().sort_index()
        imbalance_ratio = target_dist.max() / target_dist.min()
        print(f"📊 Target distribution: {len(target_dist)} classes")
        print(f"⚖️ Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        # Initialize components
        output_dir = f"./output/optimized_pipeline_{timestamp}"
        processor = OptimizedDataProcessor(verbose=True)
        imbalance_handler = ExtremeImbalanceHandler(verbose=True)
        ensemble_builder = AdvancedEnsembleBuilder(verbose=True)
        evaluator = ComprehensiveEvaluator(output_dir, verbose=True)
        
        # Analyze and preprocess data
        print("\n🔍 Analyzing dataset structure...")
        feature_analysis = processor.analyze_features(df, target_column)
        
        # Preprocess data
        X, y = processor.preprocess_data(df, target_column, fit=True)
        
        print(f"✅ Final dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
        
        # Train/test split
        print("\n✂️ Creating train/test split...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails due to class distribution
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
        print(f"   Train: {len(X_train):,} samples")
        print(f"   Test: {len(X_test):,} samples")
        
        # Handle extreme imbalance
        strategy, ratio = imbalance_handler.analyze_and_recommend(y_train)
        X_train_balanced, y_train_balanced = imbalance_handler.apply_strategy(
            X_train, y_train, strategy
        )
        
        # Build and train ensemble
        ensemble = ensemble_builder.build_ensemble(strategy)
        
        print("\n🏋️ Training ensemble...")
        start_time = datetime.now()
        
        ensemble.fit(X_train_balanced, y_train_balanced)
        
        training_time = (datetime.now() - start_time).total_seconds() / 60
        print(f"✅ Training completed in {training_time:.2f} minutes")
        
        # Evaluate model
        class_names = [f'Class_{c}' for c in sorted(df[target_column].unique())]
        results = evaluator.evaluate_model(ensemble, X_test, y_test, class_names)
        
        # Feature importance (if available)
        feature_importance = {}
        try:
            if hasattr(ensemble, 'feature_importances_'):
                importance = ensemble.feature_importances_
            else:
                # Get from first model that has it
                for name, model in ensemble_builder.models.items():
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        break
                else:
                    importance = None
                    
            if importance is not None:
                feature_importance = dict(zip(X.columns, importance))
                # Sort and get top features
                top_features = dict(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True)[:20])
                
                print(f"\n🏆 Top 10 Important Features:")
                for i, (feat, imp) in enumerate(list(top_features.items())[:10], 1):
                    print(f"   {i:2d}. {feat}: {imp:.4f}")
                    
        except Exception as e:
            print(f"⚠️ Could not extract feature importance: {e}")
        
        # Save comprehensive results
        final_results = {
            'pipeline_info': {
                'timestamp': timestamp,
                'data_source': data_path,
                'training_time_minutes': training_time,
                'strategy_used': strategy,
                'imbalance_ratio': ratio
            },
            'data_summary': {
                'original_shape': df.shape,
                'processed_shape': X.shape,
                'target_column': target_column,
                'feature_analysis': feature_analysis,
                'class_distribution': target_dist.to_dict()
            },
            'model_performance': results,
            'feature_importance': feature_importance,
            'models_used': list(ensemble_builder.models.keys())
        }
        
        # Save results
        results_path = Path(output_dir) / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
            
        # Save model
        model_path = Path(output_dir) / 'trained_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': ensemble,
                'processor': processor,
                'target_column': target_column,
                'class_names': class_names
            }, f)
        
        # Create summary report
        summary_path = Path(output_dir) / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("OPTIMIZED IMBALANCED LEARNING PIPELINE - RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dataset: {data_path}\n")
            f.write(f"Target: {target_column}\n")
            f.write(f"Original shape: {df.shape[0]:,} × {df.shape[1]}\n")
            f.write(f"Processed shape: {X.shape[0]:,} × {X.shape[1]}\n")
            f.write(f"Imbalance ratio: {ratio:.1f}:1\n")
            f.write(f"Strategy used: {strategy}\n")
            f.write(f"Training time: {training_time:.2f} minutes\n\n")
            
            f.write("MODEL PERFORMANCE:\n")
            f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
            f.write(f"Macro F1-Score: {results['classification_report']['macro avg']['f1-score']:.4f}\n")
            f.write(f"Weighted F1-Score: {results['classification_report']['weighted avg']['f1-score']:.4f}\n\n")
            
            f.write("ENSEMBLE COMPONENTS:\n")
            for i, model_name in enumerate(ensemble_builder.models.keys(), 1):
                f.write(f"{i}. {model_name.upper()}\n")
            
            if feature_importance:
                f.write(f"\nTOP 15 IMPORTANT FEATURES:\n")
                for i, (feat, imp) in enumerate(list(top_features.items())[:15], 1):
                    f.write(f"{i:2d}. {feat}: {imp:.4f}\n")
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📊 Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"🎯 Macro F1-Score: {results['classification_report']['macro avg']['f1-score']:.4f}")
        print(f"⚖️ Strategy Used: {strategy.upper()}")
        print(f"🤖 Models: {len(ensemble_builder.models)} ensemble members")
        print(f"⏱️ Training Time: {training_time:.2f} minutes")
        print(f"📁 Results saved to: {output_dir}")
        print("\n📊 Generated Files:")
        print(f"   • Model: {model_path}")
        print(f"   • Results: {results_path}")
        print(f"   • Summary: {summary_path}")
        print(f"   • Plots: {output_dir}/confusion_matrix.png")
        print("="*60)
        
        return final_results
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return None
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup
        gc.collect()


class ModelPredictor:
    """Load and use trained model for predictions"""
    
    def __init__(self, model_path):
        """Load trained model and preprocessor"""
        with open(model_path, 'rb') as f:
            self.saved_objects = pickle.load(f)
            
        self.model = self.saved_objects['model']
        self.processor = self.saved_objects['processor']
        self.target_column = self.saved_objects['target_column']
        self.class_names = self.saved_objects['class_names']
        
    def predict_new_data(self, df):
        """Make predictions on new data"""
        # Preprocess using fitted processor
        X_processed, _ = self.processor.preprocess_data(df, self.target_column, fit=False)
        
        # Predict
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'prediction_label': [self.class_names[p] for p in predictions]
        })
        
        # Add probability columns
        for i, class_name in enumerate(self.class_names):
            results[f'prob_{class_name}'] = probabilities[:, i]
            
        return results


def predict_mode():
    """Interactive prediction mode"""
    print("\n🔮 PREDICTION MODE")
    print("="*30)
    
    model_path = input("📁 Enter model file path (.pkl): ").strip().strip('"')
    if not os.path.exists(model_path):
        print("❌ Model file not found")
        return
        
    predictor = ModelPredictor(model_path)
    print(f"✅ Model loaded successfully")
    print(f"   Target: {predictor.target_column}")
    print(f"   Classes: {predictor.class_names}")
    
    while True:
        data_path = input("\n📂 Enter data file for prediction (.parquet/.csv) or 'quit': ").strip().strip('"')
        
        if data_path.lower() == 'quit':
            break
            
        if not os.path.exists(data_path):
            print("❌ File not found")
            continue
            
        try:
            # Load data
            if data_path.lower().endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)
                
            print(f"📊 Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Make predictions
            results = predictor.predict_new_data(df)
            
            # Save results
            output_path = data_path.rsplit('.', 1)[0] + '_predictions.csv'
            results.to_csv(output_path, index=False)
            
            print(f"✅ Predictions saved to: {output_path}")
            print(f"📊 Prediction summary:")
            print(results['prediction_label'].value_counts())
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")


if __name__ == "__main__":
    print("🚀 OPTIMIZED IMBALANCED LEARNING TOOLKIT")
    print("="*45)
    print("1. Train new model")
    print("2. Use existing model for predictions")
    print("="*45)
    
    while True:
        try:
            choice = int(input("Select option (1-2): "))
            if choice in [1, 2]:
                break
        except ValueError:
            pass
        print("❌ Invalid choice")
    
    if choice == 1:
        result = main()
        if result:
            print(f"\n✅ Training completed successfully!")
            print(f"📁 Check results in: {result['pipeline_info']['timestamp']}")
    else:
        predict_mode()
        
    print("\n🏁 Done!")