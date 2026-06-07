import os
import sys
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch
import torch

# Packages
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix
)

# Block
from utils.data_split_ import split_dataset

warnings.filterwarnings('ignore')

# ---------------------- Configuration Parameters ----------------------
n_folds = 5  # Number of folds
pt_base_path = "./datasets/pt_file"
file_path = "datasets/Clin/MVI-HCC-samples.xlsx"
label_col = "标签1（MVI+卫星灶；0阴性；阳性1）"  # Binary classification label column

# Base path for saving results
results_base_path = "./logs/Clin_ML"
os.makedirs(results_base_path, exist_ok=True)

# Grid search parameter configuration for each model
model_configs = {
    'LogisticRegression': {
        'model_class': LogisticRegression,
        'param_grid': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None],
        },
        'fixed_params': {
            'max_iter': 2000,
            'random_state': 42
        }
    },
    'MLP': {
        'model_class': MLPClassifier,
        'param_grid': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (500, ), (600, )],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
        },
        'fixed_params': {
            'max_iter': 2000,
            'random_state': 42,
            'early_stopping': True
        }
    },
    'XGBoost': {
        'model_class': XGBClassifier,
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
        },
        'fixed_params': {
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
    },
    'SVM': {
        'model_class': SVC,
        'param_grid': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto'],
            'class_weight': ['balanced', None],
        },
        'fixed_params': {
            'random_state': 42,
            'probability': True  # Must be enabled to calculate probabilities
        }
    },
    'RandomForest': {
        'model_class': RandomForestClassifier,
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None],
        },
        'fixed_params': {
            'random_state': 42
        }
    }
}

# Discrete variable columns
discrete_cols = [
    # "肝炎类型（0无；1肝炎；2酒精肝）",
    # "临床症状（0无；1有）",
    # "内部血流信号（报告结果：0无血流；1内部可见血流）",
    "慢性病史（0无；1有）",
    # "病灶形态（报告结果，圆形0；椭圆形1；分叶状2；不规则形3；其他4）",
    # "病灶边界(报告结果，清0；不清1)",
    # "病灶内部回声水平（报告结果；其他0；低回声1）",
    # "病灶内部回声分布（均匀0；不均匀1）",
    "增强特点（0均匀强化；1不均匀强化；2可见三期无增强区）",
    # "门脉期（0：等增强；1低增强；2高增强）",
    "延迟期(等0；1低)",
    # "边界光整（0光整；1不光整）",
    # "后方回声（0正常；1增强；2衰减）",
    # "多发融合结节（0无；1有）",
    # "周边低回声晕（0无；1有）",
    # "假包膜（增强影像：0无；1有）",
    "坏死（0无；1有）",
    # "CDFI自评",
    # "原发肿瘤病史",
    # "家族恶性肿瘤病史",
    # "背景肝脏（正常0；肝实质回声粗强1；脂肪肝2；肝硬化3）",
    "AFP分类"
]

# Continuous variable columns
float_cols = [
    "病灶最大径(单位cm）",
    # "动脉期增强开始时间（S）",
    # "达峰时间（S）",
    "达峰用时",
    "低增强开始时间(s）",
    # "年龄",
    # "CEA",
    "AFP",
    "周边浸润（mm)"
]
# --------------------------------------------------------------------------

# Global variable: Store summary results of all models
all_models_summary = {}

def load_pt_patient_ids(pt_file_path):
    """Load PT file and return patient ID list (converted to string format)"""
    try:
        data = torch.load(pt_file_path, map_location="cpu", weights_only=False)
        patient_ids = [str(key) for key in data.keys()]
        print(f"Loaded {len(patient_ids)} patient IDs from {pt_file_path}")
        return patient_ids
    except Exception as e:
        raise ValueError(f"Failed to load PT file: {str(e)}")

def preprocess_data(df, label_col, discrete_cols, float_cols, train_modes=None, train_medians=None):
    """
    Data preprocessing: Missing value imputation, label type conversion
    - Training set: Calculate and return mode/median for imputation
    - Validation/Test set: Use training set's mode/median for imputation
    """
    # Convert label to int type
    df[label_col] = df[label_col].astype(int)
    
    # Filter required columns
    all_needed_cols = [label_col, "患者编号"] + discrete_cols + float_cols
    df = df[all_needed_cols].copy()
    
    # Training set: Calculate imputation values
    if train_modes is None or train_medians is None:
        train_modes = {col: df[col].mode()[0] for col in discrete_cols}
        train_medians = {col: df[col].median() for col in float_cols}
    
    # Fill missing values
    for col in discrete_cols:
        df[col] = df[col].fillna(train_modes[col])
    for col in float_cols:
        df[col] = df[col].fillna(train_medians[col])
    
    return df, train_modes, train_medians

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate all evaluation metrics
    Returns: AUC, ACC, TPR, TNR, Precision, F1-score
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
    
    return {
        'AUC': auc,
        'ACC': acc,
        'TPR': tpr,
        'TNR': tnr,
        'Precision': precision,
        'F1-score': f1
    }

def print_metrics(metrics_dict, dataset_name="Dataset"):
    """
    Print evaluation metrics
    """
    print(f"\n{dataset_name} Performance Metrics:")
    print("="*50)
    print(f"AUC:        {metrics_dict['AUC']:.4f}")
    print(f"ACC:        {metrics_dict['ACC']:.4f}")
    print(f"TPR:        {metrics_dict['TPR']:.4f}")
    print(f"TNR:        {metrics_dict['TNR']:.4f}")
    print(f"Precision:  {metrics_dict['Precision']:.4f}")
    print(f"F1-score:   {metrics_dict['F1-score']:.4f}")
    print("="*50)

def check_param_compatibility(model_name, params):
    """
    Check parameter compatibility
    Returns: (is_compatible, skip_reason)
    """
    if model_name == 'LogisticRegression':
        if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
            return False, "L1 regularization does not support this solver"
    return True, None

def calculate_mean_std(values):
    """
    Calculate mean and standard deviation
    Parameters:
        values: 1D array/list containing metric values across folds (e.g., AUC for 5 folds)
    Returns:
        mean_val: Mean value
        std_val: Standard deviation
    """
    values = np.array(values)
    if len(values) == 0:
        return 0.0, 0.0

    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)  # Use sample standard deviation

    return mean_val, std_val

def save_all_models_results_to_txt(all_models_summary, save_path):
    """
    Save all models' mean±std results to txt file
    """
    # Define metrics list
    metrics_list = ['AUC', 'ACC', 'TPR', 'TNR', 'Precision', 'F1']

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("All Models Cross-Fold Evaluation Results (Mean ± Standard Deviation)\n")
        f.write("="*100 + "\n\n")

        for model_name, results in all_models_summary.items():
            f.write(f"【{model_name}】\n")
            f.write("-"*80 + "\n")

            # Validation set results
            f.write("Sub-validation set:\n")
            for metric in metrics_list:
                res = results['val'][metric]
                if metric == 'AUC':
                    f.write(f"  {metric}: {res['mean']:.4f} ± {res['std']:.4f}\n")
                else:
                    f.write(f"  {metric}: {res['mean']*100:.2f}% ± {res['std']*100:.2f}%\n")

            # Test set results
            f.write("\nTest set:\n")
            for metric in metrics_list:
                res = results['test'][metric]
                if metric == 'AUC':
                    f.write(f"  {metric}: {res['mean']:.4f} ± {res['std']:.4f}\n")
                else:
                    f.write(f"  {metric}: {res['mean']*100:.2f}% ± {res['std']*100:.2f}%\n")

            # External test set results
            f.write("\nExternal test set:\n")
            for metric in metrics_list:
                res = results['external_test'][metric]
                if metric == 'AUC':
                    f.write(f"  {metric}: {res['mean']:.4f} ± {res['std']:.4f}\n")
                else:
                    f.write(f"  {metric}: {res['mean']*100:.2f}% ± {res['std']*100:.2f}%\n")

            # Detailed results per fold
            f.write("\nBest results per fold:\n")
            best_per_fold = results['best_per_fold']
            for fold in best_per_fold['fold']:
                fold_data = best_per_fold[best_per_fold['fold'] == fold].iloc[0]
                f.write(f"  Fold {int(fold)} - Sub-validation AUC: {fold_data['val_AUC']:.4f} | "
                        f"Test AUC: {fold_data['test_AUC']:.4f} | "
                        f"External test AUC: {fold_data['external_test_AUC']:.4f}\n")

            f.write("\n" + "="*100 + "\n\n")

    print(f"\nAll model results saved to: {save_path}")

def train_single_model(model_name, model_config, X_train, y_train, X_val, y_val, X_test, y_test,
                       X_external_test, y_external_test, preprocessor, fold_num):
    """
    Train a single model with all parameter combinations
    """
    print("\n" + "="*80)
    print(f"Model: {model_name} | Fold {fold_num}/{n_folds}")
    print("="*80)

    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(model_config['param_grid']))
    total_combinations = len(param_combinations)
    print(f"Total {total_combinations} parameter combinations to test\n")

    # Store current model results for current fold
    fold_results = []
    best_val_auc = 0
    best_params = None
    best_model = None

    # Iterate through all parameter combinations
    for idx, params in enumerate(param_combinations, 1):
        print(f"[{model_name}][Fold {fold_num}][{idx}/{total_combinations}] Testing parameters: {params}")

        # Check parameter compatibility
        is_compatible, skip_reason = check_param_compatibility(model_name, params)
        if not is_compatible:
            print(f"  ⚠ Skipped: {skip_reason}")
            continue

        # Merge grid parameters and fixed parameters
        all_params = {**params, **model_config['fixed_params']}

        # Build model
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model_config['model_class'](**all_params))
        ])

        # Train model
        try:
            model.fit(X_train, y_train)

            # Validation set evaluation
            val_pred = model.predict(X_val)
            val_pred_proba = model.predict_proba(X_val)[:, 1]
            val_metrics = calculate_metrics(y_val, val_pred, val_pred_proba)

            # Test set evaluation
            test_pred = model.predict(X_test)
            test_pred_proba = model.predict_proba(X_test)[:, 1]
            test_metrics = calculate_metrics(y_test, test_pred, test_pred_proba)

            # External test set evaluation
            external_test_pred = model.predict(X_external_test)
            external_test_pred_proba = model.predict_proba(X_external_test)[:, 1]
            external_test_metrics = calculate_metrics(y_external_test, external_test_pred, external_test_pred_proba)

            print(f"  Validation AUC: {val_metrics['AUC']:.4f} | Test AUC: {test_metrics['AUC']:.4f} | External test AUC: {external_test_metrics['AUC']:.4f}")

            # Record results (add fold and model info)
            result = {
                'model': model_name,
                'fold': fold_num,
                **{f'param_{k}': v for k, v in params.items()},  # Add prefix to parameter names to avoid conflicts
                'val_AUC': val_metrics['AUC'],
                'val_ACC': val_metrics['ACC'],
                'val_TPR': val_metrics['TPR'],
                'val_TNR': val_metrics['TNR'],
                'val_Precision': val_metrics['Precision'],
                'val_F1': val_metrics['F1-score'],
                'test_AUC': test_metrics['AUC'],
                'test_ACC': test_metrics['ACC'],
                'test_TPR': test_metrics['TPR'],
                'test_TNR': test_metrics['TNR'],
                'test_Precision': test_metrics['Precision'],
                'test_F1': test_metrics['F1-score'],
                'external_test_AUC': external_test_metrics['AUC'],
                'external_test_ACC': external_test_metrics['ACC'],
                'external_test_TPR': external_test_metrics['TPR'],
                'external_test_TNR': external_test_metrics['TNR'],
                'external_test_Precision': external_test_metrics['Precision'],
                'external_test_F1': external_test_metrics['F1-score']
            }
            fold_results.append(result)

            # Update best model for current fold
            if val_metrics['AUC'] > best_val_auc:
                best_val_auc = val_metrics['AUC']
                best_params = params
                best_model = model
                print(f"  ✓ Found better model! Validation AUC: {best_val_auc:.4f}")

        except Exception as e:
            print(f"  ✗ Training failed: {str(e)}")
            continue

    # Output best results for current model and current fold
    if best_model is not None:
        print("\n" + "="*70)
        print(f"{model_name} - Fold {fold_num} Best Model (Based on Validation AUC)")
        print("="*70)
        print(f"Best parameter combination: {best_params}")

        # Detailed validation set evaluation
        val_pred = best_model.predict(X_val)
        val_pred_proba = best_model.predict_proba(X_val)[:, 1]
        val_metrics = calculate_metrics(y_val, val_pred, val_pred_proba)
        print_metrics(val_metrics, "Validation set")

        print("\nValidation set confusion matrix:")
        print(confusion_matrix(y_val, val_pred))

        # Detailed test set evaluation
        test_pred = best_model.predict(X_test)
        test_pred_proba = best_model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, test_pred, test_pred_proba)
        print_metrics(test_metrics, "Test set")

        print("\nTest set confusion matrix:")
        print(confusion_matrix(y_test, test_pred))

        # Detailed external test set evaluation
        external_test_pred = best_model.predict(X_external_test)
        external_test_pred_proba = best_model.predict_proba(X_external_test)[:, 1]
        external_test_metrics = calculate_metrics(y_external_test, external_test_pred, external_test_pred_proba)
        print_metrics(external_test_metrics, "External test set")

        print("\nExternal test set confusion matrix:")
        print(confusion_matrix(y_external_test, external_test_pred))

    return fold_results

def run_all_models():
    """
    Complete workflow: Iterate through all models → 5-fold cross-validation → grid search → evaluation → save results
    """
    global all_models_summary
    all_models_summary = {}

    # Define metrics list
    metrics_list = ['AUC', 'ACC', 'TPR', 'TNR', 'Precision', 'F1']

    # Iterate through all models
    for model_name, model_config in model_configs.items():
        print("\n" + "="*100)
        print(f"Starting to train model: {model_name}")
        print("="*100)

        # Store all fold results for current model
        all_folds_results = []

        # Iterate through 5 folds
        for fold_num in range(1, n_folds + 1):
            print("\n" + "="*80)
            print(f"Model: {model_name} | Fold {fold_num}/{n_folds}")
            print("="*80)

            # Use split_dataset function to split datasets
            # Training set and sub-validation set are split from original train (sub_val=True)
            # Test set is split from original val (sub_val=False)
            # External test set is split from original test (sub_val=False)
            train_ids = split_dataset(fold_num=fold_num, split="train", pt_path=pt_base_path, sub_val_ratio=0.25, sub_val=True)
            val_ids = split_dataset(fold_num=fold_num, split="val", pt_path=pt_base_path, sub_val_ratio=0.25, sub_val=True)
            test_ids = split_dataset(fold_num=fold_num, split="val", pt_path=pt_base_path, sub_val=False)
            external_test_ids = split_dataset(fold_num=fold_num, split="test", pt_path=pt_base_path, sub_val=False)

            print(f"\nDataset split results:")
            print(f"Training set: {len(train_ids)} samples")
            print(f"Sub-validation set: {len(val_ids)} samples")
            print(f"Test set: {len(test_ids)} samples")
            print(f"External test set: {len(external_test_ids)} samples")

            # Check for overlapping patient IDs between datasets
            overlap_train_val = set(train_ids) & set(val_ids)
            overlap_train_test = set(train_ids) & set(test_ids)
            overlap_val_test = set(val_ids) & set(test_ids)
            overlap_train_external = set(train_ids) & set(external_test_ids)
            overlap_val_external = set(val_ids) & set(external_test_ids)
            overlap_test_external = set(test_ids) & set(external_test_ids)

            if overlap_train_val:
                warnings.warn(f"{model_name} Fold {fold_num}: Training set and sub-validation set have {len(overlap_train_val)} duplicate IDs")
            if overlap_train_test:
                warnings.warn(f"{model_name} Fold {fold_num}: Training set and test set have {len(overlap_train_test)} duplicate IDs")
            if overlap_val_test:
                warnings.warn(f"{model_name} Fold {fold_num}: Sub-validation set and test set have {len(overlap_val_test)} duplicate IDs")
            if overlap_train_external:
                warnings.warn(f"{model_name} Fold {fold_num}: Training set and external test set have {len(overlap_train_external)} duplicate IDs")
            if overlap_val_external:
                warnings.warn(f"{model_name} Fold {fold_num}: Sub-validation set and external test set have {len(overlap_val_external)} duplicate IDs")
            if overlap_test_external:
                warnings.warn(f"{model_name} Fold {fold_num}: Test set and external test set have {len(overlap_test_external)} duplicate IDs")

            # 2. Load Excel data and process patient IDs
            df = pd.read_excel(file_path, engine='openpyxl')
            df["患者编号"] = df["患者编号"].astype(str)  # Convert to string format
            print(f"Original data has {len(df)} samples")

            # 4. Split datasets according to new splitting method
            train_df = df[df["患者编号"].isin(train_ids)].reset_index(drop=True)
            val_df = df[df["患者编号"].isin(val_ids)].reset_index(drop=True)
            test_df = df[df["患者编号"].isin(test_ids)].reset_index(drop=True)
            external_test_df = df[df["患者编号"].isin(external_test_ids)].reset_index(drop=True)

            print(f"\nFinal dataset split:")
            print(f"Training set: {len(train_df)} samples")
            print(f"Sub-validation set: {len(val_df)} samples")
            print(f"Test set: {len(test_df)} samples")
            print(f"External test set: {len(external_test_df)} samples")

            # 5. Data preprocessing (using training set statistics for imputation)
            train_df, train_modes, train_medians = preprocess_data(
                train_df, label_col, discrete_cols, float_cols
            )
            val_df, _, _ = preprocess_data(
                val_df, label_col, discrete_cols, float_cols, train_modes, train_medians
            )
            test_df, _, _ = preprocess_data(
                test_df, label_col, discrete_cols, float_cols, train_modes, train_medians
            )
            external_test_df, _, _ = preprocess_data(
                external_test_df, label_col, discrete_cols, float_cols, train_modes, train_medians
            )

            # 6. Prepare features and labels
            X_train = train_df[discrete_cols + float_cols]
            y_train = train_df[label_col]
            X_val = val_df[discrete_cols + float_cols]
            y_val = val_df[label_col]
            X_test = test_df[discrete_cols + float_cols]
            y_test = test_df[label_col]
            X_external_test = external_test_df[discrete_cols + float_cols]
            y_external_test = external_test_df[label_col]

            # 7. Build preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("discrete", OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore'), discrete_cols),
                    ("float", StandardScaler(), float_cols)
                ],
                remainder="drop"
            )

            # 8. Train all parameter combinations for current model and current fold
            fold_results = train_single_model(
                model_name, model_config, X_train, y_train, X_val, y_val, X_test, y_test,
                X_external_test, y_external_test, preprocessor, fold_num
            )

            # 9. Add to total results for current model
            all_folds_results.extend(fold_results)

        # 10. Save all fold results for current model to CSV
        if all_folds_results:
            results_df = pd.DataFrame(all_folds_results)
            # Sort by fold and val_AUC
            results_df = results_df.sort_values(['fold', 'val_AUC'], ascending=[True, False])

            # Save to CSV
            csv_path = f"{results_base_path}/{model_name}_grid_search_results_all_folds.csv"
            results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print("\n" + "="*80)
            print(f"{model_name} - All {n_folds} folds results saved to: {csv_path}")
            print("="*80)

            # 11. Calculate and output cross-fold performance (mean±std) for current model
            print("\n" + "="*80)
            print(f"{model_name} - Cross-fold performance (best model per fold) - Mean ± Standard Deviation")
            print("="*80)

            # Get best results per fold
            best_per_fold = results_df.groupby('fold').first().reset_index()

            # Calculate mean and standard deviation for validation set metrics
            val_ci_results = {}
            for metric in metrics_list:
                values = best_per_fold[f'val_{metric}']
                mean_val, std_val = calculate_mean_std(values)
                val_ci_results[metric] = {
                    'mean': mean_val,
                    'std': std_val
                }

            # Calculate mean and standard deviation for test set metrics
            test_ci_results = {}
            for metric in metrics_list:
                values = best_per_fold[f'test_{metric}']
                mean_val, std_val = calculate_mean_std(values)
                test_ci_results[metric] = {
                    'mean': mean_val,
                    'std': std_val
                }

            # Calculate mean and standard deviation for external test set metrics
            external_test_ci_results = {}
            for metric in metrics_list:
                values = best_per_fold[f'external_test_{metric}']
                mean_val, std_val = calculate_mean_std(values)
                external_test_ci_results[metric] = {
                    'mean': mean_val,
                    'std': std_val
                }

            # Print validation set results
            print("\nSub-validation set average performance (Mean ± Standard Deviation):")
            print("="*50)
            for metric in metrics_list:
                res = val_ci_results[metric]
                if metric == 'AUC':
                    print(f"{metric}:        {res['mean']:.4f} ± {res['std']:.4f}")
                else:
                    print(f"{metric}:        {res['mean']*100:.2f}% ± {res['std']*100:.2f}%")

            # Print test set results
            print("\nTest set average performance (Mean ± Standard Deviation):")
            print("="*50)
            for metric in metrics_list:
                res = test_ci_results[metric]
                if metric == 'AUC':
                    print(f"{metric}:        {res['mean']:.4f} ± {res['std']:.4f}")
                else:
                    print(f"{metric}:        {res['mean']*100:.2f}% ± {res['std']*100:.2f}%")

            # Print external test set results
            print("\nExternal test set average performance (Mean ± Standard Deviation):")
            print("="*50)
            for metric in metrics_list:
                res = external_test_ci_results[metric]
                if metric == 'AUC':
                    print(f"{metric}:        {res['mean']:.4f} ± {res['std']:.4f}")
                else:
                    print(f"{metric}:        {res['mean']*100:.2f}% ± {res['std']*100:.2f}%")
            print("="*80)

            # Save current model results to global summary
            all_models_summary[model_name] = {
                'val': val_ci_results,
                'test': test_ci_results,
                'external_test': external_test_ci_results,
                'best_per_fold': best_per_fold
            }

    # After all models are trained, save summary results to txt
    txt_save_path = f"{results_base_path}/all_models_95ci_results.txt"
    save_all_models_results_to_txt(all_models_summary, txt_save_path)

    print("\n" + "="*100)
    print("All model training completed!")
    print("="*100)

# ---------------------- Execute main workflow ----------------------
if __name__ == "__main__":
    run_all_models()