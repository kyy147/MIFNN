import os
import sys
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Packages
import copy
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from utils.data_split_ import split_dataset

warnings.filterwarnings('ignore')
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(1)


# ---------------------- Configuration parameters (modify according to your needs) ----------------------
n_folds = 5
pt_base_path = "./datasets/pt_file"
file_path = "datasets/Clin/MVI-HCC-samples.xlsx"
label_col = "MVI"
sub_val_ratio = 0.25  # Ratio to split sub-validation set from training set

results_base_path = "./logs/Clin_GCN"
os.makedirs(results_base_path, exist_ok=True)

# Keep only GCN model
model_configs = {
    'GCN': {
        'param_grid': {
            'hidden_dim': [32, 64, 128],
            'dropout': [0.2, 0.4, 0.5],
            'lr': [1e-3, 5e-4],
            'weight_decay': [1e-4, 5e-4],
            'num_layers': [2, 3],
            'k': [5, 8, 10],           # kNN graph neighbor count
            'epochs': [300]
        },
        'fixed_params': {
            'random_state': 42,
            'patience': 50
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable: Store summary results of all models
all_models_summary = {}


def preprocess_data(df, label_col, discrete_cols, float_cols, train_modes=None, train_medians=None):
    """
    Missing value imputation, convert label to int
    """
    df = df.copy()
    df[label_col] = df[label_col].astype(int)

    all_needed_cols = [label_col, "患者编号"] + discrete_cols + float_cols
    df = df[all_needed_cols].copy()

    if train_modes is None or train_medians is None:
        train_modes = {}
        for col in discrete_cols:
            mode_vals = df[col].mode()
            train_modes[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else 0

        train_medians = {}
        for col in float_cols:
            train_medians[col] = df[col].median()

    for col in discrete_cols:
        df[col] = df[col].fillna(train_modes[col])
    for col in float_cols:
        df[col] = df[col].fillna(train_medians[col])

    return df, train_modes, train_medians


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Return raw ratio values:
    AUC, ACC, TPR, TNR, Precision, F1-score
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    return {
        'AUC': auc,
        'ACC': acc,
        'TPR': tpr,
        'TNR': tnr,
        'Precision': precision,
        'F1-score': f1
    }


def format_metric(metric_name, value):
    """
    Keep AUC as decimal, convert other metrics to percentage
    """
    if metric_name == 'AUC':
        return f"{value:.4f}"
    return f"{value * 100:.2f}%"


def format_mean_std(metric_name, mean_val, std_val):
    """
    AUC: mean ± std
    Others: xx.xx% ± yy.yy%
    """
    if metric_name == 'AUC':
        return f"{mean_val:.4f} ± {std_val:.4f}"
    return f"{mean_val * 100:.2f}% ± {std_val * 100:.2f}%"


def print_metrics(metrics_dict, dataset_name="Dataset"):
    print(f"\n{dataset_name} performance metrics:")
    print("=" * 50)
    print(f"AUC:        {format_metric('AUC', metrics_dict['AUC'])}")
    print(f"ACC:        {format_metric('ACC', metrics_dict['ACC'])}")
    print(f"TPR:        {format_metric('TPR', metrics_dict['TPR'])}")
    print(f"TNR:        {format_metric('TNR', metrics_dict['TNR'])}")
    print(f"Precision:  {format_metric('Precision', metrics_dict['Precision'])}")
    print(f"F1-score:   {format_metric('F1-score', metrics_dict['F1-score'])}")
    print("=" * 50)


def calculate_mean_std(values):
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def save_all_models_results_to_txt(all_models_summary, save_path):
    """
    Save all models' cross-fold results to txt
    Five-fold results use mean ± standard deviation
    Other metrics except AUC are percentages
    Includes: sub-validation set, test set, external test set
    """
    metrics_list = ['AUC', 'ACC', 'TPR', 'TNR', 'Precision', 'F1']

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("All models cross-fold evaluation results (mean ± standard deviation)\n")
        f.write("=" * 100 + "\n\n")

        for model_name, results in all_models_summary.items():
            f.write(f"【{model_name}】\n")
            f.write("-" * 80 + "\n")

            f.write("Sub-validation set (model selection):\n")
            for metric in metrics_list:
                res = results['sub_val'][metric]
                f.write(f"  {metric}: {format_mean_std(metric, res['mean'], res['std'])}\n")

            f.write("\nTest set:\n")
            for metric in metrics_list:
                res = results['test'][metric]
                f.write(f"  {metric}: {format_mean_std(metric, res['mean'], res['std'])}\n")

            f.write("\nExternal test set:\n")
            for metric in metrics_list:
                res = results['external_test'][metric]
                f.write(f"  {metric}: {format_mean_std(metric, res['mean'], res['std'])}\n")

            f.write("\nBest results per fold:\n")
            best_per_fold = results['best_per_fold']
            for fold in best_per_fold['fold']:
                fold_data = best_per_fold[best_per_fold['fold'] == fold].iloc[0]
                f.write(
                    f"  Fold {int(fold)} - "
                    f"Sub-validation AUC: {fold_data['sub_val_AUC']:.4f} | "
                    f"Test AUC: {fold_data['test_AUC']:.4f} | "
                    f"External test AUC: {fold_data['external_test_AUC']:.4f}\n"
                )

            f.write("\n" + "=" * 100 + "\n\n")

    print(f"\nAll model results saved to: {save_path}")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_features(X_train, X_sub_val, X_test, X_external):
    """
    One-hot encode categorical features, standardize continuous features
    Fit only on training set
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("discrete", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), discrete_cols),
            ("float", StandardScaler(), float_cols)
        ],
        remainder="drop"
    )

    X_train_enc = preprocessor.fit_transform(X_train)
    X_sub_val_enc = preprocessor.transform(X_sub_val)
    X_test_enc = preprocessor.transform(X_test)
    X_external_enc = preprocessor.transform(X_external)

    return X_train_enc, X_sub_val_enc, X_test_enc, X_external_enc, preprocessor


def build_knn_graph(X_all, k=8):
    """
    Build undirected kNN graph based on all node features
    """
    n_samples = X_all.shape[0]
    if n_samples <= 1:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        return edge_index

    k = min(k, n_samples - 1)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nbrs.fit(X_all)
    indices = nbrs.kneighbors(X_all, return_distance=False)

    edges = set()
    for i in range(n_samples):
        for j in indices[i][1:]:  # Remove self
            edges.add((i, j))
            edges.add((j, i))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index


def build_graph_data(X_train_enc, y_train, X_sub_val_enc, y_sub_val, X_test_enc, y_test, X_external_enc, y_external, k=8):
    """
    Concatenate train/sub_val/test/external into one large graph, use masks to distinguish
    """
    X_all = np.vstack([X_train_enc, X_sub_val_enc, X_test_enc, X_external_enc])
    y_all = np.concatenate([y_train.values, y_sub_val.values, y_test.values, y_external.values])

    n_train = len(X_train_enc)
    n_sub_val = len(X_sub_val_enc)
    n_test = len(X_test_enc)
    n_external = len(X_external_enc)

    edge_index = build_knn_graph(X_all, k=k)

    x = torch.tensor(X_all, dtype=torch.float)
    y = torch.tensor(y_all, dtype=torch.long)

    train_mask = torch.zeros(len(X_all), dtype=torch.bool)
    sub_val_mask = torch.zeros(len(X_all), dtype=torch.bool)
    test_mask = torch.zeros(len(X_all), dtype=torch.bool)
    external_mask = torch.zeros(len(X_all), dtype=torch.bool)

    train_mask[:n_train] = True
    sub_val_mask[n_train:n_train + n_sub_val] = True
    test_mask[n_train + n_sub_val:n_train + n_sub_val + n_test] = True
    external_mask[n_train + n_sub_val + n_test:] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        sub_val_mask=sub_val_mask,
        test_mask=test_mask,
        external_mask=external_mask
    )
    return data


class GCNNet(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_classes=2, dropout=0.5, num_layers=2):
        super().__init__()
        assert num_layers >= 2, "GCN layers must be at least 2"

        self.convs = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(GCNConv(in_channels, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, num_classes))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def train_gcn_once(data, params, fixed_params):
    """
    Train a single GCN parameter combination once, select best model on sub-validation set,
    return results on sub_val/test/external
    """
    set_seed(fixed_params.get('random_state', 42))

    data = data.to(device)
    model = GCNNet(
        in_channels=data.x.shape[1],
        hidden_dim=params['hidden_dim'],
        num_classes=2,
        dropout=params['dropout'],
        num_layers=params['num_layers']
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )

    criterion = nn.CrossEntropyLoss()

    best_state_dict = None
    best_sub_val_auc = -1
    patience = fixed_params.get('patience', 50)
    wait = 0

    for epoch in range(params['epochs']):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(data.x, data.edge_index)
            probs = F.softmax(logits_eval, dim=1)[:, 1].detach().cpu().numpy()

            sub_val_probs = probs[data.sub_val_mask.cpu().numpy()]
            sub_val_pred = (sub_val_probs >= 0.5).astype(int)
            sub_val_true = data.y[data.sub_val_mask].detach().cpu().numpy()

            # If sub-validation set has only one class, AUC cannot be calculated, record as 0.5 to avoid error
            if len(np.unique(sub_val_true)) < 2:
                sub_val_auc = 0.5
            else:
                sub_val_auc = roc_auc_score(sub_val_true, sub_val_probs)

        if sub_val_auc > best_sub_val_auc:
            best_sub_val_auc = sub_val_auc
            best_state_dict = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    # Load best model
    model.load_state_dict(best_state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    # Sub-validation set metrics
    sub_val_true = data.y[data.sub_val_mask].detach().cpu().numpy()
    sub_val_probs = probs[data.sub_val_mask.cpu().numpy()]
    sub_val_pred = (sub_val_probs >= 0.5).astype(int)
    sub_val_metrics = calculate_metrics(sub_val_true, sub_val_pred, sub_val_probs)

    # Test set metrics
    test_true = data.y[data.test_mask].detach().cpu().numpy()
    test_probs = probs[data.test_mask.cpu().numpy()]
    test_pred = (test_probs >= 0.5).astype(int)
    test_metrics = calculate_metrics(test_true, test_pred, test_probs)

    # External test set metrics
    external_true = data.y[data.external_mask].detach().cpu().numpy()
    external_probs = probs[data.external_mask.cpu().numpy()]
    external_pred = (external_probs >= 0.5).astype(int)
    external_metrics = calculate_metrics(external_true, external_pred, external_probs)

    return model, sub_val_metrics, test_metrics, external_metrics


def train_single_model(model_name, model_config, X_train, y_train, X_sub_val, y_sub_val, X_test, y_test, X_external, y_external, fold_num):
    """
    Train GCN for current fold with all parameter combinations
    Select best model on sub-validation set, evaluate on test set and external test set
    """
    print("\n" + "=" * 80)
    print(f"Model: {model_name} | Fold {fold_num}/{n_folds}")
    print("=" * 80)

    param_combinations = list(ParameterGrid(model_config['param_grid']))
    total_combinations = len(param_combinations)
    print(f"Total {total_combinations} parameter combinations to test\n")

    fold_results = []
    best_sub_val_auc = -1
    best_params = None
    best_model = None
    best_data = None
    best_sub_val_metrics = None
    best_test_metrics = None
    best_external_metrics = None

    # Encode features first (only once)
    X_train_enc, X_sub_val_enc, X_test_enc, X_external_enc, _ = encode_features(
        X_train, X_sub_val, X_test, X_external
    )

    for idx, params in enumerate(param_combinations, 1):
        print(f"[{model_name}][Fold {fold_num}][{idx}/{total_combinations}] Testing parameters: {params}")

        try:
            graph_data = build_graph_data(
                X_train_enc, y_train,
                X_sub_val_enc, y_sub_val,
                X_test_enc, y_test,
                X_external_enc, y_external,
                k=params['k']
            )

            model, sub_val_metrics, test_metrics, external_metrics = train_gcn_once(
                graph_data, params, model_config['fixed_params']
            )

            print(
                f"  Sub-validation AUC: {sub_val_metrics['AUC']:.4f} | "
                f"Test AUC: {test_metrics['AUC']:.4f} | "
                f"External test AUC: {external_metrics['AUC']:.4f}"
            )

            result = {
                'model': model_name,
                'fold': fold_num,
                **{f'param_{k}': v for k, v in params.items()},
                'sub_val_AUC': sub_val_metrics['AUC'],
                'sub_val_ACC': sub_val_metrics['ACC'],
                'sub_val_TPR': sub_val_metrics['TPR'],
                'sub_val_TNR': sub_val_metrics['TNR'],
                'sub_val_Precision': sub_val_metrics['Precision'],
                'sub_val_F1': sub_val_metrics['F1-score'],
                'test_AUC': test_metrics['AUC'],
                'test_ACC': test_metrics['ACC'],
                'test_TPR': test_metrics['TPR'],
                'test_TNR': test_metrics['TNR'],
                'test_Precision': test_metrics['Precision'],
                'test_F1': test_metrics['F1-score'],
                'external_test_AUC': external_metrics['AUC'],
                'external_test_ACC': external_metrics['ACC'],
                'external_test_TPR': external_metrics['TPR'],
                'external_test_TNR': external_metrics['TNR'],
                'external_test_Precision': external_metrics['Precision'],
                'external_test_F1': external_metrics['F1-score']
            }
            fold_results.append(result)

            if sub_val_metrics['AUC'] > best_sub_val_auc:
                best_sub_val_auc = sub_val_metrics['AUC']
                best_params = params
                best_model = model
                best_data = graph_data
                best_sub_val_metrics = sub_val_metrics
                best_test_metrics = test_metrics
                best_external_metrics = external_metrics
                print(f"  ✓ Found better model! Sub-validation AUC: {best_sub_val_auc:.4f}")

        except Exception as e:
            print(f"  ✗ Training failed: {str(e)}")
            continue

    if best_model is not None:
        print("\n" + "=" * 70)
        print(f"{model_name} - Fold {fold_num} Best Model (Based on Sub-validation AUC)")
        print("=" * 70)
        print(f"Best parameter combination: {best_params}")

        print_metrics(best_sub_val_metrics, "Sub-validation set")
        print_metrics(best_test_metrics, "Test set")
        print_metrics(best_external_metrics, "External test set")

        best_model.eval()
        with torch.no_grad():
            logits = best_model(best_data.x.to(device), best_data.edge_index.to(device))
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        sub_val_true = best_data.y[best_data.sub_val_mask].detach().cpu().numpy()
        sub_val_pred = (probs[best_data.sub_val_mask.cpu().numpy()] >= 0.5).astype(int)
        print("\nSub-validation set confusion matrix:")
        print(confusion_matrix(sub_val_true, sub_val_pred, labels=[0, 1]))

        test_true = best_data.y[best_data.test_mask].detach().cpu().numpy()
        test_pred = (probs[best_data.test_mask.cpu().numpy()] >= 0.5).astype(int)
        print("\nTest set confusion matrix:")
        print(confusion_matrix(test_true, test_pred, labels=[0, 1]))

        external_true = best_data.y[best_data.external_mask].detach().cpu().numpy()
        external_pred = (probs[best_data.external_mask.cpu().numpy()] >= 0.5).astype(int)
        print("\nExternal test set confusion matrix:")
        print(confusion_matrix(external_true, external_pred, labels=[0, 1]))

    return fold_results


def run_all_models():
    """
    Complete workflow:
    Train only GCN -> 5-fold -> Grid search -> Evaluation -> Save results
    Use four parts of data: training set, sub-validation set, test set, external test set
    Select best model on sub-validation set, evaluate on test set and external test set
    """
    global all_models_summary
    all_models_summary = {}

    metrics_list = ['AUC', 'ACC', 'TPR', 'TNR', 'Precision', 'F1']

    for model_name, model_config in model_configs.items():
        print("\n" + "=" * 100)
        print(f"Starting to train model: {model_name}")
        print("=" * 100)

        all_folds_results = []

        for fold_num in range(1, n_folds + 1):
            print("\n" + "=" * 80)
            print(f"Model: {model_name} | Fold {fold_num}/{n_folds}")
            print("=" * 80)

            # 1. Use split_dataset function to read patient IDs for four datasets
            # Training set and sub-validation set are split from original train
            train_ids = split_dataset(
                fold_num=fold_num,
                split="train",
                pt_path=pt_base_path,
                sub_val_ratio=sub_val_ratio,
                sub_val=True,
                seed=42
            )
            sub_val_ids = split_dataset(
                fold_num=fold_num,
                split="val",
                pt_path=pt_base_path,
                sub_val_ratio=sub_val_ratio,
                sub_val=True,
                seed=42
            )
            # Test set uses original val
            test_ids = split_dataset(
                fold_num=fold_num,
                split="val",
                pt_path=pt_base_path,
                sub_val=False
            )
            # External test set uses original test
            external_ids = split_dataset(
                fold_num=fold_num,
                split="test",
                pt_path=pt_base_path,
                sub_val=False
            )

            # Check dataset overlap
            overlap_train_subval = set(train_ids) & set(sub_val_ids)
            overlap_train_test = set(train_ids) & set(test_ids)
            overlap_train_external = set(train_ids) & set(external_ids)
            overlap_subval_test = set(sub_val_ids) & set(test_ids)
            overlap_subval_external = set(sub_val_ids) & set(external_ids)
            overlap_test_external = set(test_ids) & set(external_ids)

            if overlap_train_subval:
                warnings.warn(f"{model_name} Fold {fold_num}: Training set and sub-validation set have {len(overlap_train_subval)} duplicate IDs")
            if overlap_train_test:
                warnings.warn(f"{model_name} Fold {fold_num}: Training set and test set have {len(overlap_train_test)} duplicate IDs")
            if overlap_train_external:
                warnings.warn(f"{model_name} Fold {fold_num}: Training set and external test set have {len(overlap_train_external)} duplicate IDs")
            if overlap_subval_test:
                warnings.warn(f"{model_name} Fold {fold_num}: Sub-validation set and test set have {len(overlap_subval_test)} duplicate IDs")
            if overlap_subval_external:
                warnings.warn(f"{model_name} Fold {fold_num}: Sub-validation set and external test set have {len(overlap_subval_external)} duplicate IDs")
            if overlap_test_external:
                warnings.warn(f"{model_name} Fold {fold_num}: Test set and external test set have {len(overlap_test_external)} duplicate IDs")

            # 2. Read Excel
            df = pd.read_excel(file_path, engine='openpyxl')
            df["患者编号"] = df["患者编号"].astype(str)
            print(f"Original data has {len(df)} samples")

            # 3. Filter IDs
            # df = df[~df["患者编号"].isin(filter_list)].reset_index(drop=True)
            # print(f"After filtering, {len(df)} samples remain")

            # 4. Split datasets
            train_df = df[df["患者编号"].isin(train_ids)].reset_index(drop=True)
            sub_val_df = df[df["患者编号"].isin(sub_val_ids)].reset_index(drop=True)
            test_df = df[df["患者编号"].isin(test_ids)].reset_index(drop=True)
            external_df = df[df["患者编号"].isin(external_ids)].reset_index(drop=True)

            print(f"\nFinal dataset split:")
            print(f"Training set: {len(train_df)} samples")
            print(f"Sub-validation set: {len(sub_val_df)} samples")
            print(f"Test set: {len(test_df)} samples")
            print(f"External test set: {len(external_df)} samples")

            # 5. Missing value handling (use only training set statistics)
            train_df, train_modes, train_medians = preprocess_data(
                train_df, label_col, discrete_cols, float_cols
            )
            sub_val_df, _, _ = preprocess_data(
                sub_val_df, label_col, discrete_cols, float_cols, train_modes, train_medians
            )
            test_df, _, _ = preprocess_data(
                test_df, label_col, discrete_cols, float_cols, train_modes, train_medians
            )
            external_df, _, _ = preprocess_data(
                external_df, label_col, discrete_cols, float_cols, train_modes, train_medians
            )

            # 6. Prepare features and labels
            X_train = train_df[discrete_cols + float_cols]
            y_train = train_df[label_col]
            X_sub_val = sub_val_df[discrete_cols + float_cols]
            y_sub_val = sub_val_df[label_col]
            X_test = test_df[discrete_cols + float_cols]
            y_test = test_df[label_col]
            X_external = external_df[discrete_cols + float_cols]
            y_external = external_df[label_col]

            # 7. Train current fold
            fold_results = train_single_model(
                model_name, model_config,
                X_train, y_train,
                X_sub_val, y_sub_val,
                X_test, y_test,
                X_external, y_external,
                fold_num
            )

            all_folds_results.extend(fold_results)

        # 8. Save all fold results for current model
        if all_folds_results:
            results_df = pd.DataFrame(all_folds_results)
            results_df = results_df.sort_values(['fold', 'sub_val_AUC'], ascending=[True, False])

            csv_path = f"{results_base_path}/{model_name}_grid_search_results_all_folds.csv"
            results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print("\n" + "=" * 80)
            print(f"{model_name} - All {n_folds} folds results saved to: {csv_path}")
            print("=" * 80)

            # 9. Take the result with highest sub-validation AUC for each fold
            best_per_fold = results_df.groupby('fold').first().reset_index()

            # 10. Calculate mean ± standard deviation
            print("\n" + "=" * 80)
            print(f"{model_name} - Cross-fold performance (best model per fold) - Mean ± Standard Deviation")
            print("=" * 80)

            sub_val_stat_results = {}
            for metric in metrics_list:
                values = best_per_fold[f'sub_val_{metric}']
                mean_val, std_val = calculate_mean_std(values)
                sub_val_stat_results[metric] = {
                    'mean': mean_val,
                    'std': std_val
                }

            test_stat_results = {}
            for metric in metrics_list:
                values = best_per_fold[f'test_{metric}']
                mean_val, std_val = calculate_mean_std(values)
                test_stat_results[metric] = {
                    'mean': mean_val,
                    'std': std_val
                }

            external_stat_results = {}
            for metric in metrics_list:
                values = best_per_fold[f'external_test_{metric}']
                mean_val, std_val = calculate_mean_std(values)
                external_stat_results[metric] = {
                    'mean': mean_val,
                    'std': std_val
                }

            print("\nSub-validation set average performance (Mean ± Standard Deviation):")
            print("=" * 50)
            for metric in metrics_list:
                res = sub_val_stat_results[metric]
                print(f"{metric}:        {format_mean_std(metric, res['mean'], res['std'])}")

            print("\nTest set average performance (Mean ± Standard Deviation):")
            print("=" * 50)
            for metric in metrics_list:
                res = test_stat_results[metric]
                print(f"{metric}:        {format_mean_std(metric, res['mean'], res['std'])}")

            print("\nExternal test set average performance (Mean ± Standard Deviation):")
            print("=" * 50)
            for metric in metrics_list:
                res = external_stat_results[metric]
                print(f"{metric}:        {format_mean_std(metric, res['mean'], res['std'])}")
            print("=" * 80)

            all_models_summary[model_name] = {
                'sub_val': sub_val_stat_results,
                'test': test_stat_results,
                'external_test': external_stat_results,
                'best_per_fold': best_per_fold
            }

    # 11. Save total results
    txt_save_path = f"{results_base_path}/all_models_mean_std_results.txt"
    save_all_models_results_to_txt(all_models_summary, txt_save_path)

    print("\n" + "=" * 100)
    print("All model training completed!")
    print("=" * 100)


# ---------------------- Execute main workflow ----------------------
if __name__ == "__main__":
    run_all_models()