import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Package
import random
import argparse
import itertools
import datetime
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, 
    precision_score, f1_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# Block
from utils.data_split_ import split_dataset

warnings.filterwarnings('ignore')

# seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(1)


# ---------------------- Parse command line arguments ----------------------
parser = argparse.ArgumentParser(description='ClinicalBERT Training with Fold and CUDA ID')
parser.add_argument('--fold_num', type=int, default=1, help='Fold number, e.g., 1, 2, 3')
parser.add_argument('--cuda_id', type=int, default=0, help='GPU ID, e.g., 0, 1, 2')
parser.add_argument('--modality', type=str, default='MVI-HCC-samples')
args = parser.parse_args()

# ---------------------- Configuration parameters ----------------------
fold_num = args.fold_num
cuda_id = args.cuda_id  
modality = args.modality
pt_path = './datasets/pt_file'
file_path = "./datasets/Clin/MVI-HCC-samples.xlsx"
label_col = "MVI"

main_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
main_save_dir = f"./logs/{modality}/Fold{fold_num}/param_search_exp_{main_timestamp}"
os.makedirs(main_save_dir, exist_ok=True)

# ClinicalBERT base configuration
MODEL_NAME = "./ckpt/pretrained_weights/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b"
MAX_LEN = 512
EPOCHS = 30
# Set specified GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
DEVICE = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} (GPU ID: {cuda_id})")

# Parameter search space
PARAM_SEARCH_SPACE = {
    'lr': [1e-5, 2e-5, 5e-5],          # Learning rate search range
    'batch_size': [8, 16, 32],          # Batch size search range
    'weight_decay': [0.0, 1e-4, 1e-5],  # Weight decay search range
    'lr_scheduler': ['cosine', 'step']  # Learning rate scheduler type
}

# Early stopping configuration
EARLY_STOP_PATIENCE = 5  # Stop if validation loss doesn't decrease for 5 epochs

# Feature columns and their English mapping (for generating English sentences)
feature_mapping = {
    "慢性病史（0无；1有）": "History of chronic disease",
    "增强特点（0均匀强化；1不均匀强化；2可见三期无增强区）": "Overall enhancement pattern",
    "延迟期(等0；1低)": "Late phase eegree of enhancement",
    "坏死（0无；1有）": "Necrosis",
    "AFP分类": "AFP classification",
    
    # Continuous variables (Chinese column name: English description)
    "病灶最大径(单位cm）": "Maximum diameter of lesion (cm)",
    "达峰用时": "Time to reach the peak (s)",
    "低增强开始时间(s）": "Start time of washout (s)",
    "AFP": "AFP level (ng/mL)",
    "周边浸润（mm)": "Tumor infiltration boundary (mm)"
}

# English mapping for discrete variable values
discrete_value_mapping = {
    "肝炎类型（0无；1肝炎；2酒精肝）": {0: "none", 1: "hepatitis", 2: "alcoholic hepatitis"},
    "临床症状（0无；1有）": {0: "absent", 1: "present"},
    "内部血流信号（报告结果：0无血流；1内部可见血流）": {0: "absent", 1: "present"},
    "慢性病史（0无；1有）": {0: "absent", 1: "present"},
    "病灶形态（报告结果，圆形0；椭圆形1；分叶状2；不规则形3；其他4）": {0: "round", 1: "oval", 2: "lobulated", 3: "irregular", 4: "other"},
    "病灶边界(报告结果，清0；不清1)": {0: "clear", 1: "unclear"},
    "病灶内部回声水平（报告结果；其他0；低回声1）": {0: "other", 1: "hypoechoic"},
    "病灶内部回声分布（均匀0；不均匀1）": {0: "uniform", 1: "non-uniform"},
    "增强特点（0均匀强化；1不均匀强化；2可见三期无增强区）": {0: "uniform enhancement", 1: "non-uniform enhancement", 2: "non-enhancing areas in three phases"},
    "门脉期（0：等增强；1低增强；2高增强）": {0: "isointense enhancement", 1: "hypo-enhancement", 2: "hyper-enhancement"},
    "延迟期(等0；1低)": {0: "isointense", 1: "hypointense"},
    "边界光整（0光整；1不光整）": {0: "regular", 1: "irregular"},
    "后方回声（0正常；1增强；2衰减）": {0: "normal", 1: "enhanced", 2: "attenuated"},
    "多发融合结节（0无；1有）": {0: "absent", 1: "present"},
    "周边低回声晕（0无；1有）": {0: "absent", 1: "present"},
    "假包膜（增强影像：0无；1有）": {0: "absent", 1: "present"},
    "坏死（0无；1有）": {0: "absent", 1: "present"},
    "AFP分类": {0: "normal", 1: "medium", 2: "high"},

    "原发肿瘤病史": {0: "absent", 1: "present"},
    "家族恶性肿瘤病史": {0: "absent", 1: "present"},
    "背景肝脏（正常0；肝实质回声粗强1；脂肪肝2；肝硬化3）": {0: "normal", 1: "coarse echotexture", 2: "fatty liver", 3: "cirrhosis"}
}


# Discrete and continuous variable columns
discrete_cols = list([k for k in feature_mapping.keys() if k in discrete_value_mapping.keys()])
float_cols = list([k for k in feature_mapping.keys() if k not in discrete_value_mapping.keys()])
# --------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all required evaluation metrics (using lowercase naming)"""
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics (changed to lowercase)
    metrics = {
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        'acc': accuracy_score(y_true, y_pred),
        'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0.0,  # Specificity/True Negative Rate
        'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # Sensitivity/Recall/True Positive Rate
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

    return metrics


def preprocess_data(df, label_col, discrete_cols, float_cols, train_modes=None, train_medians=None):
    """Data preprocessing: Missing value imputation, label type conversion"""
    df[label_col] = df[label_col].astype(int)
    all_needed_cols = [label_col, "患者编号"] + discrete_cols + float_cols
    df = df[all_needed_cols].copy()
    
    if train_modes is None or train_medians is None:
        train_modes = {col: df[col].mode()[0] for col in discrete_cols}
        train_medians = {col: df[col].median() for col in float_cols}
    
    for col in discrete_cols:
        df[col] = df[col].fillna(train_modes[col])
    for col in float_cols:
        df[col] = df[col].fillna(train_medians[col])
    
    return df, train_modes, train_medians

def generate_english_sentence(row):
    """Convert patient features to structured English sentences"""
    sentence_parts = []
    sentence_parts.append("The patient's condition is as follows:")
    
    for col in discrete_cols:
        val = row[col]
        english_col = feature_mapping[col]
        if col in discrete_value_mapping and val in discrete_value_mapping[col]:
            english_val = discrete_value_mapping[col][val]
        else:
            english_val = str(val)
        sentence_parts.append(f"{english_col} is {english_val}.")
    
    for col in float_cols:
        val = row[col]
        english_col = feature_mapping[col]
        sentence_parts.append(f"{english_col} is {val:.2f}.")
    
    full_sentence = " ".join(sentence_parts)
    return full_sentence

class ClinicalDataset(Dataset):
    """Clinical text dataset class"""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=True):
    """Create DataLoader"""
    ds = ClinicalDataset(
        texts=df['english_text'].to_numpy(),
        labels=df[label_col].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle
    )

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train one epoch"""
    model = model.train()
    losses = []
    
    for batch in tqdm(data_loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    
    return np.mean(losses)

def eval_model(model, data_loader, device):
    """Evaluate model and return complete metrics"""
    model = model.eval()
    losses = []
    predictions = []
    prediction_probs = []
    real_values = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            losses.append(loss.item())
            
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            predictions.extend(preds.cpu().numpy())
            prediction_probs.extend(probs.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
    
    # Calculate all evaluation metrics
    metrics = calculate_metrics(real_values, predictions, prediction_probs)
    
    return {
        'loss': np.mean(losses),
        'predictions': predictions,
        'prediction_probs': prediction_probs,
        'real_values': real_values,
        **metrics  # Merge all metrics
    }

def get_lr_scheduler(optimizer, scheduler_type, total_steps):
    """Get learning rate scheduler of specified type"""
    if scheduler_type == 'cosine':
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps,
            eta_min=1e-7  # Minimum learning rate
        )
    elif scheduler_type == 'step':
        # StepLR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,  # Adjust every 5 epochs
            gamma=0.8     # Learning rate decay factor
        )
    else:
        # Default to linear scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    
    return scheduler

def train_model_with_early_stop(model, train_loader, val_loader, optimizer, scheduler_type, 
                                epochs, device, patience=5):
    """Model training function with early stopping (modified: track best validation AUC)"""
    best_val_auc = 0.0  # Changed to track best AUC
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    total_steps = len(train_loader) * epochs
    
    # Get learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, scheduler_type, total_steps)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}", end=" ")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        
        # Validation
        val_results = eval_model(model, val_loader, device)
        val_loss = val_results['loss']
        val_auc = val_results['auc']

        print(f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val auc: {val_auc:.4f}")
        
        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered! Validation AUC hasn't improved for {patience} consecutive epochs, best epoch: {best_epoch + 1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    # Return best AUC and final validation results
    return model, best_val_auc, val_results

def run_param_search(train_df, val_df, test_df, external_test_df, tokenizer):
    """Execute parameter search (core modification: save model+test+record for each parameter set)"""
    # Generate parameter combinations
    param_names = list(PARAM_SEARCH_SPACE.keys())
    param_values = list(PARAM_SEARCH_SPACE.values())
    param_combinations = list(itertools.product(*param_values))

    best_params = None
    best_val_auc = 0
    best_model = None
    all_results = []

    print(f"\nStarting parameter search, total {len(param_combinations)} parameter combinations")
    print("-" * 80)

    for idx, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        lr = param_dict['lr']
        batch_size = param_dict['batch_size']
        weight_decay = param_dict['weight_decay']
        lr_scheduler = param_dict['lr_scheduler']

        print(f"\n[{idx+1}/{len(param_combinations)}] Testing parameters: {param_dict}")

        # Create DataLoader
        train_loader = create_data_loader(train_df, tokenizer, MAX_LEN, batch_size, shuffle=True)
        val_loader = create_data_loader(val_df, tokenizer, MAX_LEN, batch_size, shuffle=False)
        test_loader = create_data_loader(test_df, tokenizer, MAX_LEN, batch_size, shuffle=False)
        external_test_loader = create_data_loader(external_test_df, tokenizer, MAX_LEN, batch_size, shuffle=False)

        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            problem_type="single_label_classification"
        )
        model = model.to(DEVICE)
        # print("Model Architecture:\n", model)

        # Set optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Train model (with early stopping)
        model, best_val_auc_curr, val_results = train_model_with_early_stop(
            model, train_loader, val_loader, optimizer, lr_scheduler,
            EPOCHS, DEVICE, EARLY_STOP_PATIENCE
        )

        # 1. Save best model for current parameter combination
        param_save_dir = os.path.join(main_save_dir, f"param_{idx+1}")
        os.makedirs(param_save_dir, exist_ok=True)
        model_save_path = os.path.join(param_save_dir, "best_model.pt")
        torch.save({
            'params': param_dict,
            'model_state_dict': model.state_dict(),
            'best_val_auc': best_val_auc_curr,
            'val_results': val_results
        }, model_save_path)
        print(f"Best model for current parameters saved to: {model_save_path}")

        # 2. Evaluate test set and external test set with best model for current parameters
        print(f"Evaluating test set...")
        test_results = eval_model(model, test_loader, DEVICE)
        print(f"Evaluating external test set...")
        external_test_results = eval_model(model, external_test_loader, DEVICE)

        # 3. Record complete results for current parameters (validation+test+external test) - using lowercase metrics
        result_dict = {
            # Parameter information
            'param_idx': idx + 1,
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'lr_scheduler': lr_scheduler,
            'model_save_path': model_save_path,

            # Validation set metrics (lowercase)
            'val_auc': val_results['auc'],
            'val_acc': val_results['acc'],
            'val_tnr': val_results['tnr'],
            'val_tpr': val_results['tpr'],
            'val_precision': val_results['precision'],
            'val_f1': val_results['f1'],
            'val_loss': val_results['loss'],

            # Test set metrics (lowercase)
            'test_auc': test_results['auc'],
            'test_acc': test_results['acc'],
            'test_tnr': test_results['tnr'],
            'test_tpr': test_results['tpr'],
            'test_precision': test_results['precision'],
            'test_f1': test_results['f1'],
            'test_loss': test_results['loss'],

            # External test set metrics (lowercase)
            'external_test_auc': external_test_results['auc'],
            'external_test_acc': external_test_results['acc'],
            'external_test_tnr': external_test_results['tnr'],
            'external_test_tpr': external_test_results['tpr'],
            'external_test_precision': external_test_results['precision'],
            'external_test_f1': external_test_results['f1'],
            'external_test_loss': external_test_results['loss']
        }
        all_results.append(result_dict)

        # Print test results for current parameters
        print(f"\nTest set results for current parameters:")
        print(f"auc: {test_results['auc']:.4f} | acc: {test_results['acc']:.4f} | f1: {test_results['f1']:.4f}")
        print(f"\nExternal test set results for current parameters:")
        print(f"auc: {external_test_results['auc']:.4f} | acc: {external_test_results['acc']:.4f} | f1: {external_test_results['f1']:.4f}")

        # Update global best
        if best_val_auc_curr > best_val_auc:
            best_val_auc = best_val_auc_curr
            best_params = param_dict
            best_model = model
            print(f"Updated global best parameters! Current best Val auc: {best_val_auc:.4f}")
    
    # Save all parameter search results to CSV
    results_df = pd.DataFrame(all_results)
    results_save_path = os.path.join(main_save_dir, 'param_search_full_results.csv')
    results_df.to_csv(results_save_path, index=False, encoding='utf-8')
    print(f"\nAll parameter search results saved to: {results_save_path}")

    # Print global best results
    print(f"\nParameter search completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best validation auc: {best_val_auc:.4f}")

    # Re-evaluate test set and external test set with global best model and print detailed results
    best_test_loader = create_data_loader(test_df, tokenizer, MAX_LEN, best_params['batch_size'], shuffle=False)
    best_test_results = eval_model(best_model, best_test_loader, DEVICE)

    best_external_test_loader = create_data_loader(external_test_df, tokenizer, MAX_LEN, best_params['batch_size'], shuffle=False)
    best_external_test_results = eval_model(best_model, best_external_test_loader, DEVICE)

    print("\n" + "="*50)
    print("Global best model final test set performance")
    print("="*50)
    print(f"Parameters: {best_params}")
    print(f"auc: {best_test_results['auc']:.4f}")
    print(f"acc: {best_test_results['acc']:.4f}")
    print(f"tnr (Specificity): {best_test_results['tnr']:.4f}")
    print(f"tpr (Sensitivity): {best_test_results['tpr']:.4f}")
    print(f"precision (Precision): {best_test_results['precision']:.4f}")
    print(f"f1 Score: {best_test_results['f1']:.4f}")

    print("\n" + "="*50)
    print("Global best model final external test set performance")
    print("="*50)
    print(f"auc: {best_external_test_results['auc']:.4f}")
    print(f"acc: {best_external_test_results['acc']:.4f}")
    print(f"tnr (Specificity): {best_external_test_results['tnr']:.4f}")
    print(f"tpr (Sensitivity): {best_external_test_results['tpr']:.4f}")
    print(f"precision (Precision): {best_external_test_results['precision']:.4f}")
    print(f"f1 Score: {best_external_test_results['f1']:.4f}")

    # Save global best model
    best_model_save_path = os.path.join(main_save_dir, "global_best_model.pt")
    torch.save({
        'params': best_params,
        'model_state_dict': best_model.state_dict(),
        'test_results': best_test_results,
        'external_test_results': best_external_test_results
    }, best_model_save_path)
    print(f"\nGlobal best model saved to: {best_model_save_path}")

    return best_model, best_params, best_test_results, best_external_test_results

def clinicalbert_classification():
    """Complete workflow"""
    # 1. Use split_dataset function to load patient IDs for each dataset
    train_ids = split_dataset(fold_num=fold_num, split="train", pt_path=pt_path, sub_val_ratio=0.25, sub_val=True)
    val_ids = split_dataset(fold_num=fold_num, split="val", pt_path=pt_path, sub_val_ratio=0.25, sub_val=True)
    test_ids = split_dataset(fold_num=fold_num, split="val", pt_path=pt_path, sub_val=False)
    external_test_ids = split_dataset(fold_num=fold_num, split="test", pt_path=pt_path, sub_val=False)

    # Convert to strings
    train_ids = [str(pid) for pid in train_ids]
    val_ids = [str(pid) for pid in val_ids]
    test_ids = [str(pid) for pid in test_ids]
    external_test_ids = [str(pid) for pid in external_test_ids]

    # Check dataset overlap
    overlap_train_val = set(train_ids) & set(val_ids)
    overlap_train_test = set(train_ids) & set(test_ids)
    overlap_train_external = set(train_ids) & set(external_test_ids)
    overlap_val_test = set(val_ids) & set(test_ids)
    overlap_val_external = set(val_ids) & set(external_test_ids)
    overlap_test_external = set(test_ids) & set(external_test_ids)

    if overlap_train_val:
        warnings.warn(f"Training set and sub-validation set have {len(overlap_train_val)} duplicate IDs")
    if overlap_train_test:
        warnings.warn(f"Training set and test set have {len(overlap_train_test)} duplicate IDs")
    if overlap_train_external:
        warnings.warn(f"Training set and external test set have {len(overlap_train_external)} duplicate IDs")
    if overlap_val_test:
        warnings.warn(f"Sub-validation set and test set have {len(overlap_val_test)} duplicate IDs")
    if overlap_val_external:
        warnings.warn(f"Sub-validation set and external test set have {len(overlap_val_external)} duplicate IDs")
    if overlap_test_external:
        warnings.warn(f"Test set and external test set have {len(overlap_test_external)} duplicate IDs")

    # 2. Load Excel data
    df = pd.read_excel(file_path, engine='openpyxl')
    df["患者编号"] = df["患者编号"].astype(str)
    print(f"Original data has {len(df)} samples")

    # 4. Split datasets
    train_df = df[df["患者编号"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["患者编号"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["患者编号"].isin(test_ids)].reset_index(drop=True)
    external_test_df = df[df["患者编号"].isin(external_test_ids)].reset_index(drop=True)

    print(f"\nFinal dataset split:")
    print(f"Training set: {len(train_df)} samples")
    print(f"Sub-validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"External test set: {len(external_test_df)} samples")

    # 5. Data preprocessing
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

    # 6. Generate English text
    print("\nGenerating English clinical description text...")
    train_df['english_text'] = train_df.apply(generate_english_sentence, axis=1)
    val_df['english_text'] = val_df.apply(generate_english_sentence, axis=1)
    test_df['english_text'] = test_df.apply(generate_english_sentence, axis=1)
    external_test_df['english_text'] = external_test_df.apply(generate_english_sentence, axis=1)

    print(f"\nExample English text:")
    print(train_df['english_text'].iloc[0][:500] + "...")

    # 7. Load Tokenizer
    print("\nLoading ClinicalBERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 8. Execute parameter search and model training
    best_model, best_params, test_results, external_test_results = run_param_search(
        train_df, val_df, test_df, external_test_df, tokenizer
    )

    return best_model, best_params, test_results, external_test_results

'''
python code/step1_clin_var_clinicalbert.py --fold_num 1 --cuda_id 0 --modality Clin_ClinicalBert
'''
# ---------------------- Execute main workflow ----------------------
if __name__ == "__main__":
    trained_model, best_params, test_results, external_test_results = clinicalbert_classification()