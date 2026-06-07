'''
BUS and CDFI backbone experiments
Dataset split: 5-fold cross-validation for training set, external test
Best model selection: Early stopping based on sub-validation set loss, select model with highest AUC on sub-validation set
'''
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import sys
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models.video import swin3d_t, Swin3D_T_Weights, r3d_18, R3D_18_Weights
from transformers import AutoImageProcessor, TimesformerModel
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, f1_score, classification_report
)

# Packages
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from itertools import product
from datetime import datetime
import matplotlib.pyplot as plt

# Dataset
from utils.MultiModelDataset import MultiModelDataset

# Blocks
from utils.blood_fusion import HTCM
from utils.fusion_tokshift import TokShiftFusion
from utils.fusion_pta import PTAFusion
from utils.fusion_frames import TCStyleContextFusion # All tokens

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
pretrained_model_name = "./ckpt/pretrained_weights/dinov3-vitb16-pretrain-lvd1689m"


class Model3D(nn.Module):
    """
    Unified wrapper class for adapting Timesformer, Swin3D, and R(2+1)D video classification models
    Uniformly replace the classification head of all models with a two-layer linear + ReLU + Dropout structure, outputting only logits
    """
    def __init__(self, pretrained_model_name: str = "facebook/timesformer-base-finetuned-k600", 
                 num_classes: int = 2, dropout_prob: float = 0.2, model_name: str = "timesformer"):
        super().__init__()
        
        # Model name and base configuration
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        # Load different backbone models and replace classification heads
        if self.model_name == "timesformer":
            # Load Timesformer backbone (remove original classification head)
            self.backbone = TimesformerModel.from_pretrained(pretrained_model_name)
            self.hidden_size = self.backbone.config.hidden_size
            self.config = self.backbone.config
            
        elif self.model_name == "swin3d":
            # Load Swin3D-T and replace original head layer
            self.backbone = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
            # Remove original head layer, keep feature extraction part
            self.backbone.head = nn.Identity()  # Replace original Linear(768, 400) with identity mapping
            self.hidden_size = 768  # Feature dimension of Swin3D-T
            
        elif self.model_name == "r3d_18":
            # Load R(2+1)D-18 and replace original fc layer
            self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
            # Remove original fc layer, keep feature extraction part
            self.backbone.fc = nn.Identity()  # Replace original Linear(512, 400) with identity mapping
            self.hidden_size = 512  # Feature dimension of R(2+1)D-18
            
        else:
            raise ValueError(f"Unsupported model name: {model_name}, available options: timesformer/swin/r3d_18")
        
        # Define unified classifier structure (two linear layers + ReLU + Dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize classifier weights following best practices"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                # Initialize linear layer weights with normal distribution, bias to zero
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, 
        bus_images=None, 
        cdfi_images=None, 
        clinical_features=None, 
        original_features=None, 
        ceus_features=None
    ) -> torch.Tensor:
        """
        Forward propagation function, returns only logits
        Args:
            bus_images/cdfi_images/clinical_features/original_features: reserved parameters (compatible with original input)
            ceus_features: CEUS video features, different input formats for different models:
                - Timesformer: (B, 16, C, H, W) or (B, C, H, W) (single frame)
                - Swin3D/R2+1D: (B, 16, C, H, W)
        Returns:
            logits: tensor with shape [batch_size, num_classes]
        """
        # 1. Process input formats and feature extraction for different models
        if self.model_name == "timesformer":
            # Timesformer input format: (B, num_frames, C, H, W)
            outputs = self.backbone(
                pixel_values=ceus_features,
                return_dict=True
            )
            # Extract [CLS] token features (B, hidden_size)
            feature = outputs.last_hidden_state[:, 0, :]
            
        elif self.model_name == "swin3d":
            # Swin3D input format requirement: (B, C, D, H, W), need to adjust dimensions
            # ceus_features: (B, 16, C, H, W) -> (B, C, 16, H, W)
            ceus_input = ceus_features.permute(0, 2, 1, 3, 4)
            # Extract Swin3D features (B, 768)
            feature = self.backbone(ceus_input)
            
        elif self.model_name == "r3d_18":
            # R(2+1)D input format requirement: (B, C, D, H, W), need to adjust dimensions
            # ceus_features: (B, 16, C, H, W) -> (B, C, 16, H, W)
            ceus_input = ceus_features.permute(0, 2, 1, 3, 4)
            # Extract R(2+1)D features (B, 512)
            feature = self.backbone(ceus_input)
        
        logits = self.classifier(feature)  # (B, num_classes)
        
        return logits.squeeze(-1)


class DINOv3Classifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes=2, feature_dim=768,
                 fusion_way = 'Avg' # Avg, Max, Attention, HTCM
                 ):
        super(DINOv3Classifier, self).__init__()
        
        ## ------- Model -------
        # Backbone: ViT-B(dinov3)
        # self.backbone = AutoModel.from_pretrained(
        #     pretrained_model_name, 
        #     device_map="cuda:0", 
        # )
        # Swin-T(ImageNet)
        # self.backbone = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        # self.backbone.head = nn.Linear(self.backbone.head.in_features, feature_dim)
        # ResNet(ImageNet)
        # self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feature_dim)
        # ViT-B(ImageNet)
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads.head = nn.Linear(self.backbone.heads.head.in_features, feature_dim)
        # ViT-B(MAE)
        # self.backbone = timm.create_model(
        #     'vit_base_patch16_224.mae',
        #     pretrained=True,
        #     num_classes=0   # Remove original classifier, output features only
        # )

        # Norm
        self.fusion_way = fusion_way
        self.all_tokens = True
        self.bus_bn = nn.BatchNorm1d(feature_dim)
        self.cdfi_bn = nn.BatchNorm1d(feature_dim)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

        ## ------ Weights ------ ##
        self._initialize_weights()
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # Image processor(dinov3)
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        
        # Device
        self.device = next(self.backbone.parameters()).device
        self.classifier = self.classifier.to(self.device)

        ## ------ Blocks ------
        # projection
        self.sequence_processor = nn.Sequential(
            nn.Linear(768, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.LayerNorm(feature_dim)
        )
        # attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
        # Temporal transformer encoder
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=4,
                dim_feedforward=feature_dim * 2,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=2
        )
        # HTCM
        self.blood_encoder = HTCM(d_model=feature_dim, n_heads=4, n_layers=2, fusion_type='concat')
        self.tokshift = TokShiftFusion(dim=feature_dim, shift_ratio=0.25, agg="attn")
        self.PTA = PTAFusion()
        # self.TDSSideEncoder = TDSSideEncoderFusion()
        self.TCStyleContext = TCStyleContextFusion()

    def _initialize_weights(self):
        """Initialize weights of newly added layers"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)        
    def forward(self, bus_image, cdfi_image, clinical_features, original_features, x):
        # -------- BUS and CDFI ---------
        if self.fusion_way == 'BUS':
            bus_features = self.backbone(bus_image)
            # bus_features = self.bus_bn(bus_features['pooler_output'])
            logits = self.classifier(bus_features)
            return logits.squeeze(-1)
        if self.fusion_way == 'CDFI':
            cdfi_features = self.backbone(cdfi_image)
            # cdfi_features = self.cdfi_bn(cdfi_features['pooler_output'])
            logits = self.classifier(cdfi_features)
            return logits.squeeze(-1)
        ## -------- CEUS --------- ##
        # ----- extract features -----
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w).contiguous()
        spatial_features = self.backbone(pixel_values=x) # [batch_size * seq_len, N, hidden_size]

        if self.all_tokens:
            spatial_features = spatial_features['last_hidden_state'] # [batch_size * seq_len, 201, hidden_size]
            sequence_features = spatial_features.view(batch_size, seq_len, 201, -1) # [batch_size, seq_len, 201, hidden_size]
            cls_token = sequence_features[:, :, 0, :]
            sequence_features = sequence_features[:, :, 1:, :]
        else:
            spatial_features = spatial_features['pooler_output'] # [batch_size * seq_len, hidden_size]
            sequence_features = spatial_features.view(batch_size, seq_len, -1) # [batch_size, seq_len, hidden_size]
        # ------ fusion ------
        if self.fusion_way == 'Avg':
            pooled_features = sequence_features.mean(dim=1)
        elif self.fusion_way == 'Max':
            pooled_features = sequence_features.max(dim=1)[0]
        elif self.fusion_way == 'Attention':
            # temporal_features = self.temporal_encoder(sequence_features)     # [B, T, feature_dim]
            attn_scores = self.attention_pool(sequence_features)            # [B, T, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)                # [B, T, 1]
            pooled_features = torch.sum(sequence_features * attn_weights, dim=1)  # [B, feature_dim]
        elif self.fusion_way == 'HTCM':
            pooled_features = self.blood_encoder(sequence_features)
        elif self.fusion_way == 'Tokshift':
            pooled_features = self.tokshift(sequence_features)
        elif self.fusion_way == 'PTA':
            pooled_features = self.PTA(sequence_features)
        elif self.fusion_way == 'TCStyleContext':
            pooled_features = self.TCStyleContext(cls_token, sequence_features)

        # Predict
        logits = self.classifier(pooled_features)
        
        return logits.squeeze(-1)


def calculate_metrics(y_true, y_pred, y_scores):
    """Calculate various evaluation metrics (added Precision and F1-score)"""
    # For binary classification, directly use probability scores to calculate AUC
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
    acc = accuracy_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    precision = precision_score(y_true, y_pred, zero_division=0)  # Precision
    f1 = f1_score(y_true, y_pred, zero_division=0)  # F1 score
    
    return auc, acc, tpr, tnr, precision, f1

def plot_confusion_matrix(y_true, y_pred, epoch, phase, save_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Epoch {epoch} ({phase})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}_{phase}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_curves(train_losses, val_losses, save_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move data to device
        bus_images = batch['bus_image'].to(device)
        cdfi_images = batch['cdfi_image'].to(device)
        clinical_features = batch['clinical_features'].to(device)
        original_features = batch['feats'].to(device)
        labels = batch['label'].to(device).float()  # Convert to float type for BCELoss
        ceus_features = batch['ceus_sequence'].to(device)
        
        # Forward propagation
        optimizer.zero_grad()
        outputs = model(bus_images, cdfi_images, clinical_features, original_features, ceus_features)
        loss = criterion(outputs, labels)
        
        # Backward propagation
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predictions = (outputs > 0.5).float()  # Binary classification with threshold 0.5
        
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_scores.extend(outputs.cpu().detach().numpy())  # Use probability values directly

        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)
    
    avg_loss = running_loss / len(dataloader)
    auc, acc, tpr, tnr, precision, f1 = calculate_metrics(all_labels, all_predictions, all_scores)
    
    return avg_loss, auc, acc, tpr, tnr, precision, f1, all_labels, all_predictions

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")
        
        for batch in progress_bar:
            # Move data to device
            bus_images = batch['bus_image'].to(device)
            cdfi_images = batch['cdfi_image'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            original_features = batch['feats'].to(device)
            labels = batch['label'].to(device).float()  # Convert to float type for BCELoss
            ceus_features = batch['ceus_sequence'].to(device)
            
            # Forward propagation
            outputs = model(bus_images, cdfi_images, clinical_features, original_features, ceus_features)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()  # Binary classification with threshold 0.5
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(outputs.cpu().detach().numpy())  # Use probability values directly
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)
    
    avg_loss = running_loss / len(dataloader)
    auc, acc, tpr, tnr, precision, f1 = calculate_metrics(all_labels, all_predictions, all_scores)
    
    return avg_loss, auc, acc, tpr, tnr, precision, f1, all_labels, all_predictions

# Add a new test function to record detailed patient information
def evaluate_with_patient_info(model, dataloader, criterion, device, save_dir, mode='test'):
    """
    Unified evaluation function, used for train/validation/test/sub_val/external_test sets

    Args:
        model: model
        dataloader: data loader
        criterion: loss function
        device: device
        save_dir: save directory
        mode: mode, options: 'train', 'val', 'test', 'sub_val', 'external_test'

    Returns:
        avg_loss, auc, acc, tpr, tnr, precision, f1, all_labels, all_predictions
    """
    # Set description and filename based on mode
    mode_descriptions = {
        'train': 'Recording Train Set Patient Info',
        'val': 'Recording Validation Set Patient Info',
        'test': 'Recording Test Set Patient Info',
        'sub_val': 'Recording Sub-Validation Set Patient Info',
        'external_test': 'Recording External Test Set Patient Info'
    }

    file_names = {
        'train': 'detailed_train_results.csv',
        'val': 'detailed_val_results.csv',
        'test': 'detailed_test_results.csv',
        'sub_val': 'detailed_sub_val_results.csv',
        'external_test': 'detailed_external_test_results.csv'
    }

    description = mode_descriptions.get(mode, f'Evaluating ({mode})')
    file_name = file_names.get(mode, 'detailed_results.csv')

    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
    patient_ids = []
    is_correct = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=description)

        for batch in progress_bar:
            # Move data to device
            bus_images = batch['bus_image'].to(device)
            cdfi_images = batch['cdfi_image'].to(device)
            clinical_features = batch['clinical_features'].to(device)
            original_features = batch['feats'].to(device)
            ceus_features = batch['ceus_sequence'].to(device)
            labels = batch['label'].to(device).float()  # Convert to float type for BCELoss

            # Get patient IDs
            ids = batch['patient_id']

            # Forward propagation
            outputs = model(bus_images, cdfi_images, clinical_features, original_features, ceus_features)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()  # Binary classification with threshold 0.5

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(outputs.cpu().detach().numpy())  # Use probability values directly
            patient_ids.extend(ids)

            # Record if prediction is correct
            correct = (predictions.cpu().numpy() == labels.cpu().numpy())
            is_correct.extend(correct)

            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)

    avg_loss = running_loss / len(dataloader)
    auc, acc, tpr, tnr, precision, f1 = calculate_metrics(all_labels, all_predictions, all_scores)

    # Create DataFrame and save as CSV
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'prediction_probability': all_scores.flatten(),
        'true_label': all_labels,
        'predicted_label': all_predictions.flatten(),
        'is_correct': is_correct
    })

    # Save to CSV file
    results_df.to_csv(os.path.join(save_dir, file_name), index=False)

    return avg_loss, auc, acc, tpr, tnr, precision, f1, all_labels, all_predictions


# Keep compatibility, define alias
def test_with_patient_info(model, dataloader, criterion, device, save_dir):
    """Test model and record detailed patient information (compatibility interface)"""
    return evaluate_with_patient_info(model, dataloader, criterion, device, save_dir, mode='test')


def train_with_patient_info(model, dataloader, criterion, device, save_dir):
    """Predict on training set and record detailed patient information (compatibility interface)"""
    return evaluate_with_patient_info(model, dataloader, criterion, device, save_dir, mode='train')


def validate_with_patient_info(model, dataloader, criterion, device, save_dir):
    """Predict on validation set and record detailed patient information (compatibility interface)"""
    return evaluate_with_patient_info(model, dataloader, criterion, device, save_dir, mode='val')


def train_with_hyperparams(params, device, train_loader, sub_val_loader, test_loader, external_test_loader, save_dir, args):
    """Train model with specific hyperparameters, including train set, sub-validation set, test set and external test set"""
    # replace to 3D model
    #"CEUS_3D_R3D", "CEUS_3D_Swin3d", "CEUS_3D_TimeSformer"
    if args.modality == 'CEUS_3D_R3D':
        model = Model3D(pretrained_model_name="",
                        num_classes=1, model_name="r3d_18")  # timesformer r2plus1d_18 swin3d
    elif args.modality == 'CEUS_3D_Swin3d':
        model = Model3D(pretrained_model_name="",
                        num_classes=1, model_name="swin3d")
    elif args.modality == 'CEUS_3D_TimeSformer':
        model = Model3D(pretrained_model_name="./ckpt/pretrained_weights/models--facebook--timesformer-base-finetuned-k600",
                        num_classes=1, model_name="timesformer")
    elif args.modality == 'CEUS_2D_dinov3_v4_blood_fusion':
        model = DINOv3Classifier("./ckpt/pretrained_weights/dinov3-vitb16-pretrain-lvd1689m", num_classes=1)
    else:
        model = DINOv3Classifier("./ckpt/pretrained_weights/dinov3-vitb16-pretrain-lvd1689m", num_classes=1, fusion_way=args.fusion_way)

     
    # Modify for multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Separate parameter groups
    # clinical_params = list(model.clinical_projection.parameters())
    backbone_params = list(model.backbone.parameters()) if not isinstance(model, nn.DataParallel) else list(model.module.backbone.parameters())
    classifier_params = list(model.classifier.parameters()) if not isinstance(model, nn.DataParallel) else list(model.module.classifier.parameters())
    other_params = []
    named_parameters = model.named_parameters() if not isinstance(model, nn.DataParallel) else model.module.named_parameters()
    for name, param in named_parameters:
        if not any(name.startswith(prefix) for prefix in ['clinical_projection', 'backbone', 'classifier']):
            other_params.append(param)

    # Optimizer
    optimizer = optim.AdamW([
        # {'params': clinical_params, 'lr': params['lr_clinical'], 'weight_decay': params['weight_decay']},
        {'params': backbone_params, 'lr': params['lr_backbone'], 'weight_decay': params['weight_decay']},
        {'params': classifier_params, 'lr': params['lr_classifier'], 'weight_decay': params['weight_decay']},
        {'params': other_params, 'lr': params['lr_other'], 'weight_decay': params['weight_decay']}
    ])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=params['scheduler_step'], 
        gamma=params['scheduler_gamma']
    )
    
    # Training parameters
    num_epochs = params['num_epochs']
    # Set initial best validation metric value based on save_sign
    if args.save_sign == "AUC":
        best_val_metric = 0.0  # Best AUC is higher better
        metric_name = "AUC"
        is_better = lambda current, best: current > best + 1e-6  # Higher AUC is better
        patience = 5  # Early stopping patience
        patience_counter = 0  # Early stopping counter
    elif args.save_sign == "Loss":
        best_val_metric = float('inf')  # Best Loss is lower better
        metric_name = "Loss"
        is_better = lambda current, best: current < best - 1e-6  # Lower Loss is better
        patience = 5  # Early stopping patience
        patience_counter = 0  # Early stopping counter
    else:  # args.save_sign == "Loss_AUC"
        # Early stopping related (based on Loss)
        best_val_loss = float('inf')
        patience_loss = 5  # Early stopping patience
        patience_loss_counter = 0  # Early stopping counter
        # Save best model related (based on AUC)
        best_val_auc = 0.0
    
    # Record training history
    train_losses = []
    val_losses = []
    val_metrics = []  # Store AUC or Loss based on save_sign
    
    print(f"\n{'='*60}")
    print(f"Start training, hyperparameters: {params}")
    if args.save_sign == "Loss_AUC":
        print(f"Save mode: Loss_AUC (early stop based on Loss, save best AUC model)")
    else:
        print(f"Save metric: {args.save_sign}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # Training
        train_loss, train_auc, train_acc, train_tpr, train_tnr, train_precision, train_f1, train_labels, train_preds = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Sub-validation set
        sub_val_loss, sub_val_auc, sub_val_acc, sub_val_tpr, sub_val_tnr, sub_val_precision, sub_val_f1, sub_val_labels, sub_val_preds = validate_epoch(
            model, sub_val_loader, criterion, device
        )

        # Test set (for monitoring only)
        test_loss, test_auc, test_acc, test_tpr, test_tnr, test_precision, test_f1, test_labels, test_preds = validate_epoch(
            model, test_loader, criterion, device
        )

        # External test set (for monitoring only)
        external_test_loss, external_test_auc, external_test_acc, external_test_tpr, external_test_tnr, external_test_precision, external_test_f1, external_test_labels, external_test_preds = validate_epoch(
            model, external_test_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()

        # Record losses and metrics
        train_losses.append(train_loss)
        val_losses.append(sub_val_loss)
        if args.save_sign == "AUC":
            val_metrics.append(sub_val_auc)
        else:
            val_metrics.append(sub_val_loss)

        # Print results
        current_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Current learning rates: Backbone={current_lrs[0]:.2e}, Classifier={current_lrs[1]:.2e}, Other={current_lrs[2]:.2e}")
        print(f"Train     - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, ACC: {train_acc:.4f}, "
              f"TPR: {train_tpr:.4f}, TNR: {train_tnr:.4f}, Precision: {train_precision:.4f}, F1: {train_f1:.4f}")
        print(f"Sub_Val   - Loss: {sub_val_loss:.4f}, AUC: {sub_val_auc:.4f}, ACC: {sub_val_acc:.4f}, "
              f"TPR: {sub_val_tpr:.4f}, TNR: {sub_val_tnr:.4f}, Precision: {sub_val_precision:.4f}, F1: {sub_val_f1:.4f}")
        print(f"Test      - Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, "
              f"TPR: {test_tpr:.4f}, TNR: {test_tnr:.4f}, Precision: {test_precision:.4f}, F1: {test_f1:.4f}")
        print(f"External  - Loss: {external_test_loss:.4f}, AUC: {external_test_auc:.4f}, ACC: {external_test_acc:.4f}, "
              f"TPR: {external_test_tpr:.4f}, TNR: {external_test_tnr:.4f}, Precision: {external_test_precision:.4f}, F1: {external_test_f1:.4f}")
        
        # Plot sub-validation set confusion matrix
        plot_confusion_matrix(sub_val_labels, sub_val_preds, epoch+1, 'SubValidation', save_dir)

        # Plot loss curves
        plot_loss_curves(train_losses, val_losses, save_dir)

        # Early stopping logic (based on sub-validation set metrics)
        if args.save_sign == "AUC":
            current_val_metric = sub_val_auc
            if is_better(current_val_metric, best_val_metric):
                best_val_metric = current_val_metric
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'sub_val_auc': sub_val_auc,
                    'sub_val_acc': sub_val_acc,
                    'sub_val_loss': sub_val_loss,
                    'hyperparams': params
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved! Sub_Val {metric_name}: {current_val_metric:.4f}")
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered! Sub-validation set {metric_name} hasn't improved for {patience} epochs")
                    break
        elif args.save_sign == "Loss":
            current_val_metric = sub_val_loss
            if is_better(current_val_metric, best_val_metric):
                best_val_metric = current_val_metric
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'sub_val_auc': sub_val_auc,
                    'sub_val_acc': sub_val_acc,
                    'sub_val_loss': sub_val_loss,
                    'hyperparams': params
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved! Sub_Val {metric_name}: {current_val_metric:.4f}")
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered! Sub-validation set {metric_name} hasn't improved for {patience} epochs")
                    break
        else:  # args.save_sign == "Loss_AUC"
            # 1. Early stopping check (based on Loss)
            if sub_val_loss < best_val_loss - 1e-6:
                best_val_loss = sub_val_loss
                patience_loss_counter = 0
            else:
                patience_loss_counter += 1
                print(f"Early stopping counter: {patience_loss_counter}/{patience_loss}")

            # 2. Save best AUC model
            if sub_val_auc > best_val_auc + 1e-6:
                best_val_auc = sub_val_auc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'sub_val_auc': sub_val_auc,
                    'sub_val_acc': sub_val_acc,
                    'sub_val_loss': sub_val_loss,
                    'hyperparams': params
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved! Sub_Val AUC: {sub_val_auc:.4f}")

            # 3. Early stopping trigger (based on Loss)
            if patience_loss_counter >= patience_loss:
                print(f"Early stopping triggered! Sub-validation set Loss hasn't improved for {patience_loss} epochs")
                break
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=False)
    if not isinstance(model, nn.DataParallel):
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.module.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    print("\n" + "="*50)
    print("Evaluating best model...")

    # Training set evaluation
    train_loss_final, train_auc_final, train_acc_final, train_tpr_final, train_tnr_final, train_precision_final, train_f1_final, _, _ = evaluate_with_patient_info(
        model, train_loader, criterion, device, save_dir, mode='train'
    )

    # Sub-validation set evaluation
    sub_val_loss_final, sub_val_auc_final, sub_val_acc_final, sub_val_tpr_final, sub_val_tnr_final, sub_val_precision_final, sub_val_f1_final, _, _ = evaluate_with_patient_info(
        model, sub_val_loader, criterion, device, save_dir, mode='sub_val'
    )

    # Test set evaluation (on best sub-validation model)
    test_loss_final, test_auc_final, test_acc_final, test_tpr_final, test_tnr_final, test_precision_final, test_f1_final, test_labels_final, test_preds_final = evaluate_with_patient_info(
        model, test_loader, criterion, device, save_dir, mode='test'
    )

    # External test set evaluation (on best sub-validation model)
    external_test_loss_final, external_test_auc_final, external_test_acc_final, external_test_tpr_final, external_test_tnr_final, external_test_precision_final, external_test_f1_final, external_test_labels_final, external_test_preds_final = evaluate_with_patient_info(
        model, external_test_loader, criterion, device, save_dir, mode='external_test'
    )

    # Plot test set confusion matrix
    plot_confusion_matrix(test_labels_final, test_preds_final, 'Final', 'Test', save_dir)
    plot_confusion_matrix(external_test_labels_final, external_test_preds_final, 'Final', 'ExternalTest', save_dir)

    # Save final results
    with open(os.path.join(save_dir, 'final_results.txt'), 'w') as f:
        f.write(f"=== Hyperparameters ===\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Save metric: {args.save_sign}\n")
        f.write(f"\n=== Final Results Summary ===\n\n")

        f.write(f"Training Set Results:\n")
        f.write(f"Loss: {train_loss_final:.4f}\n")
        f.write(f"AUC: {train_auc_final:.4f}\n")
        f.write(f"ACC: {train_acc_final:.4f}\n")
        f.write(f"TPR: {train_tpr_final:.4f}\n")
        f.write(f"TNR: {train_tnr_final:.4f}\n")
        f.write(f"Precision: {train_precision_final:.4f}\n")
        f.write(f"F1-score: {train_f1_final:.4f}\n\n")

        f.write(f"Sub Validation Set Results:\n")
        f.write(f"Loss: {sub_val_loss_final:.4f}\n")
        f.write(f"AUC: {sub_val_auc_final:.4f}\n")
        f.write(f"ACC: {sub_val_acc_final:.4f}\n")
        f.write(f"TPR: {sub_val_tpr_final:.4f}\n")
        f.write(f"TNR: {sub_val_tnr_final:.4f}\n")
        f.write(f"Precision: {sub_val_precision_final:.4f}\n")
        f.write(f"F1-score: {sub_val_f1_final:.4f}\n\n")

        f.write(f"Test Set Results (Best Sub Val Model):\n")
        f.write(f"Loss: {test_loss_final:.4f}\n")
        f.write(f"AUC: {test_auc_final:.4f}\n")
        f.write(f"ACC: {test_acc_final:.4f}\n")
        f.write(f"TPR: {test_tpr_final:.4f}\n")
        f.write(f"TNR: {test_tnr_final:.4f}\n")
        f.write(f"Precision: {test_precision_final:.4f}\n")
        f.write(f"F1-score: {test_f1_final:.4f}\n\n")

        f.write(f"External Test Set Results (Best Sub Val Model):\n")
        f.write(f"Loss: {external_test_loss_final:.4f}\n")
        f.write(f"AUC: {external_test_auc_final:.4f}\n")
        f.write(f"ACC: {external_test_acc_final:.4f}\n")
        f.write(f"TPR: {external_test_tpr_final:.4f}\n")
        f.write(f"TNR: {external_test_tnr_final:.4f}\n")
        f.write(f"Precision: {external_test_precision_final:.4f}\n")
        f.write(f"F1-score: {external_test_f1_final:.4f}\n\n")

        f.write(f"Test Set Classification Report:\n")
        f.write(classification_report(test_labels_final, test_preds_final))

        f.write(f"\nExternal Test Set Classification Report:\n")
        f.write(classification_report(external_test_labels_final, external_test_preds_final))

    # Return key results
    results = {
        **params,  # Hyperparameters
        'train_auc': train_auc_final,
        'sub_val_auc': sub_val_auc_final,
        'test_auc': test_auc_final,
        'external_test_auc': external_test_auc_final,
        'train_acc': train_acc_final,
        'sub_val_acc': sub_val_acc_final,
        'test_acc': test_acc_final,
        'external_test_acc': external_test_acc_final,
        'train_tpr': train_tpr_final,
        'sub_val_tpr': sub_val_tpr_final,
        'test_tpr': test_tpr_final,
        'external_test_tpr': external_test_tpr_final,
        'train_tnr': train_tnr_final,
        'sub_val_tnr': sub_val_tnr_final,
        'test_tnr': test_tnr_final,
        'external_test_tnr': external_test_tnr_final,
        'train_precision': train_precision_final,
        'sub_val_precision': sub_val_precision_final,
        'test_precision': test_precision_final,
        'external_test_precision': external_test_precision_final,
        'train_f1': train_f1_final,
        'sub_val_f1': sub_val_f1_final,
        'test_f1': test_f1_final,
        'external_test_f1': external_test_f1_final,
        'best_epoch': checkpoint['epoch'],
        'save_dir': save_dir,
        'save_sign': args.save_sign  # Add save metric type
    }

    return results

def parse_int_list(arg):
    """
    Support 3 input formats:
      --batch_size 16
      --batch_size 8 16 32
      --batch_size 8,16,32
    """
    if isinstance(arg, list):
        # argparse nargs="+"
        vals = []
        for x in arg:
            vals.extend(x.split(','))
        return [int(v) for v in vals if v != ""]
    else:
        return [int(v) for v in str(arg).split(',')]

def build_argparser():
    parser = argparse.ArgumentParser(description="Hyperparameter search for MultiModalModel")
    parser.add_argument(
        "--modality",
        type=str,
        default="CEUS_3D_R3D",
        help="Modality name for logging/saving ()"
    )
    parser.add_argument(
        "--fusion_way",
        type=str,
        default="Avg",
        choices=["BUS", "CDFI",
                 "Avg", "Max", "Attention", 
                 "HTCM", "Tokshift", "PTA",
                 "TCStyleContext"],
        help="Modality name for logging/saving ()"
    )
    parser.add_argument(
        "--fold_num",
        type=int,
        default=1,
        help="Fold number (default: 1)"
    )
    parser.add_argument(
        "--cuda_id",
        type=int,
        default=0,
        help="CUDA device id, e.g., 0/1/2/3 (default: 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="16",
        nargs="+",  # Allow multiple values or comma
        help="Batch size search space. Examples: --batch_size 16 | --batch_size 8 16 32 | --batch_size 8,16,32"
    )
    # Add multi-GPU support argument
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Enable multi-GPU training"
    )
    parser.add_argument(
        "--save_sign",
        type=str,
        default="Loss",
        choices=["AUC", "Loss", "Loss_AUC"],
        help="Save the best model based on validation AUC or Loss (default: Loss). Loss_AUC: early stop by loss, but save best AUC model"
    )
    return parser

def main():
    args = build_argparser().parse_args()

    modality = args.modality
    fold_num = args.fold_num
    cuda_id = args.cuda_id
    batch_size_list = parse_int_list(args.batch_size)

    # Set device
    if torch.cuda.is_available():
        if args.multi_gpu:
            # Use all available GPUs
            device = torch.device("cuda:0")  # Main GPU
            print(f"Using multiple GPUs: {torch.cuda.device_count()}")
        else:
            device = torch.device(f"cuda:{cuda_id}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create main save directory
    main_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_save_dir = f"./logs/{modality}/Fold{fold_num}/param_search_exp_{main_timestamp}"
    os.makedirs(main_save_dir, exist_ok=True)

    # Data path configuration
    pt_data_dict = {
        'train': f'./datasets/pt_file/train_fold{fold_num}.pt',
        'val': f'./datasets/pt_file/internal_test_fold{fold_num}.pt',
        'test': f'./datasets/pt_file/external_test_fold{fold_num}.pt'
    }

    # pixel_diff
    # "./datasets/CEUS"
    # random
    # "./datasets/CEUS_select_frames/1_Crop_same_shape_avi_random_16frames"
    # uniform
    # "./datasets/CEUS_select_frames/1_Crop_same_shape_avi_uniform_16frames"
    image_dirs = {
        'us': "./datasets/BUS",
        'cdfi': "./datasets/CDFI",
        'ceus': "./datasets/CEUS"
    }
    clinical_data_path = "./datasets/Clin/MVI-HCC-samples.xlsx"

    # Create datasets (note: don't fix batch_size here)
    # Use MultiModalDataset, supporting four datasets:
    # - train: training set (original training set)
    # - val with sub_val=True: sub-validation set (split from training set)
    # - test: test set
    # - external_test: external test set
    print("Loading datasets...")
    use_pil = False
    sub_val_ratio = 0.25  # Split 25% from training set as sub-validation set

    # Training set
    train_dataset = MultiModelDataset(pt_data_dict, image_dirs, clinical_data_path,
                                        split='train', use_pil=use_pil, sub_val=True, raw_ceus=True, fold_num=fold_num)

    # Sub-validation set (split from training set)
    sub_val_dataset = MultiModelDataset(pt_data_dict, image_dirs, clinical_data_path,
                                          split='val', use_pil=use_pil, sub_val=True, raw_ceus=True, fold_num=fold_num)

    # Test set
    test_dataset = MultiModelDataset(pt_data_dict, image_dirs, clinical_data_path,
                                       split='test', use_pil=use_pil, raw_ceus=True, fold_num=fold_num)

    # External test set
    external_test_dataset = MultiModelDataset(pt_data_dict, image_dirs, clinical_data_path,
                                                split='external_test', use_pil=use_pil, raw_ceus=True, fold_num=fold_num)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Sub Validation samples: {len(sub_val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"External Test samples: {len(external_test_dataset)}")

    # Define hyperparameter search space
    hyperparam_space = {
        # 'lr_clinical': [1e-3],
        'lr_backbone': [1e-6, 2e-6, 3e-6],
        'lr_classifier': [1e-4],
        'lr_other': [2e-6],
        'weight_decay': [1e-3, 1e-4],
        'scheduler_step': [10],
        'scheduler_gamma': [0.9],
        'num_epochs': [100],
        'batch_size': batch_size_list,
    }

    # Generate all hyperparameter combinations (grid search)
    param_names = list(hyperparam_space.keys())
    param_values = list(hyperparam_space.values())
    all_param_combinations = list(product(*param_values))

    print(f"\nStarting hyperparameter search, total {len(all_param_combinations)} parameter combinations")
    print(f"Main save directory: {main_save_dir}")
    print(f"batch_size search space: {batch_size_list}")

    all_results = []
    best_overall_auc = 0.0
    best_exp_dir = ""

    # Iterate through all hyperparameter combinations
    for idx, params_tuple in enumerate(all_param_combinations):
        params = dict(zip(param_names, params_tuple))

        # Key: for each params combination, rebuild loader based on batch_size
        bs = params['batch_size']
        # Adjust batch size based on multi-GPU usage
        if args.multi_gpu:
            bs = bs * torch.cuda.device_count()  # Increase batch size to fully utilize multi-GPU
            print(f"Multi-GPU enabled. Adjusted batch size to: {bs}")
        
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8)
        sub_val_loader = DataLoader(sub_val_dataset, batch_size=bs, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)
        external_test_loader = DataLoader(external_test_dataset, batch_size=bs, shuffle=False, num_workers=8)

        # Create current experiment directory
        exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_save_dir = os.path.join(main_save_dir, f"exp_{idx+1}_{exp_timestamp}")
        os.makedirs(exp_save_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Hyperparameter combination {idx+1}/{len(all_param_combinations)}")
        print(f"Experiment directory: {exp_save_dir}")
        print(f"Current batch_size: {bs}")
        print(f"{'='*80}")

        results = train_with_hyperparams(
            params, device, train_loader, sub_val_loader, test_loader, external_test_loader, exp_save_dir, args
        )
        all_results.append(results)

        if results['sub_val_auc'] > best_overall_auc:
            best_overall_auc = results['sub_val_auc']
            best_exp_dir = exp_save_dir

        print(f"\nCurrent combination completed! Sub-validation set AUC: {results['sub_val_auc']:.4f}")
        print(f"Current best sub-validation set AUC: {best_overall_auc:.4f} (Experiment directory: {best_exp_dir})")

    # Save all hyperparameters and results to CSV
    results_df = pd.DataFrame(all_results)
    csv_save_path = os.path.join(main_save_dir, 'hyperparam_search_results.csv')
    results_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*80}")
    print(f"Hyperparameter search completed!")
    print(f"All results saved to: {csv_save_path}")
    print(f"Best experiment: {best_exp_dir}")
    print(f"Best sub-validation set AUC: {best_overall_auc:.4f}")
    print(f"{'='*80}")

    # Print best results summary
    best_results = results_df.loc[results_df['sub_val_auc'].idxmax()]
    print(f"\nBest results summary:")
    print(f"Hyperparameters:")
    for param_name in param_names:
        print(f"  {param_name}: {best_results[param_name]}")
    print(f"\nPerformance metrics:")
    print(f"  Training set AUC: {best_results['train_auc']:.4f}")
    print(f"  Sub-validation set AUC: {best_results['sub_val_auc']:.4f}")
    print(f"  Test set AUC: {best_results['test_auc']:.4f}")
    print(f"  External test set AUC: {best_results['external_test_auc']:.4f}")
    print(f"  Test set ACC: {best_results['test_acc']:.4f}")
    print(f"  Test set TPR: {best_results['test_tpr']:.4f}")
    print(f"  Test set TNR: {best_results['test_tnr']:.4f}")
    print(f"  Test set Precision: {best_results['test_precision']:.4f}")
    print(f"  Test set F1-score: {best_results['test_f1']:.4f}")
    print(f"  External test set ACC: {best_results['external_test_acc']:.4f}")
    print(f"  External test set TPR: {best_results['external_test_tpr']:.4f}")
    print(f"  External test set TNR: {best_results['external_test_tnr']:.4f}")
    print(f"  External test set Precision: {best_results['external_test_precision']:.4f}")
    print(f"  External test set F1-score: {best_results['external_test_f1']:.4f}")


'''
python code/step2_BUS_CDFI.py --fold_num 1 --modality CDFI_ViT_B_MAE_v2 --batch_size 16 --save_sign Loss_AUC --fusion_way CDFI
'''
if __name__ == "__main__":
    main()