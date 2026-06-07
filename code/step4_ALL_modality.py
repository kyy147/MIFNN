'''
Modality Combination Experiment
Dataset Splitting: 5-fold cross-validation for training set, external test
Best Model Selection: Early stopping based on sub-validation set loss, select model with highest AUC on sub-validation set
'''
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModel
from torch.utils.data import DataLoader

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
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt

# Dataset
from utils.MultiModelDataset import MultiModelDataset

# Blocks
from utils.blood_fusion import HTCM

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
clinical_bert_model_name = "emilyalsentzer/Bio_ClinicalBERT"


class DINOv3Classifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes=1, feature_dim=768,
                fusion_way = 'HTCM', # Avg, Max, Attention, HTCM
                fusion_mode = "ALL",
                use_contrastive_loss=False,
                pretrained_clinical_path=None,  # Path to pretrained clinical model
                pretrained_ceus_path=None,  # Path to pretrained CEUS model (step3)
                num_transformer_layers=2,
                num_heads=4,
                 ):
        super(DINOv3Classifier, self).__init__()

        ## ------- Model ------- ##
        # Visual Backbone
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name,
            device_map="cpu",
        )
        self.static_backbone = AutoModel.from_pretrained(
            pretrained_model_name,
            device_map="cpu",
        )
        # self.static_backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        # self.static_backbone.heads.head = nn.Linear(self.static_backbone.heads.head.in_features, feature_dim)
        self.imagenet = False

        # clinical_backbone (using AutoModel)
        self.clinical_backbone = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            device_map="cpu"
        )

        # Load pretrained weights and freeze clinical_backbone
        self._setup_clinical_backbone(pretrained_clinical_path)
        # Norm
        self.fusion_way = fusion_way
        self.fusion_mode =  fusion_mode
        self.use_contrastive_loss = use_contrastive_loss
        self.all_tokens = False
        self.bus_bn = nn.BatchNorm1d(feature_dim)
        self.cdfi_bn = nn.BatchNorm1d(feature_dim)
        self.cdfi_score = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

        # Classifier
        classifier_input_dim = feature_dim * 2 if fusion_mode in ['ALL', 'BUS_CDFI'] else feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

        ## ------ Weights ------ ##
        self._initialize_weights()

        # Device
        self.device = next(self.backbone.parameters()).device
        self.classifier = self.classifier.to(self.device)

        ## ------ Blocks ------
        # attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
        # HTCM
        self.blood_encoder = HTCM(d_model=feature_dim, n_heads=4, n_layers=2, fusion_type='weighted_sum')

        ## ------ [CLS] token and Fusion ------
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # Learnable modality embeddings (BUS, CDFI, CEUS)
        self.modality_embeddings = nn.Parameter(torch.randn(3, 1, feature_dim))
        nn.init.normal_(self.modality_embeddings, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Load CEUS pretrained weights (backbone + blood_encoder)
        self._setup_ceus_backbone(pretrained_ceus_path)

    def _setup_clinical_backbone(self, pretrained_clinical_path):
        """
        Load pretrained weights into clinical_backbone, then freeze parameters
        Use AutoModel as feature extractor (without classification head)

        Args:
            pretrained_clinical_path: Path to step1 pretrained Clinical model
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if pretrained_clinical_path is not None:
            print(f"\nLoading Clinical pretrained weights: {pretrained_clinical_path}")
            checkpoint = torch.load(pretrained_clinical_path, map_location=device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            else:
                pretrained_dict = checkpoint

            # Get clinical_backbone state_dict
            clinical_state_dict = self.clinical_backbone.state_dict()

            loaded_count = 0
            skipped_keys = []

            for k, v in pretrained_dict.items():
                # Try multiple key name mappings
                target_key = None

                # Direct match clinical_backbone.xxx -> bert.xxx
                if k.startswith('clinical_backbone.'):
                    # Remove 'clinical_backbone.' prefix to get bert internal key name
                    target_key = k[len('clinical_backbone.'):]
                elif k.startswith('bert.'):
                    target_key = k[len('bert.'):]
                elif not any(k.startswith(prefix) for prefix in ['classifier', 'fc', 'head', 'dropout', 'pooler']):
                    # Try using directly (assuming pretrained weights are already bert weights without prefix)
                    target_key = k

                # Skip classification head related parameters
                if target_key and any(skip in target_key for skip in ['classifier', 'fc', 'head', 'dropout']):
                    continue

                if target_key is not None and target_key in clinical_state_dict:
                    if clinical_state_dict[target_key].shape == v.shape:
                        clinical_state_dict[target_key] = v
                        loaded_count += 1
                    else:
                        skipped_keys.append(f"{k} -> {target_key} (shape mismatch)")
                else:
                    if target_key is not None:
                        skipped_keys.append(f"{k} -> {target_key} (not found)")

            # Load updated state_dict
            self.clinical_backbone.load_state_dict(clinical_state_dict)
            print(f"Clinical weights loading completed: {loaded_count} parameters")

            if skipped_keys:
                print(f"Skipped Clinical parameters ({len(skipped_keys)}):")
                for key in skipped_keys[:5]:
                    print(f"  - {key}")
                if len(skipped_keys) > 5:
                    print(f"  ... and {len(skipped_keys) - 5} more parameters were skipped")
        else:
            print("No pretrained_clinical_path provided, clinical_backbone uses pretrained weights without fine-tuned weights")

        # Freeze all parameters of clinical_backbone
        for param in self.clinical_backbone.parameters():
            param.requires_grad = False
        print("clinical_backbone parameters have been frozen, used only as feature extractor")

    def _initialize_weights(self):
        """Initialize weights of newly added layers"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _setup_ceus_backbone(self, pretrained_ceus_path):
        """
        Load pretrained weights into backbone and blood_encoder (HTCM)
        These weights come from step3 pretrained CEUS model

        Args:
            pretrained_ceus_path: Path to step3 pretrained CEUS model
        """
        if pretrained_ceus_path is None:
            print("No pretrained_ceus_path provided, backbone and blood_encoder use default initialized weights")
            return

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading CEUS pretrained weights: {pretrained_ceus_path}")

        checkpoint = torch.load(pretrained_ceus_path, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        else:
            pretrained_dict = checkpoint

        # Get current model state_dict
        model_dict = self.state_dict()

        # Define module prefixes to load
        ceus_keys_to_load = ['backbone.', 'blood_encoder.']

        loaded_count = 0
        skipped_keys = []

        for k, v in pretrained_dict.items():
            if any(k.startswith(prefix) for prefix in ceus_keys_to_load):
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded_count += 1
                else:
                    skipped_keys.append(k)

        # Load updated state_dict
        self.load_state_dict(model_dict)

        print(f"CEUS weights loading completed: {loaded_count} parameters")
        print(f"Loaded modules: backbone, blood_encoder (HTCM)")

        if skipped_keys:
            print(f"Skipped CEUS parameters ({len(skipped_keys)}):")
            for key in skipped_keys[:5]:
                print(f"  - {key}")
            if len(skipped_keys) > 5:
                print(f"  ... and {len(skipped_keys) - 5} more parameters were skipped")
    
    def info_nce_loss(self, dynamic_feat, static_feat, temperature=0.07):
        """Symmetric InfoNCE Loss"""
        dynamic_feat = F.normalize(dynamic_feat, dim=1)
        static_feat = F.normalize(static_feat, dim=1)

        logits = torch.matmul(dynamic_feat, static_feat.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits.T, labels)
        contrastive_loss = (loss1 + loss2) / 2.0
        # Ensure contrastive_loss is a scalar
        if isinstance(contrastive_loss, torch.Tensor) and contrastive_loss.numel() > 1:
            contrastive_loss = contrastive_loss.mean()

        return contrastive_loss

    def forward(self, bus_image, cdfi_image, input_ids, attention_mask, x):
        ## -------- BUS CDFI --------- ##
        bus_features = self.static_backbone(bus_image)
        cdfi_features = self.static_backbone(cdfi_image)
        if not self.imagenet:
            bus_features = self.bus_bn(bus_features['pooler_output'])
            cdfi_features = self.cdfi_bn(cdfi_features['pooler_output'])
        bus_features = self.cdfi_score(cdfi_features) * bus_features
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
        # ------ 2d fusion ------
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
            dynamic_feat = self.blood_encoder(sequence_features)

        ## ------ Clinical BERT ------ ##
        # Process input_ids and attention_mask using clinical_backbone (Bio_ClinicalBERT)
        clinical_outputs = self.clinical_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        clinical_features = clinical_outputs.last_hidden_state[:, 0, :]  # [B, 768]

        ## ------ Contrasitive Loss ------ ##
        if self.use_contrastive_loss:
            # α, β = 0.5, 0.5
            bus_contrastive_loss = self.info_nce_loss(dynamic_feat, bus_features)
            cdfi_contrastive_loss = self.info_nce_loss(dynamic_feat, cdfi_features)
            contrastive_loss = bus_contrastive_loss + cdfi_contrastive_loss
        else:
            # Create a zero tensor as fake contrastive loss, doesn't affect optimization
            contrastive_loss = torch.tensor(0.0, device=dynamic_feat.device, requires_grad=False)

        ## ------ Modality Fusion ------ ##
        B = bus_features.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)                  # [B, 1, 768]

        # Get modality embeddings (BUS, CDFI, CEUS, Clinical)
        # self.modality_embeddings: [4, 1, 768] -> expand to [B, 4, 768]
        mod_embeds = self.modality_embeddings.expand(-1, B, -1).transpose(0, 1)  # [B, 4, 768]

        # Add corresponding modality embeddings to each modality feature
        bus_feat_encoded = bus_features + mod_embeds[:, 0, :]           # [B, 768]
        cdfi_feat_encoded = cdfi_features + mod_embeds[:, 1, :]        # [B, 768]
        ceus_feat_encoded = dynamic_feat + mod_embeds[:, 2, :]          # [B, 768]
        # clinical_feat_encoded = clinical_features + mod_embeds[:, 3, :]  # [B, 768]

        ## Concatenate all modality features
        ## ------ BUS_CDFI ------ ##
        if self.fusion_mode == 'BUS_CDFI':
            fused_repr = torch.cat([bus_features, cdfi_features], dim=-1)
            logits = self.classifier(fused_repr)
            return logits.squeeze(-1), contrastive_loss
        ## ------ BUS_CEUS ------ ##
        elif self.fusion_mode == 'BUS_CEUS':
            seq = torch.cat([cls_tokens, 
                            bus_feat_encoded.unsqueeze(1), 
                            ceus_feat_encoded.unsqueeze(1)], dim=1)  # [B, 4, 768]
            transformer_out = self.transformer(seq)                        # [B, 4, 768]
            fused_repr = transformer_out[:, 0, :]                          # [B, 768]
            logits = self.classifier(fused_repr)
            return logits.squeeze(-1), contrastive_loss
        ## ------ CDFI_CEUS ------ ##
        elif self.fusion_mode == 'CDFI_CEUS':
            seq = torch.cat([cls_tokens, 
                            cdfi_feat_encoded.unsqueeze(1),
                            ceus_feat_encoded.unsqueeze(1)], dim=1)  # [B, 4, 768]
            transformer_out = self.transformer(seq)                        # [B, 4, 768]
            fused_repr = transformer_out[:, 0, :]                          # [B, 768]
            logits = self.classifier(fused_repr)
            return logits.squeeze(-1), contrastive_loss
        ## ------ BUS_CDFI_CEUS ------ ##
        elif self.fusion_mode == 'BUS_CDFI_CEUS':
            seq = torch.cat([cls_tokens, 
                            bus_feat_encoded.unsqueeze(1), 
                            cdfi_feat_encoded.unsqueeze(1),
                            ceus_feat_encoded.unsqueeze(1)], dim=1)  # [B, 4, 768]
            transformer_out = self.transformer(seq)                        # [B, 4, 768]
            fused_repr = transformer_out[:, 0, :]                          # [B, 768]
            logits = self.classifier(fused_repr)
            return logits.squeeze(-1), contrastive_loss
        ## ------ ALL ------ ##
        elif self.fusion_mode == 'ALL':
            seq = torch.cat([cls_tokens, 
                            bus_feat_encoded.unsqueeze(1), 
                            cdfi_feat_encoded.unsqueeze(1),
                            ceus_feat_encoded.unsqueeze(1)], dim=1)  # [B, 4, 768]
            transformer_out = self.transformer(seq)                        # [B, 4, 768]
            fused_repr = transformer_out[:, 0, :]                          # [B, 768]
            fused_repr = torch.cat([fused_repr, clinical_features], dim=1)  # [B, 1536]
            logits = self.classifier(fused_repr)
            return logits.squeeze(-1), contrastive_loss
        # seq = torch.cat([cls_tokens, 
        #                  bus_feat_encoded.unsqueeze(1), 
        #                  cdfi_feat_encoded.unsqueeze(1),
        #                  ceus_feat_encoded.unsqueeze(1)], dim=1)  # [B, 4, 768]
        # transformer_out = self.transformer(seq)                        # [B, 4, 768]
        # fused_repr = transformer_out[:, 0, :]                          # [B, 768]

        # # Concatenate transformer output and clinical_features (for classification)
        # fused_repr = torch.cat([fused_repr, clinical_features], dim=1)  # [B, 1536]

        # ## ------ Prediction ------ ##
        # logits = self.classifier(fused_repr)
        # return logits.squeeze(-1), contrastive_loss


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

def plot_contrastive_loss_curves(train_contrastive_losses, val_contrastive_losses, save_dir):
    """Plot training and validation contrastive loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_contrastive_losses) + 1)

    plt.plot(epochs, train_contrastive_losses, 'b-', label='Training Contrastive Loss', linewidth=2)
    plt.plot(epochs, val_contrastive_losses, 'r-', label='Validation Contrastive Loss', linewidth=2)

    plt.title('Training and Validation Contrastive Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Contrastive Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_dir, 'contrastive_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_epoch(model, dataloader, criterion, optimizer, device, contrastive_weight=0.1):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0  # Classification loss
    running_contrastive_loss = 0.0  # Contrastive loss
    all_labels = []
    all_predictions = []
    all_scores = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move data to device
        bus_images = batch['bus_image'].to(device)
        cdfi_images = batch['cdfi_image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).float()  # Convert to float type for BCELoss
        ceus_features = batch['ceus_sequence'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs, contrastive_loss = model(bus_images, cdfi_images, input_ids, attention_mask, ceus_features)
        if isinstance(contrastive_loss, torch.Tensor) and contrastive_loss.numel() > 1:
            contrastive_loss = contrastive_loss.mean()

        # Calculate total loss = classification loss + contrastive loss weight * contrastive loss
        ce_loss = criterion(outputs, labels)
        loss = ce_loss + contrastive_weight * contrastive_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0
        predictions = (outputs > 0.5).float()  # Binary classification with threshold 0.5

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_scores.extend(outputs.cpu().detach().numpy())  # Directly use probability values

        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'CE': f'{ce_loss.item():.4f}',
            'Contrast': f'{contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0:.4f}'
        })

    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)

    avg_loss = running_loss / len(dataloader)
    avg_ce_loss = running_ce_loss / len(dataloader)
    avg_contrastive_loss = running_contrastive_loss / len(dataloader)
    auc, acc, tpr, tnr, precision, f1 = calculate_metrics(all_labels, all_predictions, all_scores)

    return avg_loss, avg_ce_loss, avg_contrastive_loss, auc, acc, tpr, tnr, precision, f1, all_labels, all_predictions

def validate_epoch(model, dataloader, criterion, device, contrastive_weight=0.1):
    """Validate one epoch"""
    model.eval()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_contrastive_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating")

        for batch in progress_bar:
            # Move data to device
            bus_images = batch['bus_image'].to(device)
            cdfi_images = batch['cdfi_image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()  # Convert to float type for BCELoss
            ceus_features = batch['ceus_sequence'].to(device)

            # Forward pass
            outputs, contrastive_loss = model(bus_images, cdfi_images, input_ids, attention_mask, ceus_features)
            if isinstance(contrastive_loss, torch.Tensor) and contrastive_loss.numel() > 1:
                contrastive_loss = contrastive_loss.mean()
            ce_loss = criterion(outputs, labels)
            loss = ce_loss + contrastive_weight * contrastive_loss

            # Statistics
            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0
            predictions = (outputs > 0.5).float()  # Binary classification with threshold 0.5

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(outputs.cpu().detach().numpy())  # Directly use probability values

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'Contrast': f'{contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0:.4f}'
            })

    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)

    avg_loss = running_loss / len(dataloader)
    avg_ce_loss = running_ce_loss / len(dataloader)
    avg_contrastive_loss = running_contrastive_loss / len(dataloader)
    auc, acc, tpr, tnr, precision, f1 = calculate_metrics(all_labels, all_predictions, all_scores)

    return avg_loss, avg_ce_loss, avg_contrastive_loss, auc, acc, tpr, tnr, precision, f1, all_labels, all_predictions

# Add a new test function to record detailed patient information
def evaluate_with_patient_info(model, dataloader, criterion, device, save_dir, mode='test', contrastive_weight=0.1):
    """
    Unified evaluation function, used for train/val/test/sub_val/external_test sets

    Args:
        model: model
        dataloader: data loader
        criterion: loss function
        device: device
        save_dir: save directory
        mode: mode, options: 'train', 'val', 'test', 'sub_val', 'external_test'
        contrastive_weight: contrastive loss weight

    Returns:
        avg_loss, avg_ce_loss, avg_contrastive_loss, auc, acc, tpr, tnr, precision, f1, all_labels, all_predictions
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
    running_ce_loss = 0.0
    running_contrastive_loss = 0.0
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ceus_features = batch['ceus_sequence'].to(device)
            labels = batch['label'].to(device).float()  # Convert to float type for BCELoss

            # Get patient IDs
            ids = batch['patient_id']

            # Forward pass
            outputs, contrastive_loss = model(bus_images, cdfi_images, input_ids, attention_mask, ceus_features)
            if isinstance(contrastive_loss, torch.Tensor) and contrastive_loss.numel() > 1:
                contrastive_loss = contrastive_loss.mean()
            ce_loss = criterion(outputs, labels)
            loss = ce_loss + contrastive_weight * contrastive_loss

            # Statistics
            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0
            predictions = (outputs > 0.5).float()  # Binary classification with threshold 0.5

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(outputs.cpu().detach().numpy())  # Directly use probability values
            patient_ids.extend(ids)

            # Record if prediction is correct
            correct = (predictions.cpu().numpy() == labels.cpu().numpy())
            is_correct.extend(correct)

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'CE': f'{ce_loss.item():.4f}',
                'Contrast': f'{contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else 0.0:.4f}'
            })

    # Calculate metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_scores = np.array(all_scores)

    avg_loss = running_loss / len(dataloader)
    avg_ce_loss = running_ce_loss / len(dataloader)
    avg_contrastive_loss = running_contrastive_loss / len(dataloader)
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

    return avg_loss, avg_ce_loss, avg_contrastive_loss, auc, acc, tpr, tnr, precision, f1, all_labels, all_predictions


# Keep compatibility, define aliases
def test_with_patient_info(model, dataloader, criterion, device, save_dir):
    """Test model and record detailed patient information (compatibility interface)"""
    return evaluate_with_patient_info(model, dataloader, criterion, device, save_dir, mode='test')


def train_with_patient_info(model, dataloader, criterion, device, save_dir):
    """Train set prediction and record detailed patient information (compatibility interface)"""
    return evaluate_with_patient_info(model, dataloader, criterion, device, save_dir, mode='train')


def validate_with_patient_info(model, dataloader, criterion, device, save_dir):
    """Validation set prediction and record detailed patient information (compatibility interface)"""
    return evaluate_with_patient_info(model, dataloader, criterion, device, save_dir, mode='val')


def train_with_hyperparams(params, device, train_loader, sub_val_loader, test_loader, external_test_loader, save_dir, args):
    """Train model with specific hyperparameters, including train, sub-validation, test, and external test sets"""
    # Pretrained weights
    pretrained_ceus_path_dict = {
        "1":"ckpt/network_weights/DCE-US_HTCM_best_model_fold1.pth",
        "2":"xxx",
        "3":"xxx",
        "4":"xxx",
        "5":"xxx"
    }
    pretrained_clinical_path_dict = {
        "1": "ckpt/network_weights/ClinicalBERT_best_model_fold1.pt",
        "2": "xxx",
        "3": "xxx",
        "4": "xxx",
        "5": "xxx"
    }
    pretrained_ceus_path = pretrained_ceus_path_dict[str(args.fold_num)] if getattr(args, 'pretrained_ceus', None) is None else args.pretrained_ceus
    pretrained_clinical_path = pretrained_clinical_path_dict[str(args.fold_num)] if getattr(args, 'pretrained_clinical', None) is None else args.pretrained_clinical
    # Create Model
    if args.modality == 'CEUS_2D_dinov3_v4_blood_fusion':
        model = DINOv3Classifier(
            "./ckpt/pretrained_weights/dinov3-vitb16-pretrain-lvd1689m",
            num_classes=1,
            pretrained_clinical_path=pretrained_clinical_path,  # Load clinical weights during initialization
            pretrained_ceus_path=pretrained_ceus_path  # Load CEUS weights during initialization
        )
    else:
        model = DINOv3Classifier(
            "./ckpt/pretrained_weights/dinov3-vitb16-pretrain-lvd1689m",
            num_classes=1,
            fusion_way=args.fusion_way,
            pretrained_clinical_path=pretrained_clinical_path,  # Load clinical weights during initialization
            pretrained_ceus_path=pretrained_ceus_path,  # Load CEUS weights during initialization
            fusion_mode=args.fusion_mode,
            use_contrastive_loss=args.use_contrastive_loss
        )

    # Get contrastive loss weight
    contrastive_weight = params.get('contrastive_weight', 0.1)

    # Modify for multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Loss function
    criterion = nn.BCELoss()

    # backbone parameters (CEUS backbone)
    backbone_params = []
    # static_backbone parameters
    static_backbone_params = []
    # transformer parameters
    transformer_params = []
    # classifier parameters
    classifier_params = []
    # other parameters (blood_encoder, modality_embeddings, attention_pool, etc.)
    other_params = []

    named_parameters = model.named_parameters() if not isinstance(model, nn.DataParallel) else model.module.named_parameters()
    for name, param in named_parameters:
        # Skip frozen parameters
        if name.startswith('clinical_backbone'):
            continue

        if name.startswith('backbone.') and not name.startswith('static_backbone.'):
            backbone_params.append(param)
        elif name.startswith('static_backbone.'):
            static_backbone_params.append(param)
        elif name.startswith('transformer.'):
            transformer_params.append(param)
        elif name.startswith('classifier.'):
            classifier_params.append(param)
        else:
            other_params.append(param)

    # Optimizer - Use different learning rates
    param_groups = []

    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': params.get('lr_backbone', 1e-7), 'weight_decay': params.get('weight_decay', 1e-4)})
    if static_backbone_params:
        param_groups.append({'params': static_backbone_params, 'lr': params.get('lr_static_backbone', 2e-6), 'weight_decay': params.get('weight_decay', 1e-4)})
    if transformer_params:
        param_groups.append({'params': transformer_params, 'lr': params.get('lr_transformer', 1e-5), 'weight_decay': params.get('weight_decay', 1e-4)})
    if classifier_params:
        param_groups.append({'params': classifier_params, 'lr': params.get('lr_classifier', 1e-4), 'weight_decay': params.get('weight_decay', 1e-4)})
    if other_params:
        param_groups.append({'params': other_params, 'lr': params.get('lr_other', 1e-5), 'weight_decay': params.get('weight_decay', 1e-4)})

    optimizer = optim.AdamW(param_groups)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=params['scheduler_step'], 
        gamma=params['scheduler_gamma']
    )
    
    # Training parameters
    num_epochs = params['num_epochs']
    # Set initial best validation metric based on save_sign
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
        # Best model saving related (based on AUC)
        best_val_auc = 0.0
    
    # Record training history
    train_losses = []
    val_losses = []
    train_contrastive_losses = []
    val_contrastive_losses = []
    val_metrics = []  # Store AUC or Loss based on save_sign
    
    print(f"\n{'='*60}")
    print(f"Start training, hyperparameters: {params}")
    print(f"Contrastive loss weight (contrastive_weight): {contrastive_weight}")
    if args.save_sign == "Loss_AUC":
        print(f"Save mode: Loss_AUC (early stop based on Loss, save best AUC model)")
    else:
        print(f"Save metric: {args.save_sign}")
    print(f"{'='*60}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        # Training
        train_loss, train_ce_loss, train_contrast_loss, train_auc, train_acc, train_tpr, train_tnr, train_precision, train_f1, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, contrastive_weight=contrastive_weight
        )

        # Sub-validation set
        sub_val_loss, sub_val_ce_loss, sub_val_contrast_loss, sub_val_auc, sub_val_acc, sub_val_tpr, sub_val_tnr, sub_val_precision, sub_val_f1, sub_val_labels, sub_val_preds = validate_epoch(
            model, sub_val_loader, criterion, device, contrastive_weight=contrastive_weight
        )

        # Test set (only for monitoring)
        test_loss, test_ce_loss, test_contrast_loss, test_auc, test_acc, test_tpr, test_tnr, test_precision, test_f1, _, _ = validate_epoch(
            model, test_loader, criterion, device, contrastive_weight=contrastive_weight
        )

        # External test set (only for monitoring)
        external_test_loss, external_test_ce_loss, external_test_contrast_loss, external_test_auc, external_test_acc, external_test_tpr, external_test_tnr, external_test_precision, external_test_f1, _, _ = validate_epoch(
            model, external_test_loader, criterion, device, contrastive_weight=contrastive_weight
        )
        
        # Update learning rate
        scheduler.step()

        # Record losses and metrics
        train_losses.append(train_loss)
        val_losses.append(sub_val_loss)
        train_contrastive_losses.append(train_contrast_loss)
        val_contrastive_losses.append(sub_val_contrast_loss)
        if args.save_sign == "AUC":
            val_metrics.append(sub_val_auc)
        else:
            val_metrics.append(sub_val_loss)

        # Print results
        current_lrs = {group['lr'] for group in optimizer.param_groups}
        print(f"Current learning rate groups: {sorted([f'{lr:.2e}' for lr in current_lrs])}")
        print(f"Train     - Loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}, Contrast: {train_contrast_loss:.4f}), "
              f"AUC: {train_auc:.4f}, ACC: {train_acc:.4f}, TPR: {train_tpr:.4f}, TNR: {train_tnr:.4f}")
        print(f"Sub_Val   - Loss: {sub_val_loss:.4f} (CE: {sub_val_ce_loss:.4f}, Contrast: {sub_val_contrast_loss:.4f}), "
              f"AUC: {sub_val_auc:.4f}, ACC: {sub_val_acc:.4f}, TPR: {sub_val_tpr:.4f}, TNR: {sub_val_tnr:.4f}")
        print(f"Test      - Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, "
              f"TPR: {test_tpr:.4f}, TNR: {test_tnr:.4f}, Precision: {test_precision:.4f}, F1: {test_f1:.4f}")
        print(f"External  - Loss: {external_test_loss:.4f}, AUC: {external_test_auc:.4f}, ACC: {external_test_acc:.4f}, "
              f"TPR: {external_test_tpr:.4f}, TNR: {external_test_tnr:.4f}, Precision: {external_test_precision:.4f}, F1: {external_test_f1:.4f}")

        # Plot sub-validation set confusion matrix
        plot_confusion_matrix(sub_val_labels, sub_val_preds, epoch+1, 'SubValidation', save_dir)

        # Plot loss curves
        plot_loss_curves(train_losses, val_losses, save_dir)
        # Plot contrastive loss curves
        plot_contrastive_loss_curves(train_contrastive_losses, val_contrastive_losses, save_dir)

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

    # Train set evaluation
    train_loss_final, train_ce_loss_final, train_contrast_loss_final, train_auc_final, train_acc_final, train_tpr_final, train_tnr_final, train_precision_final, train_f1_final, _, _ = evaluate_with_patient_info(
        model, train_loader, criterion, device, save_dir, mode='train', contrastive_weight=contrastive_weight
    )

    # Sub-validation set evaluation
    sub_val_loss_final, sub_val_ce_loss_final, sub_val_contrast_loss_final, sub_val_auc_final, sub_val_acc_final, sub_val_tpr_final, sub_val_tnr_final, sub_val_precision_final, sub_val_f1_final, _, _ = evaluate_with_patient_info(
        model, sub_val_loader, criterion, device, save_dir, mode='sub_val', contrastive_weight=contrastive_weight
    )

    # Test set evaluation (on best sub-validation model)
    test_loss_final, test_ce_loss_final, test_contrast_loss_final, test_auc_final, test_acc_final, test_tpr_final, test_tnr_final, test_precision_final, test_f1_final, test_labels_final, test_preds_final = evaluate_with_patient_info(
        model, test_loader, criterion, device, save_dir, mode='test', contrastive_weight=contrastive_weight
    )

    # External test set evaluation (on best sub-validation model)
    external_test_loss_final, external_test_ce_loss_final, external_test_contrast_loss_final, external_test_auc_final, external_test_acc_final, external_test_tpr_final, external_test_tnr_final, external_test_precision_final, external_test_f1_final, external_test_labels_final, external_test_preds_final = evaluate_with_patient_info(
        model, external_test_loader, criterion, device, save_dir, mode='external_test', contrastive_weight=contrastive_weight
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

        f.write(f"Train Set Results:\n")
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
        default="ALL_modality_combined_fusion_addition",
        help="Modality name for logging/saving"
    )
    parser.add_argument(
        "--fusion_way",
        type=str,
        default="HTCM",
        choices=["Avg", "Max", "Attention", 
                 "HTCM", "Tokshift", "PTA",
                 "TCStyleContext"],
        help="Modality name for logging/saving ()"
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="BUS_CDFI",
        choices=["BUS_CDFI", "BUS_CEUS", "CDFI_CEUS", "BUS_CDFI_CEUS", "ALL"],
        help="Modality compent name for logging/saving ()"
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
        "--use_contrastive_loss",
        action="store_true",
        help="Contrastive loss"
    )
    parser.add_argument(
        "--save_sign",
        type=str,
        default="Loss_AUC",
        choices=["AUC", "Loss", "Loss_AUC"],
        help="Save the best model based on validation AUC or Loss (default: Loss). Loss_AUC: early stop by loss, but save best AUC model"
    )
    parser.add_argument(
        "--pretrained_ceus",
        type=str,
        default=None,
        help="Path to pretrained CEUS model weights (step3 best_model.pth). Only backbone and HTCM weights will be loaded."
    )
    parser.add_argument(
        "--pretrained_clinical",
        type=str,
        default=None,
        help="Path to pretrained Clinical model weights (step1 best_model.pth). Only clinical_backbone weights will be loaded."
    )
    return parser

def main():
    args = build_argparser().parse_args()

    modality = args.modality
    fold_num = args.fold_num
    cuda_id = args.cuda_id
    batch_size_list = parse_int_list(args.batch_size)
    use_contrastive_loss = args.use_contrastive_loss

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
    # Use MultiModelDataset, supporting four datasets:
    # - train: training set (original training set)
    # - val with sub_val=True: sub-validation set (split from training set)
    # - test: test set
    # - external_test: external test set
    print("Loading datasets...")
    use_pil = False
    use_clinical_text = True  # Enable clinical text features, load input_ids and attention_mask

    # Training set
    train_dataset = MultiModelDataset(pt_data_dict, image_dirs, clinical_data_path,
                                        split='train', use_pil=use_pil, sub_val=True, raw_ceus=True,
                                        fold_num=fold_num, use_clinical_text=use_clinical_text)

    # Sub-validation set (split from training set)
    sub_val_dataset = MultiModelDataset(pt_data_dict, image_dirs, clinical_data_path,
                                          split='val', use_pil=use_pil, sub_val=True, raw_ceus=True,
                                          fold_num=fold_num, use_clinical_text=use_clinical_text)

    # Test set
    test_dataset = MultiModelDataset(pt_data_dict, image_dirs, clinical_data_path,
                                       split='test', use_pil=use_pil, raw_ceus=True,
                                       fold_num=fold_num, use_clinical_text=use_clinical_text)

    # External test set
    external_test_dataset = MultiModelDataset(pt_data_dict, image_dirs, clinical_data_path,
                                                split='external_test', use_pil=use_pil, raw_ceus=True,
                                                fold_num=fold_num, use_clinical_text=use_clinical_text)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Sub Validation samples: {len(sub_val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"External Test samples: {len(external_test_dataset)}")

    # Define hyperparameter search space
    hyperparam_space = {
        # Use different learning rates for different modules
        'lr_backbone': [1e-6],
        'lr_static_backbone': [1e-6],
        'lr_transformer': [1e-5],
        'lr_classifier': [1e-4, 1e-3],
        'lr_other': [1e-5],
        'weight_decay': [1e-4],
        'scheduler_step': [10],
        'scheduler_gamma': [0.9],
        'num_epochs': [50],
        'batch_size': batch_size_list,
        'contrastive_weight': [0.1],  # Contrastive loss weight, controls the magnitude of contrastive loss
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
    print(f"  Train set AUC: {best_results['train_auc']:.4f}")
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
# Example of loading step3 pretrained CEUS weights and step1 pretrained Clinical weights:
python code/step4_ALL_modality.py \
    --modality ALL_modality_all \
    --fold_num 1 \
    --batch_size 16 \
    --save_sign Loss_AUC \
    --fusion_way HTCM \
    --fusion_mode ALL \
    --pretrained_ceus /path/to/step3_ceus/best_model.pth \
    --pretrained_clinical /path/to/step1_clinical/best_model.pth
'''
if __name__ == "__main__":
    main()