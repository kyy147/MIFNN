# Grad-CAM visualization for transformer fusion model on external test set
# BUS / CDFI / CEUS: Grad-CAM heatmaps for final fusion prediction
# ClinicalBERT: token / feature importance for final fusion prediction

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Packages
import cv2
import glob
import random
import warnings
import argparse
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from transformers import AutoModel, AutoTokenizer

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Dataset
from utils.MultiModelDataset import MultiModelDataset

# Fusion blocks
from utils.cross_attention import VectorAttentionBottleneck
from utils.blood_fusion import HTCM

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

##------ Time Roman ------##
import matplotlib.font_manager as fm
from pathlib import Path
font_dir = Path("/usr/local/share/fonts/truetype/times-new-roman")
font_files = list(font_dir.glob("*.TTF")) + list(font_dir.glob("*.ttf"))

for font_file in font_files:
    fm.fontManager.addfont(str(font_file))
regular_font = font_dir / "TIMES.TTF"
times_name = fm.FontProperties(fname=str(regular_font)).get_name()

print("Using font:", times_name)

plt.rcParams.update({
    "font.family": times_name,
    "font.serif": [times_name],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ===================== Fixed seed =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(1)

pretrained_model_name = "./ckpt/pretrained_weights/dinov3-vitb16-pretrain-lvd1689m"
CLINICALBERT_MODEL_NAME = "./ckpt/pretrained_weights/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b"

# ===================== Feature mapping from dataset =====================
feature_mapping = {
    "慢性病史（0无；1有）": "History of chronic disease",
    "增强特点（0均匀强化；1不均匀强化；2可见三期无增强区）": "Overall enhancement pattern",
    "延迟期(等0；1低)": "Late phase eegree of enhancement",
    "坏死（0无；1有）": "Necrosis",
    "AFP分类": "AFP classification",

    "病灶最大径(单位cm）": "Maximum diameter of lesion (cm)",
    "达峰用时": "Time to reach the peak (s)",
    "低增强开始时间(s）": "Start time of washout (s)",
    "AFP": "AFP level (ng/mL)",
    "周边浸润（mm)": "Tumor infiltration boundary (mm)"
}

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

discrete_cols = [k for k in feature_mapping.keys() if k in discrete_value_mapping.keys()]
float_cols = [k for k in feature_mapping.keys() if k not in discrete_value_mapping.keys()]


# ===================== Model definition =====================
class DINOv3Classifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name,
        num_classes=1,
        feature_dim=768,
        fusion_way="transformer",
        fusion_mode="ALL",
        use_contrastive_loss=False,
        pretrained_clinical_path=None,
        pretrained_ceus_path=None,
        num_transformer_layers=2,
        num_heads=4,
    ):
        super(DINOv3Classifier, self).__init__()

        self.backbone = AutoModel.from_pretrained(pretrained_model_name, device_map="cpu")
        self.static_backbone = AutoModel.from_pretrained(pretrained_model_name, device_map="cpu")
        self.imagenet = False

        self.clinical_backbone = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            device_map="cpu",
            local_files_only=True,
        )
        self._setup_clinical_backbone(pretrained_clinical_path)

        self.fusion_way = fusion_way
        self.fusion_mode = fusion_mode
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

        classifier_input_dim = feature_dim * 2 if fusion_way == "transformer" else feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

        self._initialize_weights()
        self.device = next(self.backbone.parameters()).device
        self.classifier = self.classifier.to(self.device)

        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

        self.blood_encoder = HTCM(
            d_model=feature_dim,
            n_heads=4,
            n_layers=2,
            fusion_type="weighted_sum"
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.modality_embeddings = nn.Parameter(torch.randn(3, 1, feature_dim))
        nn.init.normal_(self.modality_embeddings, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.bottle_neck = VectorAttentionBottleneck(
            d_model=768,
            num_heads=4,
            bottleneck_size=768
        )

        self._setup_ceus_backbone(pretrained_ceus_path)

    def _setup_clinical_backbone(self, pretrained_clinical_path):
        if pretrained_clinical_path is None:
            return
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(pretrained_clinical_path, map_location=device, weights_only=False)
        pretrained_dict = checkpoint.get("model_state_dict", checkpoint)
        clinical_state_dict = self.clinical_backbone.state_dict()
        loaded_count = 0

        for k, v in pretrained_dict.items():
            target_key = None
            if k.startswith("clinical_backbone."):
                target_key = k[len("clinical_backbone."):]
            elif k.startswith("bert."):
                target_key = k[len("bert."):]
            elif not any(p in k for p in ["classifier", "fc", "head", "dropout", "pooler"]):
                target_key = k

            if target_key and any(p in target_key for p in ["classifier", "fc", "head", "dropout"]):
                continue

            if target_key and target_key in clinical_state_dict:
                if clinical_state_dict[target_key].shape == v.shape:
                    clinical_state_dict[target_key] = v
                    loaded_count += 1

        self.clinical_backbone.load_state_dict(clinical_state_dict)
        print(f"Clinical weights loaded: {loaded_count} params")

        for param in self.clinical_backbone.parameters():
            param.requires_grad = False

    def _setup_ceus_backbone(self, pretrained_ceus_path):
        if pretrained_ceus_path is None:
            return

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading CEUS weights: {pretrained_ceus_path}")
        checkpoint = torch.load(pretrained_ceus_path, map_location=device, weights_only=False)
        pretrained_dict = checkpoint.get("model_state_dict", checkpoint)
        model_dict = self.state_dict()

        ceus_keys = ["backbone.", "blood_encoder."]
        loaded_count = 0
        for k, v in pretrained_dict.items():
            if any(k.startswith(p) for p in ceus_keys):
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded_count += 1

        self.load_state_dict(model_dict)
        print(f"CEUS weights loaded: {loaded_count} params")

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def info_nce_loss(self, dynamic_feat, static_feat, temperature=0.07):
        dynamic_feat = torch.nn.functional.normalize(dynamic_feat, dim=1)
        static_feat = torch.nn.functional.normalize(static_feat, dim=1)
        logits = torch.matmul(dynamic_feat, static_feat.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss1 = torch.nn.functional.cross_entropy(logits, labels)
        loss2 = torch.nn.functional.cross_entropy(logits.T, labels)
        cl = (loss1 + loss2) / 2.0
        if isinstance(cl, torch.Tensor) and cl.numel() > 1:
            cl = cl.mean()
        return cl

    # ---------- helper methods for interpretation ----------
    def extract_static_feature(self, image, branch="bus"):
        out = self.static_backbone(image)
        pooled = out["pooler_output"]
        if not self.imagenet:
            if branch == "bus":
                pooled = self.bus_bn(pooled)
            elif branch == "cdfi":
                pooled = self.cdfi_bn(pooled)
            else:
                raise ValueError(f"Unknown static branch: {branch}")
        return pooled

    def extract_ceus_frame_feature(self, frame_image):
        out = self.backbone(pixel_values=frame_image)
        return out["pooler_output"]

    def extract_ceus_all_frame_features(self, ceus_seq):
        batch_size, seq_len, c, h, w = ceus_seq.size()
        x_flat = ceus_seq.view(batch_size * seq_len, c, h, w).contiguous()
        spatial_features = self.backbone(pixel_values=x_flat)["pooler_output"]
        sequence_features = spatial_features.view(batch_size, seq_len, -1)
        return sequence_features

    def aggregate_ceus_sequence_features(self, sequence_features):
        if self.fusion_way == "transformer_ablation":
            return sequence_features.mean(dim=1)
        return self.blood_encoder(sequence_features)

    def extract_clinical_feature(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        clinical_outputs = self.clinical_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
        )
        return clinical_outputs.last_hidden_state[:, 0, :]

    def fuse_features(self, bus_features, cdfi_features, dynamic_feat, clinical_features):
        # real forward logic: BUS is gated by CDFI
        gated_bus_features = self.cdfi_score(cdfi_features) * bus_features

        if self.fusion_way == "transformer":
            B = gated_bus_features.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1)
            mod_embeds = self.modality_embeddings.expand(-1, B, -1).transpose(0, 1)

            bus_feat_enc = gated_bus_features + mod_embeds[:, 0, :]
            cdfi_feat_enc = cdfi_features + mod_embeds[:, 1, :]
            ceus_feat_enc = dynamic_feat + mod_embeds[:, 2, :]

            seq = torch.cat(
                [
                    cls_tokens,
                    bus_feat_enc.unsqueeze(1),
                    cdfi_feat_enc.unsqueeze(1),
                    ceus_feat_enc.unsqueeze(1),
                ],
                dim=1
            )
            transformer_out = self.transformer(seq)
            fused_repr = transformer_out[:, 0, :]
            fused_repr = torch.cat([fused_repr, clinical_features], dim=-1)
        else:
            fused_repr = dynamic_feat

        pred = self.classifier(fused_repr).squeeze(-1)
        return pred

    def forward(self, bus_image, cdfi_image, input_ids, attention_mask, ceus_seq):
        bus_features = self.extract_static_feature(bus_image, branch="bus")
        cdfi_features = self.extract_static_feature(cdfi_image, branch="cdfi")
        sequence_features = self.extract_ceus_all_frame_features(ceus_seq)
        dynamic_feat = self.aggregate_ceus_sequence_features(sequence_features)
        clinical_features = self.extract_clinical_feature(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_contrastive_loss:
            contrastive_loss = self.info_nce_loss(dynamic_feat, bus_features) + \
                               self.info_nce_loss(dynamic_feat, cdfi_features)
        else:
            contrastive_loss = torch.tensor(0.0, device=dynamic_feat.device, requires_grad=False)

        pred = self.fuse_features(
            bus_features=bus_features,
            cdfi_features=cdfi_features,
            dynamic_feat=dynamic_feat,
            clinical_features=clinical_features
        )
        return pred, contrastive_loss


# ===================== Grad-CAM target =====================
class FusionBinaryTarget:
    """
    For single-output sigmoid probability:
    - target_label = 1: explain positive evidence, score = p
    - target_label = 0: explain negative evidence, score = 1 - p
    """
    def __init__(self, target_label):
        self.target_label = int(target_label)

    def __call__(self, model_output):
        score = model_output.view(-1)[0]
        return score if self.target_label == 1 else (1.0 - score)


# ===================== Grad-CAM wrappers (for final fusion output) =====================
class BUSFusionGradCAMWrapper(nn.Module):
    def __init__(self, original_model, fixed_cdfi, fixed_ceus_seq, fixed_input_ids, fixed_attention_mask):
        super().__init__()
        self.model = original_model.module if isinstance(original_model, nn.DataParallel) else original_model

        with torch.no_grad():
            self.fixed_cdfi_features = self.model.extract_static_feature(fixed_cdfi, branch="cdfi").detach()
            fixed_sequence_features = self.model.extract_ceus_all_frame_features(fixed_ceus_seq)
            self.fixed_dynamic_feat = self.model.aggregate_ceus_sequence_features(fixed_sequence_features).detach()
            self.fixed_clinical_features = self.model.extract_clinical_feature(
                input_ids=fixed_input_ids,
                attention_mask=fixed_attention_mask
            ).detach()

    def forward(self, bus_image):
        bus_features = self.model.extract_static_feature(bus_image, branch="bus")
        pred = self.model.fuse_features(
            bus_features=bus_features,
            cdfi_features=self.fixed_cdfi_features,
            dynamic_feat=self.fixed_dynamic_feat,
            clinical_features=self.fixed_clinical_features
        )
        return pred


class CDFIFusionGradCAMWrapper(nn.Module):
    def __init__(self, original_model, fixed_bus, fixed_ceus_seq, fixed_input_ids, fixed_attention_mask):
        super().__init__()
        self.model = original_model.module if isinstance(original_model, nn.DataParallel) else original_model

        with torch.no_grad():
            self.fixed_bus_features = self.model.extract_static_feature(fixed_bus, branch="bus").detach()
            fixed_sequence_features = self.model.extract_ceus_all_frame_features(fixed_ceus_seq)
            self.fixed_dynamic_feat = self.model.aggregate_ceus_sequence_features(fixed_sequence_features).detach()
            self.fixed_clinical_features = self.model.extract_clinical_feature(
                input_ids=fixed_input_ids,
                attention_mask=fixed_attention_mask
            ).detach()

    def forward(self, cdfi_image):
        cdfi_features = self.model.extract_static_feature(cdfi_image, branch="cdfi")
        pred = self.model.fuse_features(
            bus_features=self.fixed_bus_features,
            cdfi_features=cdfi_features,
            dynamic_feat=self.fixed_dynamic_feat,
            clinical_features=self.fixed_clinical_features
        )
        return pred


class CEUSFusionGradCAMWrapper(nn.Module):
    """
    Explain final fusion prediction for one CEUS frame at a time:
    - current frame goes through backbone WITH grad / hooks
    - all other frames are fixed cached features
    """
    def __init__(self, original_model, fixed_bus, fixed_cdfi, fixed_ceus_seq, fixed_input_ids, fixed_attention_mask):
        super().__init__()
        self.model = original_model.module if isinstance(original_model, nn.DataParallel) else original_model
        self.frame_idx = 0

        with torch.no_grad():
            self.fixed_bus_features = self.model.extract_static_feature(fixed_bus, branch="bus").detach()
            self.fixed_cdfi_features = self.model.extract_static_feature(fixed_cdfi, branch="cdfi").detach()
            self.fixed_clinical_features = self.model.extract_clinical_feature(
                input_ids=fixed_input_ids,
                attention_mask=fixed_attention_mask
            ).detach()
            self.fixed_sequence_features = self.model.extract_ceus_all_frame_features(fixed_ceus_seq).detach()

    def set_frame_index(self, frame_idx):
        self.frame_idx = int(frame_idx)

    def forward(self, frame_image):
        variable_frame_feature = self.model.extract_ceus_frame_feature(frame_image)   # [1, D]

        sequence_features = self.fixed_sequence_features.clone()
        sequence_features[:, self.frame_idx, :] = variable_frame_feature

        dynamic_feat = self.model.aggregate_ceus_sequence_features(sequence_features)

        pred = self.model.fuse_features(
            bus_features=self.fixed_bus_features,
            cdfi_features=self.fixed_cdfi_features,
            dynamic_feat=dynamic_feat,
            clinical_features=self.fixed_clinical_features
        )
        return pred


# ===================== Utility functions =====================
def get_base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def get_vit_target_layer(backbone, target_layer_num=-1):
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer[target_layer_num].norm1
    if hasattr(backbone, "layer"):
        return backbone.layer[target_layer_num].norm1
    return None


def infer_num_prefix_tokens(num_tokens):
    # Prefer perfect square remaining tokens, e.g. 197 -> 1 + 196, 201 -> 5 + 196
    for prefix in range(0, min(16, num_tokens)):
        remaining = num_tokens - prefix
        if remaining <= 0:
            continue
        side = int(np.sqrt(remaining))
        if side * side == remaining:
            return prefix
    return 0


def dinov3_reshape_transform(tensor, hidden_dim=None):
    # tensor: [B, N, C]
    batch_size = tensor.size(0)
    total_tokens = tensor.size(1)
    if hidden_dim is None:
        hidden_dim = tensor.size(2)

    prefix_tokens = infer_num_prefix_tokens(total_tokens)
    tensor = tensor[:, prefix_tokens:, :]  # remove CLS / register tokens if present

    num_tokens = tensor.size(1)
    height = int(np.sqrt(num_tokens))
    width = height

    if height * width != num_tokens:
        found = False
        for h in range(int(np.sqrt(num_tokens)), 0, -1):
            if num_tokens % h == 0:
                height = h
                width = num_tokens // h
                found = True
                break
        if not found:
            raise ValueError(f"Cannot reshape token sequence with {num_tokens} tokens into spatial map.")

    result = tensor.reshape(batch_size, height, width, hidden_dim)
    result = result.permute(0, 3, 1, 2)
    return result


def denormalize_img(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    return img


def save_image(img_0_1, save_path):
    img_uint8 = (img_0_1 * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(save_path)


def find_best_model(fold_num, modality_base):
    pattern = f"./logs_modalities/{modality_base}/Fold{fold_num}/**/best_model.pth"
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No best_model.pth found for {modality_base} Fold{fold_num}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


# ===================== ClinicalBERT visualization =====================
def normalize_scores(scores):
    scores = np.array(scores, dtype=np.float64)
    total = scores.sum()
    if total <= 1e-12:
        return np.zeros_like(scores)
    return scores / total


def compute_gradient_importance_fusion(
    full_model,
    bus_image,
    cdfi_image,
    ceus_seq,
    input_ids,
    attention_mask,
    target_label
):
    full_model.eval()
    model = get_base_model(full_model)

    embedding_layer = model.clinical_backbone.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids).detach().clone().requires_grad_(True)

    bus_features = model.extract_static_feature(bus_image, branch="bus")
    cdfi_features = model.extract_static_feature(cdfi_image, branch="cdfi")
    sequence_features = model.extract_ceus_all_frame_features(ceus_seq)
    dynamic_feat = model.aggregate_ceus_sequence_features(sequence_features)
    clinical_features = model.extract_clinical_feature(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask
    )

    pred = model.fuse_features(
        bus_features=bus_features,
        cdfi_features=cdfi_features,
        dynamic_feat=dynamic_feat,
        clinical_features=clinical_features
    )

    model.zero_grad(set_to_none=True)
    if inputs_embeds.grad is not None:
        inputs_embeds.grad.zero_()

    target_score = pred.view(-1)[0] if int(target_label) == 1 else (1.0 - pred.view(-1)[0])
    target_score.backward()

    gradients = inputs_embeds.grad
    importance_scores = gradients.norm(dim=-1).squeeze(0).detach().cpu().numpy()
    return importance_scores


def map_token_to_feature(text, offsets, importance_scores):
    feature_importance = {}
    text_lower = text.lower()

    for feat_cn, feat_en in feature_mapping.items():
        candidate_names = [feat_cn.lower(), feat_en.lower()]

        matched = False
        for feat_lower in candidate_names:
            start_idx = text_lower.find(feat_lower)
            if start_idx == -1:
                continue

            end_idx = start_idx + len(feat_lower)
            score_sum = 0.0
            for i, (s, e) in enumerate(offsets):
                if e <= s:
                    continue
                overlap = max(0, min(e, end_idx) - max(s, start_idx))
                if overlap > 0 and i < len(importance_scores):
                    score_sum += float(importance_scores[i])

            feature_importance[feat_en] = feature_importance.get(feat_en, 0.0) + score_sum
            matched = True
            break

        if not matched:
            continue

    if not feature_importance:
        return {}

    features = list(feature_importance.keys())
    scores = normalize_scores(np.array([feature_importance[f] for f in features], dtype=np.float64))
    return {f: float(s) for f, s in zip(features, scores)}


def visualize_feature_importance(feature_importance, save_path):
    if not feature_importance:
        return

    items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in items]
    scores = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    color_values = np.linspace(0.9, 0.35, len(scores))
    bar_colors = plt.cm.Reds(color_values)
    bars = ax.barh(range(len(features)), scores, color=bar_colors)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized Importance", fontsize=12)
    ax.set_title("ClinicalBERT Feature Importance", fontsize=14)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center",
            fontsize=9
        )

    ax.set_xlim(0, max(scores) * 1.2 if len(scores) > 0 else 1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature importance: {save_path}")


# ===================== Main visualization =====================
def run_gradcam_visualization(model, external_test_loader, device, save_dir, fold_num, max_samples=None):
    model.eval()
    base_model = get_base_model(model)

    # Find target layers once from base model
    target_layer_num = 0
    visual_target_layer = get_vit_target_layer(base_model.static_backbone, target_layer_num)
    ceus_target_layer = get_vit_target_layer(base_model.backbone, target_layer_num)

    print(f"Visual backbone target layer: {visual_target_layer}")
    print(f"CEUS backbone target layer: {ceus_target_layer}")

    if visual_target_layer is None:
        raise ValueError("Failed to locate target layer for static visual backbone.")
    if ceus_target_layer is None:
        raise ValueError("Failed to locate target layer for CEUS backbone.")

    visual_hidden_dim = base_model.static_backbone.config.hidden_size \
        if hasattr(base_model.static_backbone, "config") else 768
    ceus_hidden_dim = base_model.backbone.config.hidden_size \
        if hasattr(base_model.backbone, "config") else 768

    tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL_NAME, local_files_only=True)

    results = []
    sample_count = 0

    print("\nProcessing external test set...")
    for batch in tqdm(external_test_loader, desc="Grad-CAM visualization"):
        patient_ids = batch["patient_id"]
        bus_images = batch["bus_image"].to(device)
        cdfi_images = batch["cdfi_image"].to(device)
        ceus_seq = batch["ceus_sequence"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        B = bus_images.size(0)

        for i in range(B):
            pid = patient_ids[i]
            true_label = labels[i].item()

            bus_i = bus_images[i:i + 1]
            cdfi_i = cdfi_images[i:i + 1]
            ceus_i = ceus_seq[i:i + 1]
            input_ids_i = input_ids[i:i + 1]
            attention_mask_i = attention_mask[i:i + 1]

            with torch.no_grad():
                pred_prob, _ = model(
                    bus_i,
                    cdfi_i,
                    input_ids_i,
                    attention_mask_i,
                    ceus_i
                )
                pred_prob = pred_prob.item()
                pred_label = 1 if pred_prob >= 0.5 else 0

            print(f"\n[{sample_count + 1}] Patient: {pid}, True: {true_label}, Pred: {pred_label} ({pred_prob:.3f})")

            patient_dir = os.path.join(save_dir, f"patient_{pid}")
            os.makedirs(os.path.join(patient_dir, "BUS"), exist_ok=True)
            os.makedirs(os.path.join(patient_dir, "CDFI"), exist_ok=True)
            os.makedirs(os.path.join(patient_dir, "CEUS"), exist_ok=True)
            os.makedirs(os.path.join(patient_dir, "ClinicalBERT"), exist_ok=True)

            targets = [FusionBinaryTarget(pred_label)]

            # ===================== BUS Grad-CAM =====================
            bus_wrapper = BUSFusionGradCAMWrapper(
                model, fixed_cdfi=cdfi_i, fixed_ceus_seq=ceus_i,
                fixed_input_ids=input_ids_i, fixed_attention_mask=attention_mask_i
            ).eval()

            bus_cam = GradCAM(
                model=bus_wrapper,
                target_layers=[visual_target_layer],
                reshape_transform=lambda t: dinov3_reshape_transform(t, hidden_dim=visual_hidden_dim)
            )

            bus_original_np = denormalize_img(bus_i[0].cpu())
            grayscale_cam = bus_cam(
                input_tensor=bus_i,
                targets=targets,
                eigen_smooth=True,
                aug_smooth=True
            )[0]
            grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
            bus_overlay = show_cam_on_image(bus_original_np, grayscale_cam, use_rgb=True, image_weight=0.0)

            save_image(bus_original_np, os.path.join(patient_dir, "BUS", "original.png"))
            Image.fromarray(bus_overlay).save(os.path.join(patient_dir, "BUS", "gradcam_overlay.png"))
            print("  BUS: saved original + overlay")

            # ===================== CDFI Grad-CAM =====================
            cdfi_wrapper = CDFIFusionGradCAMWrapper(
                model, fixed_bus=bus_i, fixed_ceus_seq=ceus_i,
                fixed_input_ids=input_ids_i, fixed_attention_mask=attention_mask_i
            ).eval()

            cdfi_cam = GradCAM(
                model=cdfi_wrapper,
                target_layers=[visual_target_layer],
                reshape_transform=lambda t: dinov3_reshape_transform(t, hidden_dim=visual_hidden_dim)
            )

            cdfi_original_np = denormalize_img(cdfi_i[0].cpu())
            grayscale_cam = cdfi_cam(
                input_tensor=cdfi_i,
                targets=targets,
                eigen_smooth=True,
                aug_smooth=True
            )[0]
            grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
            cdfi_overlay = show_cam_on_image(cdfi_original_np, grayscale_cam, use_rgb=True, image_weight=0.0)

            save_image(cdfi_original_np, os.path.join(patient_dir, "CDFI", "original.png"))
            Image.fromarray(cdfi_overlay).save(os.path.join(patient_dir, "CDFI", "gradcam_overlay.png"))
            print("  CDFI: saved original + overlay")

            # ===================== CEUS Grad-CAM (per frame, final fusion output) =====================
            ceus_wrapper = CEUSFusionGradCAMWrapper(
                model,
                fixed_bus=bus_i,
                fixed_cdfi=cdfi_i,
                fixed_ceus_seq=ceus_i,
                fixed_input_ids=input_ids_i,
                fixed_attention_mask=attention_mask_i
            ).eval()

            ceus_cam = GradCAM(
                model=ceus_wrapper,
                target_layers=[ceus_target_layer],
                reshape_transform=lambda t: dinov3_reshape_transform(t, hidden_dim=ceus_hidden_dim)
            )

            ceus_seq_i = ceus_i[0]  # [T, 3, 224, 224]
            for frame_idx in range(ceus_seq_i.size(0)):
                ceus_wrapper.set_frame_index(frame_idx)

                frame_tensor = ceus_seq_i[frame_idx:frame_idx + 1]
                frame_original_np = denormalize_img(ceus_seq_i[frame_idx].cpu())

                grayscale_cam = ceus_cam(
                    input_tensor=frame_tensor,
                    targets=targets,
                    eigen_smooth=True,
                    aug_smooth=True
                )[0]
                grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
                frame_overlay = show_cam_on_image(frame_original_np, grayscale_cam, use_rgb=True, image_weight=0.0)

                frame_name = f"frame_{frame_idx + 1:02d}"
                save_image(frame_original_np, os.path.join(patient_dir, "CEUS", f"{frame_name}_original.png"))
                Image.fromarray(frame_overlay).save(os.path.join(patient_dir, "CEUS", f"{frame_name}_gradcam_overlay.png"))

            print(f"  CEUS: saved {ceus_seq_i.size(0)} frames (original + overlay)")

            # ===================== ClinicalBERT Feature Importance =====================
            clinical_text = batch.get("clinical_text", [""])
            text = clinical_text[i] if isinstance(clinical_text, list) else clinical_text[i]

            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True
            )
            offsets = list(encoding["offset_mapping"])

            importance_scores = compute_gradient_importance_fusion(
                model,
                bus_i,
                cdfi_i,
                ceus_i,
                input_ids_i,
                attention_mask_i,
                target_label=pred_label
            )

            # zero-out special tokens / invalid spans
            for idx, (s, e) in enumerate(offsets):
                if idx >= len(importance_scores):
                    break
                if e <= s:
                    importance_scores[idx] = 0.0

            feature_importance = map_token_to_feature(text, offsets, importance_scores)

            visualize_feature_importance(
                feature_importance,
                os.path.join(patient_dir, "ClinicalBERT", "feature_importance.png")
            )

            # token -> character heatmap
            char_scores = np.zeros(len(text), dtype=np.float64)
            char_counts = np.zeros(len(text), dtype=np.float64)

            for score, (s, e) in zip(importance_scores, offsets):
                if e <= s:
                    continue
                s, e = max(0, s), min(len(text), e)
                if s >= e:
                    continue
                char_scores[s:e] += float(score)
                char_counts[s:e] += 1.0

            valid_mask = char_counts > 0
            char_scores[valid_mask] = char_scores[valid_mask] / char_counts[valid_mask]

            if char_scores.max() > 0:
                char_scores = char_scores / (char_scores.max() + 1e-12)

            cmap = cm.get_cmap("Reds")
            norm = colors.Normalize(vmin=0.0, vmax=1.0)

            lines = []
            current_line = ""
            current_scores = []

            for ch, sc in zip(text, char_scores):
                current_line += ch
                current_scores.append(sc)
                if len(current_line) >= 90 and ch in [" ", "，", "。", "；", ",", "."]:
                    lines.append((current_line, current_scores))
                    current_line = ""
                    current_scores = []

            if current_line:
                lines.append((current_line, current_scores))

            fig_h = max(3, 0.65 * len(lines) + 1.5)
            fig, ax = plt.subplots(figsize=(18, fig_h))
            ax.axis("off")
            x0, y = 0.01, 0.95
            line_height = 0.12

            for line_text, line_scores in lines:
                x = x0
                for ch, sc in zip(line_text, line_scores):
                    rgba = cmap(norm(sc))
                    rgba = (rgba[0], rgba[1], rgba[2], 0.85 if sc > 0 else 0.0)
                    ax.text(
                        x,
                        y,
                        ch,
                        transform=ax.transAxes,
                        fontsize=16,
                        fontweight="bold",
                        va="top",
                        ha="left",
                        bbox=dict(boxstyle="square,pad=0.08", facecolor=rgba, edgecolor="none")
                    )
                    x += 0.0125
                y -= line_height

            ax.set_title("Clinical Text Importance Heatmap", fontsize=20, fontweight="bold")
            plt.tight_layout()
            plt.savefig(
                os.path.join(patient_dir, "ClinicalBERT", "text_importance_heatmap.png"),
                dpi=180,
                bbox_inches="tight"
            )
            plt.close()
            print("  ClinicalBERT: saved feature importance + text heatmap")

            if feature_importance:
                fi_df = pd.DataFrame({
                    "feature": list(feature_importance.keys()),
                    "normalized_importance": list(feature_importance.values())
                }).sort_values("normalized_importance", ascending=False)
                fi_df.to_csv(
                    os.path.join(patient_dir, "ClinicalBERT", "feature_importance.csv"),
                    index=False
                )

            results.append({
                "patient_id": pid,
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_prob": pred_prob
            })

            sample_count += 1
            if max_samples and sample_count >= max_samples:
                break

        if max_samples and sample_count >= max_samples:
            break

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, "inference_results.csv"), index=False)
    print(f"\nTotal samples processed: {sample_count}")
    print(f"Results saved to: {save_dir}")
    return results


def build_argparser():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for transformer fusion model")
    parser.add_argument("--fold_num", type=int, default=1, help="Fold number")
    parser.add_argument("--cuda_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (None=all)")
    parser.add_argument(
        "--modality",
        type=str,
        default="ALL_modality_combined",
        help="Model modality name"
    )
    parser.add_argument("--fusion_way", type=str, default="transformer", help="Fusion way")

    return parser

# ===================== Main =====================
def main():
    args = build_argparser().parse_args()

    fold_num = args.fold_num
    cuda_id = args.cuda_id
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
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

    pretrained_ceus_path = pretrained_ceus_path_dict[str(fold_num)]
    pretrained_clinical_path = pretrained_clinical_path_dict[str(fold_num)]

    print("Loading model...")
    model = DINOv3Classifier(
        pretrained_model_name,
        num_classes=1,
        fusion_way=args.fusion_way,
        pretrained_clinical_path=pretrained_clinical_path,
        pretrained_ceus_path=pretrained_ceus_path
    )

    model_paths = [
        "ckpt/network_weights/MIFNN_best_model_fold1.pth",
        "xxx",
        "xxx",
        "xxx",
        "xxx"
    ]
    if args.fold_num == 1:
        model_path = model_paths[0]
    elif args.fold_num == 2:
        model_path = model_paths[1]
    elif args.fold_num == 3:
        model_path = model_paths[2]
    elif args.fold_num == 4:
        model_path = model_paths[3]
    elif args.fold_num == 5:
        model_path = model_paths[4]

    print(f"Loading weights from: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    print("Model loaded successfully!")

    pt_data_dict = {
        "train": f"./datasets/pt_file/train_fold{fold_num}.pt",
        "val": f"./datasets/pt_file/internal_test_fold{fold_num}.pt",
        "test": f"./datasets/pt_file/external_test_fold{fold_num}.pt"
    }

    image_dirs = {
        "us": "./datasets/BUS",
        "cdfi": "./datasets/CDFI",
        "ceus": "./datasets/CEUS"
    }

    clinical_data_path = "./datasets/Clin/MVI-HCC-samples.xlsx"

    external_test_dataset = MultiModelDataset(
        pt_data_dict,
        image_dirs,
        clinical_data_path,
        split="external_test",
        use_pil=False,
        raw_ceus=True,
        fold_num=fold_num,
        use_clinical_text=True
    )

    external_test_loader = DataLoader(
        external_test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    print(f"External test set size: {len(external_test_dataset)}")

    save_dir = f"./gram-cam/{args.modality}/Fold{fold_num}"
    os.makedirs(save_dir, exist_ok=True)

    run_gradcam_visualization(
        model,
        external_test_loader,
        device,
        save_dir,
        fold_num,
        max_samples=args.max_samples
    )

    print(f"\nAll done! Results saved to: {save_dir}")


if __name__ == "__main__":
    main()