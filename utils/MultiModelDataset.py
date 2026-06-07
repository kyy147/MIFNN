import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Torch
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Package
import random
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ClinicalBERT related configuration
CLINICALBERT_MODEL_NAME = "./ckpt/pretrained_weights/models--emilyalsentzer--Bio_ClinicalBERT/snapshots/d5892b39a4adaed74b92212a44081509db72f87b"
MAX_TEXT_LENGTH = 512

# Feature mapping and value mapping (used for generating English text)
# Significant variables
feature_mapping = {
    # Discrete variables
    "慢性病史（0无；1有）": "History of chronic disease",
    "增强特点（0均匀强化；1不均匀强化；2可见三期无增强区）": "Overall enhancement pattern",
    "延迟期(等0；1低)": "Late Phase Degree of Enhancement",
    "坏死（0无；1有）": "Necrosis",
    "AFP分类": "AFP classification",
    
    # Continuous variables
    "病灶最大径(单位cm）": "Maximum diameter of lesion(cm)",
    "达峰用时": "Time to reach the peak（s）",
    "低增强开始时间(s）": "Start time of washout (s)",
    "AFP": "AFP level(ng/mL)",
    "周边浸润（mm)": "Tumor infiltration boundary(mm)"
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

# Automatically identify discrete/continuous columns
discrete_cols = [k for k in feature_mapping.keys() if k in discrete_value_mapping.keys()]
float_cols = [k for k in feature_mapping.keys() if k not in discrete_value_mapping.keys()]

class MultiModelDataset(Dataset):
    def __init__(self, pt_data_dict, image_dirs, clinical_data_path, split='train', use_pil=False,
                 use_clinical_text=True, tokenizer=None, ceus_seq_feat=False,
                 raw_ceus=False, fold_num=1, sub_val_ratio=0.25, sub_val=False):
        """
        Args:
            pt_data_dict: Dictionary containing 'train', 'val', 'test' keys, each corresponding to a pt file path
            image_dirs: Dictionary containing 'us' and 'cdfi' keys, corresponding to image folder paths for two modalities
            clinical_data_path: Path to clinical data xlsx file
            split: Current dataset type ('train', 'val', 'test', 'external_test')
            use_pil: Whether to return CEUS images as PIL format list, True returns PIL list, False returns tensor
            use_clinical_text: Whether to generate clinical text features (for ClinicalBERT)
            tokenizer: ClinicalBERT tokenizer, automatically loaded if None
            ceus_seq_feat: Whether to use CEUS sequence features
            raw_ceus: Whether to load raw CEUS image data, True loads, False returns None
            fold_num: Number of folds for data splitting
            sub_val_ratio: Ratio for splitting validation set from training set
            sub_val: Whether to use sub-validation set splitting
        """
        self.use_pil = use_pil
        self.use_clinical_text = use_clinical_text
        self.raw_ceus = raw_ceus
        self.fold_num = fold_num
        self.sub_val_ratio = sub_val_ratio
        self.sub_val = sub_val
        self.ceus_seq_feat = ceus_seq_feat

        # Import split_dataset function
        import sys
        sys.path.insert(0, './utils')
        from data_split_ import split_dataset

        # Get valid ID list corresponding to current split
        if split == 'external_test':
            # External test set
            self.valid_ids = split_dataset(fold_num=fold_num, split='test',
                                          pt_path='./datasets/pt_file',
                                          sub_val_ratio=sub_val_ratio, sub_val=False)
        elif split == 'test':
            # Test set (using val split)
            self.valid_ids = split_dataset(fold_num=fold_num, split='val',
                                          pt_path='./datasets/pt_file',
                                          sub_val_ratio=sub_val_ratio, sub_val=False)
        else:
            # Training or validation set
            self.valid_ids = split_dataset(fold_num=fold_num, split=split,
                                          pt_path='./datasets/pt_file',
                                          sub_val_ratio=sub_val_ratio, sub_val=sub_val)
        
        # 1. Load and process clinical data (integrated text generation)
        self.clinical_data_path = clinical_data_path
        self.clinical_df = self._process_clinical_data(clinical_data_path)
        
        # 2. Initialize ClinicalBERT tokenizer (if needed)
        if self.use_clinical_text:
            self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(CLINICALBERT_MODEL_NAME, local_files_only=True)
            self.max_text_length = MAX_TEXT_LENGTH
            # Generate English text
            self.clinical_df['english_text'] = self.clinical_df.apply(self._generate_english_sentence, axis=1)
        
        # 3. Load image file lists
        us_images = {os.path.basename(f).split('_')[0]: os.path.join(image_dirs['us'], f) 
                    for f in os.listdir(image_dirs['us']) if f.endswith('.png')}
        cdfi_images = {os.path.basename(f).split('_')[0]: os.path.join(image_dirs['cdfi'], f) 
                       for f in os.listdir(image_dirs['cdfi']) if f.endswith('.png')}
        
        # CEUS image processing logic
        ceus_root_dir = image_dirs.get('ceus', None)
        ceus_sequences = {}
        if ceus_root_dir and os.path.exists(ceus_root_dir):
            for patient_id in os.listdir(ceus_root_dir):
                patient_ceus_dir = os.path.join(ceus_root_dir, patient_id)
                if os.path.exists(patient_ceus_dir):
                    ceus_files = [os.path.join(patient_ceus_dir, f) 
                                  for f in sorted(os.listdir(patient_ceus_dir)) 
                                  if f.endswith('.png')]  # Fix duplicate loop issue in original code
                    if len(ceus_files) >= 16:  # Ensure at least 16 images
                        ceus_sequences[patient_id] = ceus_files[:16]  # Take only first 16 images
                    else:
                        warnings.warn(f"Patient {patient_id} has less than 16 CEUS images.")
        
        # 4. Load pt data
        self.split = split

        # Determine which pt file to load based on split
        if split == 'external_test':
            pt_file_key = 'test'
        elif split == 'test':
            pt_file_key = 'val'
        elif split == 'val' and sub_val:
            pt_file_key = 'train'  # Use original train split for val
        elif split == 'train' and sub_val:
            pt_file_key = 'train'

        pt_data = torch.load(pt_data_dict[pt_file_key], weights_only=False)
        self.ceus_seq_feat = 'sequence_features' if ceus_seq_feat else 'feat'

        if self.valid_ids is not None:
            # common_ids = self.valid_ids
            common_ids = list(set(self.clinical_df['患者编号'].astype(str)) & set(self.valid_ids))

        # 6. Filter data, keep only common IDs
        self.clinical_data = self.clinical_df[self.clinical_df['患者编号'].astype(str).isin(common_ids)]
        self.us_images = {k: v for k, v in us_images.items() if k in common_ids}
        self.cdfi_images = {k: v for k, v in cdfi_images.items() if k in common_ids}
        self.pt_data = {k: v for k, v in pt_data.items() if k in common_ids}
        
        # Add CEUS data filtering
        if ceus_root_dir:
            self.ceus_sequences = {k: v for k, v in ceus_sequences.items() if k in common_ids}
        else:
            self.ceus_sequences = {}
        
        # Ensure consistent ordering
        self.patient_ids = sorted(common_ids)
        
        # Preprocessing parameters
        self.image_val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.image_train_transform = transforms.Compose([
            # Randomly crop 80%~100% of the image area, then resize to 224x224
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=(0.7, 1.0),    # Crop area ratio: 80%~100%
                ratio=(0.9, 1.1),    # Aspect ratio range to avoid excessive distortion
                interpolation=transforms.InterpolationMode.BILINEAR
            ),

            # ==========Non-rigid deformation=========
            # Random perspective transformation (non-rigid: simulating view tilt, lens distortion)
            # transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            # Random shear deformation (non-rigid: simulating object tilt, stretch distortion)
            # transforms.RandomAffine(degrees=0, shear=(-8, 8, -8, 8)),
            # Elastic transformation (standard non-rigid: local pixel distortion, simulating soft tissue/object elastic deformation)
            # transforms.ElasticTransform(alpha=50.0, sigma=7.0),
            # Random grid distortion (non-rigid: local grid distortion, fine-grained deformation)
            # transforms.RandomGridShuffle(grid=(3, 3), p=0.2),

            # ==========Spatial transformation =========
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10,expand=False),  # Random image rotation

            # ===========Non-linear image transformation==========
            # Random histogram equalization (non-linear brightness/contrast transformation, classic non-linear operation)
            transforms.RandomEqualize(p=0.2),
            # Random image sharpness adjustment (non-linear filter transformation)
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            # Random tone separation (non-linear color quantization)
            transforms.RandomPosterize(bits=6, p=0.2),
            # Random color inversion (non-linear pixel transformation)
            transforms.RandomSolarize(threshold=200, p=0.1),

            # ===========Color variation==========
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _generate_english_sentence(self, row):
        """Generate structured English clinical description text"""
        sentence_parts = ["The patient's condition is as follows:"]
        
        # Process discrete variables
        for col in discrete_cols:
            if col in row and pd.notna(row[col]):
                val = row[col]
                english_col = feature_mapping[col]
                # Handle numeric type conversion
                try:
                    val = int(val)
                    english_val = discrete_value_mapping[col].get(val, str(val))
                except (ValueError, TypeError):
                    english_val = str(val)
                sentence_parts.append(f"{english_col} is {english_val}.")
        
        # Process continuous variables (using original values, not standardized values)
        # First try to get original column, if not exists use current value
        for col in float_cols:
            if col in row and pd.notna(row[col]):
                # Check if there's an original value column (_raw suffix)
                val = row.get(f"{col}_raw", row[col])
                english_col = feature_mapping[col]
                try:
                    val = float(val)
                    sentence_parts.append(f"{english_col} is {val:.2f}.")
                except (ValueError, TypeError):
                    sentence_parts.append(f"{english_col} is {val}.")
        
        return " ".join(sentence_parts)
    
    def _tokenize_clinical_text(self, text):
        """Tokenize clinical text and return model input"""
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_text_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
    
    def _preload_ceus_sequences_tensor(self):
        """Preload CEUS image sequences as tensor format (original logic)"""
        print(f"Preloading CEUS sequences (tensor) for {len(self.patient_ids)} patients...")
        for patient_id in self.patient_ids:
            ceus_images = []
            if patient_id in self.ceus_sequences:
                for img_path in self.ceus_sequences[patient_id]:
                    ceus_img = Image.open(img_path).convert('RGB')
                    if self.split == 'train':
                        ceus_img = self.image_train_transform(ceus_img)
                    else:
                        ceus_img = self.image_val_transform(ceus_img)
                    ceus_images.append(ceus_img)
                # Convert list to tensor and store
                self.preloaded_ceus_sequences[patient_id] = torch.stack(ceus_images, dim=0)  # Shape: (16, C, H, W)
            else:
                # If no CEUS data, create empty tensor placeholder
                self.preloaded_ceus_sequences[patient_id] = torch.zeros((16, 3, 224, 224))
        print("CEUS sequences (tensor) preloading completed.")
    
    def _preload_ceus_sequences_pil(self):
        """Preload CEUS image sequences as PIL format"""
        print(f"Preloading CEUS sequences (PIL) for {len(self.patient_ids)} patients...")
        for patient_id in self.patient_ids:
            ceus_images = []
            if patient_id in self.ceus_sequences:
                for img_path in self.ceus_sequences[patient_id]:
                    # Load only PIL images, no transform (keep original format)
                    ceus_img = Image.open(img_path).convert('RGB')
                    ceus_images.append(ceus_img)
            else:
                # If no CEUS data, create empty list or placeholder PIL images
                ceus_images = [Image.new('RGB', (224, 224), color=0) for _ in range(16)]
            
            self.preloaded_ceus_sequences_pil[patient_id] = ceus_images
        print("CEUS sequences (PIL) preloading completed.")
    
    def _process_clinical_data(self, clinical_data_path):
        """Process clinical data, keep original values for text generation, and process structured features"""
        # Read Excel file
        df = pd.read_excel(clinical_data_path)
        
        # Ensure patient ID is string
        df['患者编号'] = df['患者编号'].astype(str)
        
        # Define columns
        categorical_columns = [
            "慢性病史（0无；1有）",
            "增强特点（0均匀强化；1不均匀强化；2可见三期无增强区）",
            "门脉期（0：等增强；1低增强；2高增强）",
            "延迟期(等0；1低)",
            "边界光整（0光整；1不光整）",
            "坏死（0无；1有）",
            "AFP分类"
        ]
        
        continuous_columns = [
            "病灶最大径(单位cm）",
            "低增强开始时间(s）",
            "AFP",
        ]
        
        # Save original continuous values (for text generation)
        for col in continuous_columns:
            if col in df.columns:
                df[f"{col}_raw"] = df[col].copy()
        
        # Process labels
        label_col = "xx"
        if label_col not in df.columns:
            label_col = "MVI"  # Compatible with original label column name
        df['label'] = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)

        # Process categorical features
        for col in categorical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process continuous features
        for col in continuous_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Missing value imputation
        # Categorical features filled with mode
        cat_imputer = SimpleImputer(strategy='most_frequent')
        if categorical_columns:
            df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
        
        # Continuous features filled with mean
        cont_imputer = SimpleImputer(strategy='mean')
        if continuous_columns:
            df[continuous_columns] = cont_imputer.fit_transform(df[continuous_columns])
        
        # Standardize and normalize continuous features
        scaler = StandardScaler()
        if continuous_columns:
            df[continuous_columns] = scaler.fit_transform(df[continuous_columns])
        
        # OneHot encode categorical features
        encoded_cats = np.array([])
        if categorical_columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_cats = encoder.fit_transform(df[categorical_columns])
        
        # Create feature DataFrame
        if len(encoded_cats) > 0:
            encoded_cat_df = pd.DataFrame(encoded_cats, 
                                        columns=encoder.get_feature_names_out(categorical_columns))
            cont_df = df[continuous_columns].reset_index(drop=True)
            # Merge features
            df['clinical_features'] = pd.concat([encoded_cat_df, cont_df], axis=1).values.tolist()
        else:
            df['clinical_features'] = df[continuous_columns].values.tolist()
        
        return df
    
    def __len__(self):
        return len(self.patient_ids)
    
    def _apply_consistent_ceus_transform(self, pil_images):
        """
        Apply completely consistent random augmentation parameters to CEUS sequences of the same case.
        Only enabled for training set; validation/test sets still use regular val transform.
        """
        if self.split != 'train':
            return [self.image_val_transform(img) for img in pil_images]

        # Generate a random seed for current case
        seed = int.from_bytes(os.urandom(4), 'little')

        # Save current random state to avoid affecting BUS / CDFI random augmentation
        torch_rng_state = torch.get_rng_state()
        py_rng_state = random.getstate()

        transformed_images = []
        try:
            for img in pil_images:
                # Return to the same seed for each image
                # This way, all 16 images will sample exactly the same random augmentation parameters
                random.seed(seed)
                torch.manual_seed(seed)

                transformed_images.append(self.image_train_transform(img))
        finally:
            # Restore random state to avoid affecting other modalities
            torch.set_rng_state(torch_rng_state)
            random.setstate(py_rng_state)

        return transformed_images
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # 1. Get pt features
        if self.split in ['test', 'external_test']:
            feats = self.pt_data[patient_id][self.ceus_seq_feat]
        else:
            feats = self.pt_data[patient_id][self.ceus_seq_feat].float()
        
        # 2. Load images
        bus_image = Image.open(self.us_images[patient_id]).convert('RGB')
        cdfi_image = Image.open(self.cdfi_images[patient_id]).convert('RGB')

        # Process raw CEUS images
        if self.raw_ceus:
            if patient_id in self.ceus_sequences:
                ceus_pil_images = [
                    Image.open(img_path).convert('RGB')
                    for img_path in self.ceus_sequences[patient_id]
                ]

                # Apply consistent data augmentation to 16 CEUS images of the same case
                ceus_images = self._apply_consistent_ceus_transform(ceus_pil_images)
                ceus_sequence = torch.stack(ceus_images, dim=0)  # Shape: (16, C, H, W)
            else:
                ceus_sequence = None
        else:
            ceus_sequence = torch.tensor(0.0, dtype=torch.float32)

        if self.split == 'train':
            bus_image = self.image_train_transform(bus_image)
            cdfi_image = self.image_train_transform(cdfi_image)
        else:
            bus_image = self.image_val_transform(bus_image)
            cdfi_image = self.image_val_transform(cdfi_image)
        
        # 3. Get clinical features and label
        clinical_row = self.clinical_data[self.clinical_data['患者编号'] == patient_id].iloc[0]
        clinical_feature = torch.tensor(clinical_row['clinical_features'], dtype=torch.float32)
        label = torch.tensor(clinical_row['label'], dtype=torch.long)
        
        # Build return dictionary
        result = {
            'patient_id': patient_id,
            'feats': feats,
            'bus_image': bus_image,
            'cdfi_image': cdfi_image,
            'ceus_sequence': ceus_sequence,  # Returns tensor when raw_ceus=True, None when False
            'clinical_features': clinical_feature,
            'label': label,
        }
        
        # 4. If needed, add clinical text features
        if self.use_clinical_text:
            english_text = clinical_row['english_text']
            text_encoding = self._tokenize_clinical_text(english_text)
            
            result.update({
                'clinical_text': english_text,
                'input_ids': text_encoding['input_ids'],
                'attention_mask': text_encoding['attention_mask']
            })
        
        return result


# ---------------------- Usage example ----------------------
if __name__ == "__main__":
    # Example
    fold_num = 1
    pt_data_dict = {
        'train': f'./datasets/pt_file/train_fold{fold_num}.pt',
        'val': f'./datasets/pt_file/internal_test_fold{fold_num}.pt',
        'test': f'./datasets/pt_file/external_test_fold{fold_num}.pt'
    }
    image_dirs = {
        'us': "./datasets/BUS",
        'cdfi': "./datasets/CDFI",
        'ceus': "./datasets/CEUS"
    }
    clinical_data_path = "./datasets/Clin/MVI-HCC-samples.xlsx"

    # Create dataset
    dataset = MultiModelDataset(
        pt_data_dict=pt_data_dict,
        image_dirs=image_dirs,
        clinical_data_path=clinical_data_path,
        split='external_test', # external_test
        use_pil=False,
        use_clinical_text=True,
        ceus_seq_feat=False,
        raw_ceus=True,
        fold_num=1,
        sub_val_ratio=0.25,
        sub_val=True
    )
    
    # Test data loading
    print("Dataset size:", len(dataset))
    sample = dataset[0]
    print("Patient ID:", sample['patient_id'])
    print("PT feature shape:", sample['feats'].shape)
    print("BUS image shape:", sample['bus_image'].shape)
    print("Clinical structured feature shape:", sample['clinical_features'].shape)
    print("CEUS sequence shape:", sample['ceus_sequence'].shape)
    print("Label:", sample['label'])
    if 'clinical_text' in sample:
        print("Clinical text (first 100 characters):", sample['clinical_text'][:512])
        print("Text input IDs shape:", sample['input_ids'].shape)
        print("Text attention mask shape:", sample['attention_mask'].shape)
    clinical_bert = AutoModelForSequenceClassification.from_pretrained(
                CLINICALBERT_MODEL_NAME,
                num_labels=2,
                problem_type="single_label_classification",
                weights_only=True
            )
    bert_pretrained_model = "./ckpt/network_weights/ClinicalBERT_best_model_fold1.pt"
    checkpoint = torch.load(bert_pretrained_model, map_location='cpu', weights_only=False)
    if "model_state_dict" in checkpoint:
        clinical_bert.load_state_dict(checkpoint["model_state_dict"])
    else:
        clinical_bert.load_state_dict(checkpoint)