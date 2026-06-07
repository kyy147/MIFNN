# Microvascular Invasion Prediction in Hepatocellular Carcinoma
The use of multimodal ultrasound imaging and clinical parameters, as well as cross-center generalizability, remains insufficient in the prediction of microvascular invasion (MVI). \
We aimed to develop and validate a deep learning model, the multi-modality information fusion neural network (**MIFNN**), integrating multimodal ultrasound imaging and clinical parameters for noninvasive preoperative prediction of MVI status in patients with hepatocellular carcinoma (HCC).
## Overall workflow
![workflow](/utils/images/workflow.png)

## Architecture of the MIFNN
![MIFNN](/utils/images/MIFNN.png)

## Main experiment
### Experiment1: Diagnostic Performance of Single-Modality Models
This experiment evaluates the diagnostic performance of individual modalities for MVI prediction, including clinical parameters, BUS, CDFI, and DCE-US. For clinical parameters, machine learning models and ClinicalBERT were compared. For BUS and CDFI, different image backbones and pretraining strategies were evaluated. For DCE-US, different frame sampling strategies and temporal modeling methods were tested.

The results show that DCE-US achieved the best single-modality performance, with an external validation AUC of 0.8545 ± 0.0198. Clinical parameters achieved moderate performance, while BUS and CDFI showed relatively lower but complementary predictive value. These results indicate that dynamic perfusion information from DCE-US is the most informative single source for MVI prediction.

### Experiment2: Diagnostic Performance of Multimodal Fusion
This experiment investigates whether combining multiple modalities can improve MVI prediction. Different modality combinations were evaluated, including BUS + CDFI, BUS + DCE-US, CDFI + DCE-US, BUS + CDFI + DCE-US, and the full multimodal model integrating BUS, CDFI, DCE-US, and clinical parameters.

The results demonstrate that multimodal fusion consistently improved model performance compared with single-modality models. The full multimodal model achieved the best overall performance, with an external validation AUC of 0.8953 ± 0.0180, accuracy of 81.18% ± 2.83%, sensitivity of 86.40% ± 6.69%, specificity of 78.14% ± 6.28%, and F1 score of 77.16% ± 2.74%. These findings suggest that morphological information, blood flow distribution, dynamic perfusion, and clinical parameters provide complementary information for MVI prediction.

### Experiment3: Effectiveness of Modules in MIFNN
This experiment evaluates the effectiveness of key modules in the proposed MIFNN framework, including the Transformer-based fusion module, the hemodynamic temporal change module (HTCM), and the representation consistency learning module (RCLM). Different fusion strategies were compared, including addition, average, concatenation, bilinear fusion, Attention Bottleneck, and Transformer-based fusion. Ablation experiments were further conducted by removing HTCM and/or RCLM.

The results show that Transformer-based fusion outperformed traditional fusion strategies, indicating that attention-based cross-modal interaction is more effective for integrating BUS, CDFI, and DCE-US features. In the ablation study, both HTCM and RCLM improved model performance, and the best performance was achieved when both modules were included. This confirms that dynamic temporal modeling of DCE-US and cross-modal representation alignment are both important for improving MVI prediction.


## Project Structure
```
MIFNN
├─ ckpt
│  ├─ network_weights
│  │  ├─ ClinicalBERT_best_model.pt
│  │  ├─ DCE-US_HTCM_best_model.pth
│  │  └─ MIFNN_best_model.pth
│  └─ pretrained_weights
│     ├─ dinov3-vitb16-pretrain-lvd1689m
│     ├─ models--emilyalsentzer--Bio_ClinicalBERT
│     └─ models--facebook--timesformer-base-finetuned-k600
│  
├─ code
│  ├─ step0_show_var_details.py
│  ├─ step0_clin_stat_analysis.py
│  ├─ step1_clin_var_clinicalbert.py
│  ├─ step1_clin_var_gcn.py
│  ├─ step1_clin_var_ml.py
│  ├─ step2_BUS_CDFI.py
│  ├─ step3_CEUS_2D_merge_Compare.py
│  ├─ step3_CEUS_3D.py
│  ├─ step4_ALL_modality.py
│  ├─ step4_ALL_modality_fusion_method_Compare.py
│  └─ step4_ALL_modality_grad-cam.py
│  
├─ dataset
│  ├─ BUS
│  │  ├─ id1_BUS.png
│  │  ├─ ...
│  │  └─ idx_BUS.png
│  ├─ CDFI
│  │  ├─ id1_CDFI.png
│  │  ├─ ...
│  │  └─ idx_CDFI.png
│  ├─ Clin
│  │  └─ idx_BUS.xlsx
│  ├─ CEUS
│  │  ├─ id1
│  │  |   ├─ id1_00001.png
│  │  |   ├─ ...
│  │  |   └─ id1_00016.png
│  │  ├─ ...
│  │  └─ idx
│  │      ├─ idx_00001.png
│  │      ├─ ...
│  │      └─ idx_00016.png
│  ├─ Clin
│  │  └─ clinical_data.xlsx
│  └─ pt_file
│     ├─ train_fold1.pt
│     ├─ internal_test_fold1.pt
│     └─ external_test_fold1.pt
│  
├─ gram-cam
│  └─ patient_xxx
│
├─ utils
│  ├─ MultiModelDataset.py
│  ├─ blood_fusion.py
│  ├─ ...
│  └─ fusion_xxx.py
│
├─ README.md
└─ requirements.txt
```
**ckpt:** the network_weights folder contains the trained model weights; the pretrained_weights folder contains the official and original model weights. 
[Download](https://drive.google.com/drive/folders/10Ri_dwGoNbCkSx-5_QIENBl7Uu2YKznu?usp=drive_link)
<br>
**code:** this folder contains the code for training and testing.<br>
**datasets:** this folder contains multimodal sample data.<br>
**gram-cam:** example results of Gram-Cam inference, visualization of the importance of multimodal data<br>
**utils:** this folder contains the implementation of the dataset division and fusion module.<br>

## How to run the code
### Install the dependencies
> conda create -n MIFNN python=3.9 \
> conda activate MIFNN \
> pip install -r requirements.txt 

**Note.** This project uses Python 3.9.19, torch 2.7.1+cu118. Also, we used four NVIDIA GTX 3090 for the experiments.<br>

### The operation details are incode/crun_code.md
Run "step4_ALL_modality.py" as an example:
> python code/step4_ALL_modality.py \
    --modality ALL_modality_all \
    --fold_num 1 \
    --batch_size 16 \
    --save_sign Loss_AUC \
    --fusion_way HTCM \
    --fusion_mode ALL \
    --pretrained_ceus /path/to/step3_ceus/best_model.pth \
    --pretrained_clinical /path/to/step1_clinical/best_model.pth

Inference gram-cam:
> python code/step4_ALL_modality_grad-cam.py

## Contact information
If you have any questions, feel free to contact me.

Yuanyuan Kong
Shenzhen University, Shenzhen, China.

E-mail: kyy2019look@163.com

