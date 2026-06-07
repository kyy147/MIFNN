## step0_show_var_details.py
Displays detailed information and statistics about clinical variables in the dataset for exploratory data analysis.<br>
**Run:** 
> python step0_show_var_details.py

## step0_clin_stat_analysis.py:
Performs clinical statistical analysis on patient data to examine relationships between clinical variables and outcomes.<br>
**Run:** 
> python step0_clin_stat_analysis.py

## step1_clin_var_gcn.py:
Applies Graph Convolutional Network (GCN) approach to analyze clinical variables with graph-based relationships.<br>
**Run:** 
> python step1_clin_var_gcn.py

## step1_clin_var_ml.py:
Implements machine learning approaches(LogisticRegression, XGBoost, SVM, RandomForest) for analyzing clinical variables and predicting outcomes.<br>
**Run:** 
> python step1_clin_var_ml.py

## step1_clin_var_clinicalbert.py:
Implements clinical variable processing using ClinicalBERT model for natural language understanding of medical text data.<br>
**Run:** 
> python code/step1_clin_var_clinicalbert.py --fold_num 1 --cuda_id 0 --modality Clin_ClinicalBert

**Parameter Explanation:**<br>
**fold_num:** the number of folds for the 50% cross-validation, ranging **from 1 to 5<br>
**cuda_id:** gpu id<br>
**modality:** the name for saving the experimental folder<br>

## step2_BUS_CDFI.py:
Processes Breast Ultrasound (BUS) and Color Doppler Flow Imaging (CDFI) modalities for feature extraction and analysis.<br>
**Run:** <br>
> python code/step2_BUS_CDFI.py --fold_num 1 --modality CDFI_ViT_B_MAE_v2 --batch_size 16 --save_sign Loss_AUC --fusion_way CDFI

**Parameter Explanation:**<br>
**fold_num:** the number of folds for the 50% cross-validation, ranging from 1 to 5<br>
**modality:** the name for saving the experimental folder<br>
**batch_size:** the batch size of the training<br>
**save_sign:** Strategy for saving the best model. Loss: Save the model with the lowest Loss on the validation set; AUC: Save the model with the highest AUC on the validation set; Loss_AUC: Perform early stopping when the Loss on the validation set does not decrease for 5 consecutive rounds, and save the model with the highest AUC on the validation set.<br>
**fusion_way:** specify to use BUS or CDFI for the experiment<br>

## step3_CEUS_2D_merge_Compare.py:
Merges and compares Contrast-Enhanced Ultrasound (CEUS) 2D features with different fusion strategies.<br>
**Run:** 
> python code/step3_CEUS_2D_merge_Compare.py --modality CEUS_2D_dinov3_HTCM --fold_num 1 --batch_size 16 --save_sign Loss_AUC --fusion_way HTCM

**Parameter Explanation:**<br>
**fold_num:** the number of folds for the 50% cross-validation, ranging from 1 to 5<br>
**modality:** the name for saving the experimental folder<br>
**batch_size:** the batch size of the training<br>
**save_sign:** Strategy for saving the best model. Loss: Save the model with the lowest Loss on the validation set; AUC: Save the model with the highest AUC on the validation set; Loss_AUC: Perform early stopping when the Loss on the validation set does not decrease for 5 consecutive rounds, and save the model with the highest AUC on the validation set.<br>
**fusion_way:** specify the 2D fusion methods, including HTCM, TokShift, PTA, and TCStyleContext.<br>

## step3_CEUS_3D.py:
Processes Contrast-Enhanced Ultrasound (CEUS) 3D modality for volumetric feature extraction and analysis.<br>
**Run:** 
> python code/step3_CEUS_3D.py --modality CEUS_3D_TimeSformer --fold_num 1 --batch_size 16 --save_sign Loss_AUC

**Parameter Explanation:**<br>
**fold_num:** the number of folds for the 50% cross-validation, ranging from 1 to 5<br>
**modality:** the name for saving the experimental folder<br>
**batch_size:** the batch size of the training<br>
**save_sign:** Strategy for saving the best model. Loss: Save the model with the lowest Loss on the validation set; AUC: Save the model with the highest AUC on the validation set; Loss_AUC: Perform early stopping when the Loss on the validation set does not decrease for 5 consecutive rounds, and save the model with the highest AUC on the validation set.<br>

## step4_ALL_modality.py:
Integrates all imaging modalities for comprehensive multi-modal analysis and prediction.<br>
**Run:** 
> python code/step4_ALL_modality.py \
    --modality ALL_modality_all \
    --fold_num 1 \
    --batch_size 16 \
    --save_sign Loss_AUC \
    --fusion_way HTCM \
    --pretrained_ceus /path/to/step3_ceus/best_model.pth \
    --pretrained_clinical /path/to/step1_clinical/best_model.pth

**Parameter Explanation:**<br>
**fold_num:** the number of folds for the 50% cross-validation, ranging from 1 to 5<br>
**modality:** the name for saving the experimental folder<br>
**batch_size:** the batch size of the training<br>
**save_sign:** Strategy for saving the best model. Loss: Save the model with the lowest Loss on the validation set; AUC: Save the model with the highest AUC on the validation set; Loss_AUC: Perform early stopping when the Loss on the validation set does not decrease for 5 consecutive rounds, and save the model with the highest AUC on the validation set.<br>
**fusion_way：** specify the 2D fusion method, which can be used for ablation, including Avg, Max, Attention, and HTCM.<br>
**pretrained_ceus:** CEUS model weight path<br>
**pretrained_clinical:** ClinicalBERT model weight path<br>
**use_contrastive_loss:** whether to use contrastive loss<br>

## step4_ALL_modality_fusion_method_Compare.py:
Compares different fusion methods for combining multiple imaging modalities to determine optimal integration strategy.<br>
**Run:** 
> python code/step4_ALL_modality_fusion_method_Compare.py \
    --modality ALL_modality_combined_fusion_addition \
    --fold_num 1 \
    --batch_size 16 \
    --save_sign Loss_AUC \
    --fusion_way addition \
    --pretrained_ceus /path/to/step3_ceus/best_model.pth \
    --pretrained_clinical /path/to/step1_clinical/best_model.pth \
    --use_contrastive_loss

**Parameter Explanation:**<br>
**fold_num:** the number of folds for the 50% cross-validation, ranging from 1 to 5<br>
**modality:** the name for saving the experimental folder<br>
**batch_size:** the batch size of the training<br>
**save_sign:** Strategy for saving the best model. Loss: Save the model with the lowest Loss on the validation set; AUC: Save the model with the highest AUC on the validation set; Loss_AUC: Perform early stopping when the Loss on the validation set does not decrease for 5 consecutive rounds, and save the model with the highest AUC on the validation set.<br>
**fusion_way:** specify modal fusion methods: addition, avg, concate, bottleneck, transformer<br>
**pretrained_ceus:** CEUS model weight path<br>
**pretrained_clinical：** ClinicalBERT model weight path<br>
**use_contrastive_loss:** whether to use contrastive loss<br>

## step4_ALL_modality_grad-cam.py:*
Implements Grad-CAM visualization technique for interpreting deep learning models trained on multi-modal data.<br>
**Run:** 
> python code/step4_ALL_modality_grad-cam.py

