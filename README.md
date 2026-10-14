Chest X-Ray Pneumonia Classification

Purpose

This project trains and compares three CNN models for automated pneumonia detection from chest X-rays. Then interprets their decisions with visual explanation methods (Grad-CAM and Occlusion sensitivity).

Models

ResNet18 |  DenseNet121 | InceptionV3

Methods 

Dataset: Pneumonia Chest X-Ray Images.
Classes: NORMAL (0) / PNEUMONIA (1).
Preprocessing: grayscale converted to 3 channels, resize (224x224 for ResNet18/DenseNet121, 299x299 for InceptionV3), ImageNet normalization.
Augmentation (only for training):  random horizontal flip, affine translation/scale, random resized crop.
Split: stratified 80/20 train/validation split of the official training set (random_state=42); the official test set (624 images: 234 NORMAL, 390 PNEUMONIA) is used only for final evaluation and analysis.
Class imbalance: weighted cross-entropy (NORMAL 2.0 / PNEUMONIA 1.0 for ResNet18).
Optimization: lr=1e-3, batch size 32, 20 epochs; only the classifier head is trained (backbone frozen).
Model selection: best checkpoint by validation Macro F1.
Evaluation: threshold 0.5 on P(PNEUMONIA); metrics include accuracy, precision, sensitivity, specificity, F1, AUC, confusion matrix, ROC curve; per-image predictions exported to CSV.

Results

All metrics below are taken from the existing result files under "results/". Test set: 624 images.

Overall comparison

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 | AUC |

| ResNet18 | 0.8510| 0.8180 | 0.9795 | 0.6368 | 0.8915 | 0.9449 |
| DenseNet121 | 0.8494 | 0.8246 | 0.9641 | 0.6581 | 0.8889 | 0.9460 |
| InceptionV3 | 0.7452 | 0.7104 | 1.0000 | 0.3205 | 0.8307 | 0.9364 |

Explainability

Grad-CAM

Generated for TP / TN / FP / FN categories (up to 6 random samples each, seed 42), overlaid on the original X-ray with the "jet" colormap.
(T: TRUE; F: FALSE; N: NORMAL; P: PNEUMONIA)

Occlusion Sensitivity

Systematic sliding-window occlusion; patch/stride = 32/16 (ResNet18, DenseNet121) and 40/20 (InceptionV3). Reports per-image max probability drop, its location, and an edge-vs-center comparison (edge = outer 25% of image, center = central 50%).

Limitations

The dataset is class-imbalanced, results depend on the loss-weighting scheme. Despite weighting, all models remain biased toward the majority (PNEUMONIA) class, and InceptionV3 effectively collapses to a majority-class predictor at threshold 0.5.
