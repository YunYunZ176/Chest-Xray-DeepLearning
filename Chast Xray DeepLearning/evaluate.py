import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# You Can Use 3 Models
MODEL_NAME = "DenseNet121"
ARCH = "densenet121"
MODEL_PATH = "best_model_DenseNet121.pth"
TEST_DIR = r"chest_xray/test"

INPUT_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join("evaluation_results", MODEL_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Device:", DEVICE)

# Transform
test_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
print("Classes:", test_dataset.class_to_idx)
print("Test samples:", len(test_dataset))

image_paths = [x[0] for x in test_dataset.samples]
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
def create_model(arch):
    if arch == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
    elif arch == "densenet121":
        model = torchvision.models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, 2)
    elif arch == "inception_v3":
        model = torchvision.models.inception_v3(weights=None, aux_logits=False)
        model.fc = nn.Linear(2048, 2)
    else:
        raise ValueError("Unknown architecture")
    return model

model = create_model(ARCH)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()
print("Model loaded:", MODEL_NAME)

# Testing
all_targets = []
all_predictions = []
all_probs = []

with torch.no_grad():
    for imgs, targets in test_loader:

        imgs = imgs.to(DEVICE)
        outputs = model(imgs)

        probs = torch.softmax(outputs, dim=1)
        pneumonia_prob = probs[:, 1]
        predictions = (pneumonia_prob >= 0.5).long()

        all_targets.extend(targets.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probs.extend(pneumonia_prob.cpu().numpy())

all_targets = np.array(all_targets)
all_predictions = np.array(all_predictions)
all_probs = np.array(all_probs)

# Metrics
accuracy = accuracy_score(all_targets, all_predictions)
precision = precision_score(all_targets, all_predictions, zero_division=0)
recall = recall_score(all_targets, all_predictions, zero_division=0)
f1 = f1_score(all_targets, all_predictions, zero_division=0)
auc = roc_auc_score(all_targets, all_probs)
cm = confusion_matrix(all_targets, all_predictions)

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Save Prediction
df = pd.DataFrame({
    "image": image_paths,
    "true_label": all_targets,
    "pred_label": all_predictions,
    "pneumonia_probability": all_probs
})

df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

print("\n")
print("=" * 60)
print(MODEL_NAME, "RESULT")
print("=" * 60)
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Sensitivity  : {sensitivity:.4f}")
print(f"Specificity  : {specificity:.4f}")
print(f"F1-score     : {f1:.4f}")
print(f"AUC          : {auc:.4f}")
print("\nConfusion Matrix")
print(cm)

# Save Metrics
normal_total = cm[0].sum()
normal_correct = cm[0, 0]
normal_wrong = cm[0, 1]

pneumonia_total = cm[1].sum()
pneumonia_correct = cm[1, 1]
pneumonia_wrong = cm[1, 0]

normal_recall = normal_correct / normal_total
pneumonia_recall = pneumonia_correct / pneumonia_total

metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")

with open(metrics_path, "w", encoding="utf-8") as f:

    f.write("=" * 60 + "\n")
    f.write(f"{MODEL_NAME} Test Results\n")
    f.write("=" * 60 + "\n\n")

    # Overall metrics
    f.write("Overall Performance\n")
    f.write("-" * 40 + "\n")
    f.write(f"Test Samples : {len(all_targets)}\n")
    f.write(f"Accuracy     : {accuracy:.4f}\n")
    f.write(f"Precision    : {precision:.4f}\n")
    f.write(f"Sensitivity  : {sensitivity:.4f}\n")
    f.write(f"Specificity  : {specificity:.4f}\n")
    f.write(f"F1-score     : {f1:.4f}\n")
    f.write(f"AUC          : {auc:.4f}\n")
    f.write("\n\n")

    # Confusion matrix
    f.write("Confusion Matrix\n")
    f.write("-" * 40 + "\n")
    f.write(str(cm))
    f.write("\n\n")

    # Classification report
    f.write("Classification Report\n")
    f.write("-" * 40 + "\n")
    f.write(classification_report(all_targets, all_predictions, target_names=["NORMAL", "PNEUMONIA"], digits=4, zero_division=0))
    f.write("\n\n")

    # Class-wise analysis
    f.write("Class-wise Analysis\n")
    f.write("-" * 40 + "\n\n")

    f.write("NORMAL\n")
    f.write(f"Total       : {normal_total}\n")
    f.write(f"Correct     : {normal_correct}\n")
    f.write(f"Wrong       : {normal_wrong}\n")
    f.write(f"Recall      : {normal_recall * 100:.2f}%\n")

    f.write("\n")

    f.write("PNEUMONIA\n")
    f.write(f"Total       : {pneumonia_total}\n")
    f.write(f"Correct     : {pneumonia_correct}\n")
    f.write(f"Wrong       : {pneumonia_wrong}\n")
    f.write(f"Recall      : {pneumonia_recall * 100:.2f}%\n")

print("Metrics saved:")
print(metrics_path)

# Confusion Matrix Plot
plt.figure(figsize=(5, 5))
plt.imshow(cm)
plt.title(MODEL_NAME + " Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["NORMAL", "PNEUMONIA"])
plt.yticks([0, 1], ["NORMAL", "PNEUMONIA"])
plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(all_targets, all_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(MODEL_NAME + " ROC Curve")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "ROC_curve.png"), dpi=300)
plt.close()

print("\nEvaluation finished.")
print("Saved to:", OUTPUT_DIR)
