import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
DATA_DIR = r"../chest_xray"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
MODEL_PATH = "best_model_DenseNet121.pth"

# Transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=7),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset
train_full = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_full = datasets.ImageFolder(root=TRAIN_DIR, transform=val_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=val_transform)

print()
print("类别：", train_full.class_to_idx)
print("原始训练集：", len(train_full))
print("测试集：", len(test_dataset))

#Train / Validation Split
indices = list(range(len(train_full)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42, stratify=train_full.targets)
train_dataset = Subset(train_full, train_indices)
val_dataset = Subset(val_full, val_indices)

print()
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# DataLoader

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# DenseNet121
model = torchvision.models.densenet121(weights="DEFAULT")
for param in model.parameters():
    param.requires_grad = False
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)  # 二分类: NORMAL / PNEUMONIA
model = model.to(DEVICE)

class_weights = torch.tensor([2.0, 1.0], dtype=torch.float32).to(DEVICE)  # NORMAL / PNEUMONIA
print("Class weights:", class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

best_val_macro_f1 = 0.0

# Training
for epoch in range(EPOCHS):

    print()
    print("=" * 70)
    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    print("=" * 70)
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_bar = tqdm(train_loader, desc="Training", leave=True)
    for imgs, targets in train_bar:
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += targets.size(0)
        train_correct += (predicted == targets).sum().item()
        current_loss = train_loss / train_total
        current_acc = train_correct / train_total
        train_bar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc * 100:.2f}%"})
    train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss = 0.0
    val_total = 0
    all_val_targets = []
    all_val_predictions = []
    val_bar = tqdm(val_loader, desc="Validation", leave=True)
    with torch.no_grad():
        for imgs, targets in val_bar:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * imgs.size(0)
            val_total += targets.size(0)
            _, predicted = torch.max(outputs, 1)
            all_val_targets.extend(targets.cpu().numpy())
            all_val_predictions.extend(predicted.cpu().numpy())
            current_loss = val_loss / val_total
            current_acc = (predicted == targets).sum().item() / targets.size(0)
            val_bar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc * 100:.2f}%"})
    val_loss = val_loss / val_total
    val_acc = accuracy_score(all_val_targets, all_val_predictions)
    val_precision = precision_score(all_val_targets, all_val_predictions, average="macro", zero_division=0)
    val_recall = recall_score(all_val_targets, all_val_predictions, average="macro", zero_division=0)
    val_macro_f1 = f1_score(all_val_targets, all_val_predictions, average="macro", zero_division=0)
    val_f1_per_class = f1_score(all_val_targets, all_val_predictions, average=None, zero_division=0)
    val_precision_per_class = precision_score(all_val_targets, all_val_predictions, average=None, zero_division=0)
    val_recall_per_class = recall_score(all_val_targets, all_val_predictions, average=None, zero_division=0)
    cm = confusion_matrix(all_val_targets, all_val_predictions)

    print()
    print("-" * 70)
    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss       : {train_loss:.4f}")
    print(f"Train Accuracy   : {train_acc * 100:.2f}%")
    print()
    print(f"Val Loss         : {val_loss:.4f}")
    print(f"Val Accuracy     : {val_acc * 100:.2f}%")
    print(f"Val Macro Precision: {val_precision * 100:.2f}%")
    print(f"Val Macro Recall : {val_recall * 100:.2f}%")
    print(f"Val Macro F1     : {val_macro_f1 * 100:.2f}%")
    print()
    print(f"NORMAL Precision : {val_precision_per_class[0] * 100:.2f}%")
    print(f"NORMAL Recall    : {val_recall_per_class[0] * 100:.2f}%")
    print(f"NORMAL F1        : {val_f1_per_class[0] * 100:.2f}%")
    print()
    print(f"PNEUMONIA Precision: {val_precision_per_class[1] * 100:.2f}%")
    print(f"PNEUMONIA Recall   : {val_recall_per_class[1] * 100:.2f}%")
    print(f"PNEUMONIA F1       : {val_f1_per_class[1] * 100:.2f}%")
    print()
    print("Confusion Matrix:")
    print(cm)
    print("-" * 70)

    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1
        torch.save(model.state_dict(), MODEL_PATH)
        print()
        print(f"Save Best Model，Val Macro F1 = {val_macro_f1 * 100:.2f}%")

# Training Finished

print()
print("=" * 70)
print("训练完成")
print("=" * 70)
print(f"Best Validation Macro F1: {best_val_macro_f1 * 100:.2f}%")
print(f"Best Model: {MODEL_PATH}")