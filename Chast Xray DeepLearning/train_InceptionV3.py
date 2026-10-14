import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from tqdm import tqdm

DATA_DIR = r"../chest_xray"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
SAVE_DIR = r"../train_model"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model_InceptionV3.pth")
IMAGE_SIZE = 299
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
VAL_RATIO = 0.20
SEED = 42
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

print("=" * 72)
print("InceptionV3 Frozen Binary Classification")
print("=" * 72)
print(f"Device        : {DEVICE}")
print(f"Dataset       : {TRAIN_DIR}")
print(f"Model output  : {MODEL_PATH}")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Epochs        : {EPOCHS}")
print(f"Learning rate : {LEARNING_RATE}")
print(f"Validation ratio: {VAL_RATIO}")
print(f"Random seed   : {SEED}")
print("=" * 72)

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory does not exist: {TRAIN_DIR}")
os.makedirs(SAVE_DIR, exist_ok=True)

# Transform
train_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.Grayscale(num_output_channels=3), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
eval_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Load Data
print("Loading dataset...")
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
full_eval_dataset = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)
expected_mapping = {"NORMAL": 0, "PNEUMONIA": 1}
print(f"Class mapping: {full_train_dataset.class_to_idx}")
if full_train_dataset.class_to_idx != expected_mapping:
    raise RuntimeError(f"Unexpected class mapping. Expected {expected_mapping}, got {full_train_dataset.class_to_idx}")
train_size = int((1.0 - VAL_RATIO) * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
split_generator = torch.Generator().manual_seed(SEED)
train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size], generator=split_generator)
train_dataset = torch.utils.data.Subset(full_train_dataset, train_subset.indices)
val_dataset = torch.utils.data.Subset(full_eval_dataset, val_subset.indices)
print(f"Total images: {len(full_train_dataset)} | Train: {len(train_dataset)} | Validation: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

# InceptionV3
print("Loading pretrained InceptionV3...")
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
model.AuxLogits = None
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 2)
for param in model.fc.parameters():
    param.requires_grad = True
model = model.to(DEVICE)
trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
total_params = sum(param.numel() for param in model.parameters())
print(f"Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# Validation Function
def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    all_targets, all_preds, all_probs = [], [], []
    progress = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS} Validation", leave=False, unit="batch")
    with torch.no_grad():
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())
            progress.set_postfix(loss=f"{loss.item():.4f}")
    all_targets, all_preds, all_probs = np.array(all_targets), np.array(all_preds), np.array(all_probs)
    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
    macro_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"loss": avg_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1, "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1, "auc": auc, "cm": cm}

# Training
best_val_f1 = -1.0
best_epoch = -1
best_state_dict = None
print("Starting training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    running_loss = 0.0
    train_targets, train_preds = [], []
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Training", leave=True, unit="batch")
    for images, labels in train_progress:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        predictions = torch.argmax(logits, dim=1)
        train_targets.extend(labels.detach().cpu().numpy())
        train_preds.extend(predictions.detach().cpu().numpy())
        train_progress.set_postfix(loss=f"{loss.item():.4f}")
    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = accuracy_score(train_targets, train_preds)
    val_results = evaluate(model, val_loader, criterion, DEVICE, epoch)
    print(f"\nEpoch [{epoch:02d}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Val Loss: {val_results['loss']:.4f} | Val Accuracy: {val_results['accuracy'] * 100:.2f}%")
    print(f"Val Precision: {val_results['precision'] * 100:.2f}% | Val Recall: {val_results['recall'] * 100:.2f}% | Val F1: {val_results['f1'] * 100:.2f}%")
    print(f"Val Macro Precision: {val_results['macro_precision'] * 100:.2f}% | Val Macro Recall: {val_results['macro_recall'] * 100:.2f}% | Val Macro F1: {val_results['macro_f1'] * 100:.2f}%")
    print(f"Val Specificity: {val_results['specificity'] * 100:.2f}% | Val AUC: {val_results['auc']:.4f}")
    print(f"Val Confusion Matrix:\n{val_results['cm']}")
    current_val_f1 = val_results["macro_f1"]
    if current_val_f1 > best_val_f1:
        best_val_f1 = current_val_f1
        best_epoch = epoch
        best_state_dict = copy.deepcopy(model.state_dict())
        torch.save({"model_state_dict": best_state_dict, "class_to_idx": full_train_dataset.class_to_idx, "epoch": best_epoch, "macro_f1": best_val_f1}, MODEL_PATH)
        print(f"Best model updated: Macro F1={best_val_f1 * 100:.2f}% | Saved to: {MODEL_PATH}")

# Training Finished
print("=" * 72)
print("TRAINING FINISHED")
print(f"Best epoch: {best_epoch}")
print(f"Best validation Macro F1: {best_val_f1 * 100:.2f}%")
print(f"Best model path: {MODEL_PATH}")
print("=" * 72)
