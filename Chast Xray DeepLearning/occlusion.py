import os
import csv
import random
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# elect one model to run: "resnet18" | "densenet121" | "inception_v3"
MODEL_TYPE = "inception_v3"
MODEL_PATHS = {
    "resnet18": r"best_model_resnet18.pth",
    "densenet121": r"best_model_densenet121.pth",
    "inception_v3": r"best_model_inception_v3.pth",
}
TEST_DIR = r"chest_xray/test"
if MODEL_TYPE not in MODEL_PATHS:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")
MODEL_PATH = MODEL_PATHS[MODEL_TYPE]
# Per-model input size and occlusion parameters
if MODEL_TYPE == "inception_v3":
    IMAGE_SIZE = 299
    PATCH_SIZE = 40
    STRIDE = 20
else:
    IMAGE_SIZE = 224
    PATCH_SIZE = 32
    STRIDE = 16

SAVE_NUM = 6
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
OUTPUT_DIR = Path("occlusion_results") / MODEL_TYPE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("Model:", MODEL_TYPE)
print("Model path:", MODEL_PATH)
print("Test directory:", TEST_DIR)
print("Image size:", IMAGE_SIZE)
print("Patch size:", PATCH_SIZE)
print("Stride:", STRIDE)
print("Samples per category:", SAVE_NUM)
print("Output directory:", OUTPUT_DIR)
print("=" * 65)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

# Transform and Dataset
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

print("\nClass mapping:", test_dataset.class_to_idx)
print("Test samples:", len(test_dataset))

if test_dataset.class_to_idx.get("NORMAL") != 0:
    raise ValueError(f"Expected NORMAL label 0. Actual mapping: {test_dataset.class_to_idx}")
if test_dataset.class_to_idx.get("PNEUMONIA") != 1:
    raise ValueError(f"Expected PNEUMONIA label 1. Actual mapping: {test_dataset.class_to_idx}")

# Build Model
def build_model():
    if MODEL_TYPE == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
    elif MODEL_TYPE == "densenet121":
        model = torchvision.models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, 2)
    elif MODEL_TYPE == "inception":
        model = torchvision.models.inception_v3(weights=None, aux_logits=False)
        model.fc = nn.Linear(2048, 2)
    else:
        raise ValueError(f"Unsupported model: {MODEL_TYPE}")
    return model.to(DEVICE)
model = build_model()

# Load Trained Checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)
model.eval()
print("\nModel loaded successfully.")

# Classify TP / TN / FP / FN
def classify_samples():
    categories = {"TP": [], "TN": [], "FP": [], "FN": []}
    print("\nClassifying test samples...")

    for index in tqdm(range(len(test_dataset)), desc="Classification"):
        image_tensor, true_label = test_dataset[index]
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_label = torch.argmax(output, dim=1).item()

        if true_label == 1 and predicted_label == 1:
            category = "TP"
        elif true_label == 0 and predicted_label == 0:
            category = "TN"
        elif true_label == 0 and predicted_label == 1:
            category = "FP"
        else:
            category = "FN"

        categories[category].append(index)

    print("\n" + "=" * 55)
    print("Classification Results")
    print("=" * 55)
    for category, indices in categories.items():
        print(f"{category}: {len(indices)}")

    return categories

categories = classify_samples()

# Occlusion Sensitivity Core Function
def occlusion_analysis(model, image_tensor, patch_size=PATCH_SIZE, stride=STRIDE, batch_size=32):
    model.eval()
    if image_tensor.ndim != 3:
        raise ValueError("image_tensor must have shape [C, H, W]")
    _, height, width = image_tensor.shape
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if patch_size > height or patch_size > width:
        raise ValueError("patch_size must not exceed image dimensions")
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        original_output = model(input_tensor)
        original_probability = torch.softmax(original_output, dim=1)[0, 1].item()

    y_positions = list(range(0, height - patch_size + 1, stride))
    x_positions = list(range(0, width - patch_size + 1, stride))
    if y_positions[-1] != height - patch_size:y_positions.append(height - patch_size)
    if x_positions[-1] != width - patch_size:x_positions.append(width - patch_size)

    positions = [(y, x) for y in y_positions for x in x_positions]
    occluded_images = []
    for y, x in positions:
        occluded = image_tensor.clone()
        occluded[:, y:y + patch_size, x:x + patch_size] = 0.0
        occluded_images.append(occluded)
    occluded_images = torch.stack(occluded_images, dim=0)
    occluded_probabilities = []
    with torch.no_grad():
        for start in range(0, len(occluded_images), batch_size):
            batch = occluded_images[start:start + batch_size].to(DEVICE)
            outputs = model(batch)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            occluded_probabilities.extend(probabilities.cpu().numpy().tolist())
    occluded_probabilities = np.asarray(occluded_probabilities, dtype=np.float32)
    probability_drops = original_probability - occluded_probabilities

    occlusion_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)
    for (y, x), drop in zip(positions, probability_drops):
        occlusion_map[y:y + patch_size, x:x + patch_size] += drop
        count_map[y:y + patch_size, x:x + patch_size] += 1

    occlusion_map /= np.maximum(count_map, 1e-8)

    max_index = int(np.argmax(probability_drops))
    max_drop = float(probability_drops[max_index])
    max_y, max_x = positions[max_index]
    center_y = max_y + patch_size / 2
    center_x = max_x + patch_size / 2

    edge_limit_x = width * 0.25
    edge_limit_y = height * 0.25
    center_min_x = edge_limit_x
    center_max_x = width - edge_limit_x
    center_min_y = edge_limit_y
    center_max_y = height - edge_limit_y
    max_drop_is_edge = not (
        center_min_x <= center_x <= center_max_x
        and center_min_y <= center_y <= center_max_y
    )
    edge_drops = []
    center_drops = []

    for (y, x), drop in zip(positions, probability_drops):
        patch_center_y = y + patch_size / 2
        patch_center_x = x + patch_size / 2
        is_center = (
            center_min_x <= patch_center_x <= center_max_x
            and center_min_y <= patch_center_y <= center_max_y
        )
        if is_center:
            center_drops.append(float(drop))
        else:
            edge_drops.append(float(drop))
    max_edge_drop = max(edge_drops) if edge_drops else float("nan")
    max_center_drop = max(center_drops) if center_drops else float("nan")
    edge_drop_greater = (
        max_edge_drop > max_center_drop
        if np.isfinite(max_edge_drop) and np.isfinite(max_center_drop)
        else False
    )

    return {
        "original_probability": float(original_probability),
        "occluded_probabilities": occluded_probabilities,
        "probability_drops": probability_drops,
        "occlusion_map": occlusion_map,
        "max_drop": max_drop,
        "max_position": (max_y, max_x),
        "max_center": (center_y, center_x),
        "max_drop_is_edge": bool(max_drop_is_edge),
        "max_edge_drop": float(max_edge_drop),
        "max_center_drop": float(max_center_drop),
        "edge_drop_greater": bool(edge_drop_greater),
        "positions": positions
    }


print("\nPart 1 initialized successfully.")

# Run occlusion analysis, save heatmaps and CSV
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "occlusion_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "occlusion_summary.txt"
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}
CATEGORY_NAMES = ["TP", "TN", "FP", "FN"]

# Denormalize a tensor back to a displayable RGB image
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def tensor_to_display_image(image_tensor):
    image = image_tensor.detach().cpu()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()

# Predict the label of a single sample
device = next(model.parameters()).device
@torch.no_grad()
def predict_image(image_tensor):
    model.eval()
    logits = model(image_tensor.unsqueeze(0).to(device))
    if hasattr(logits, "logits"):logits = logits.logits
    probabilities = torch.softmax(logits, dim=1)
    pred_label = int(probabilities.argmax(dim=1).item())
    pneumonia_probability = float(probabilities[0, 1].item())
    return pred_label, pneumonia_probability
# Save a single occlusion heatmap
def save_occlusion_heatmap(image_tensor, occlusion_map, image_path, category, true_label, pred_label, pneumonia_probability, max_position, max_drop, output_path):
    original_image = tensor_to_display_image(image_tensor)
    heatmap = np.asarray(occlusion_map, dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original\nTrue: {CLASS_NAMES[true_label]}, Pred: {CLASS_NAMES[pred_label]}")
    axes[0].axis("off")
    im = axes[1].imshow(heatmap, cmap="jet", interpolation="nearest")
    axes[1].set_title(f"Occlusion Map\nP(PNEUMONIA)={pneumonia_probability:.4f}\nMax drop={max_drop:.4f}, position={max_position}")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(f"{category}: {Path(image_path).name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Check classification results
print("\n========== Dataset classification ==========")
for category in CATEGORY_NAMES:
    print(f"{category}: {len(categories.get(category, []))} images")
print("============================================\n")

# Run occlusion analysis for all four categories
results = []
summary_lines = []

summary_lines.append("Occlusion Sensitivity Analysis Summary")
summary_lines.append("=" * 60)
summary_lines.append(f"Model type: {MODEL_TYPE}")
summary_lines.append(f"Test dataset: {TEST_DIR}")
summary_lines.append(f"Device: {device}")
summary_lines.append(f"Random seed: {RANDOM_SEED}")
summary_lines.append(f"Patch size: {PATCH_SIZE}")
summary_lines.append(f"Stride: {STRIDE}")
summary_lines.append("")

for category in CATEGORY_NAMES:
    samples = categories.get(category, [])
    summary_lines.append("-" * 60)
    summary_lines.append(f"{category}")
    summary_lines.append("-" * 60)
    summary_lines.append(f"Total images in category: {len(samples)}")
    if len(samples) == 0:
        summary_lines.append("No images found in this category.")
        summary_lines.append("")
        print(f"[Warning] No images in {category}")
        continue
    save_count = min(SAVE_NUM, len(samples))
    selected_indices = set(random.sample(range(len(samples)), save_count))
    category_results = []
    for sample_idx, sample in enumerate(samples):
        dataset_index = int(sample)
        image_tensor, dataset_label = test_dataset[dataset_index]
        true_label = int(dataset_label)
        image_path = test_dataset.samples[dataset_index][0]
        pred_label, pneumonia_probability = predict_image(image_tensor)
        if dataset_label != true_label:
            raise ValueError(f"Label mismatch for {image_path}: category record={true_label}, dataset={dataset_label}")
        occlusion_result = occlusion_analysis(model, image_tensor)

        pneumonia_probability = float(occlusion_result["original_probability"])
        max_drop = float(occlusion_result["max_drop"])
        max_position = occlusion_result["max_position"]

        max_y = int(max_position[0])
        max_x = int(max_position[1])

        max_drop_is_edge = bool(occlusion_result["max_drop_is_edge"])
        max_edge_drop = float(occlusion_result["max_edge_drop"])
        max_center_drop = float(occlusion_result["max_center_drop"])
        edge_drop_greater = bool(occlusion_result["edge_drop_greater"])

        # CSV row
        row = {
            "Category": category,
            "Index": dataset_index,
            "Image Path": image_path,
            "True Label": CLASS_NAMES[true_label],
            "Pred Label": CLASS_NAMES[pred_label],
            "P(PNEUMONIA)": pneumonia_probability,
            "Max Drop": max_drop,
            "Max Y": max_y,
            "Max X": max_x,
            "Max Drop Is Edge": max_drop_is_edge,
            "Max Edge Drop": max_edge_drop,
            "Max Center Drop": max_center_drop,
            "Edge Drop > Center Drop": edge_drop_greater,
        }

        results.append(row)
        category_results.append(row)

        # Save heatmap only for randomly selected samples
        if sample_idx in selected_indices:
            safe_stem = Path(image_path).stem
            heatmap_path = HEATMAP_DIR / f"{category}_{dataset_index}_{safe_stem}_occlusion.png"
            save_occlusion_heatmap(
                image_tensor=image_tensor,
                occlusion_map=occlusion_result["occlusion_map"],
                image_path=image_path,
                category=category,
                true_label=true_label,
                pred_label=pred_label,
                pneumonia_probability=pneumonia_probability,
                max_position=max_position,
                max_drop=max_drop,
                output_path=heatmap_path,
            )

        if (sample_idx + 1) % 20 == 0 or (sample_idx + 1) == len(samples):
            print(f"[{category}] Processed {sample_idx + 1}/{len(samples)}")

    # Per-category statistics

    max_drops = np.array([r["Max Drop"] for r in category_results], dtype=float)
    edge_drops = np.array([r["Max Edge Drop"] for r in category_results], dtype=float)
    center_drops = np.array([r["Max Center Drop"] for r in category_results], dtype=float)
    edge_flags = np.array([r["Max Drop Is Edge"] for r in category_results], dtype=bool)
    edge_greater_flags = np.array([r["Edge Drop > Center Drop"] for r in category_results], dtype=bool)

    summary_lines.append(f"Occlusion analyses completed: {len(category_results)}")
    summary_lines.append(f"Mean max probability drop: {max_drops.mean():.6f}")
    summary_lines.append(f"Median max probability drop: {np.median(max_drops):.6f}")
    summary_lines.append(f"Mean edge drop: {edge_drops.mean():.6f}")
    summary_lines.append(f"Mean center drop: {center_drops.mean():.6f}")
    summary_lines.append(f"Max-drop location classified as edge: {edge_flags.sum()}/{len(edge_flags)} ({edge_flags.mean() * 100:.2f}%)")
    summary_lines.append(f"Edge drop > center drop: {edge_greater_flags.sum()}/{len(edge_greater_flags)} ({edge_greater_flags.mean() * 100:.2f}%)")
    summary_lines.append(f"Random heatmaps saved: {min(SAVE_NUM, len(samples))}")
    summary_lines.append("")

    print(f"\n[{category}] Complete: {len(category_results)} images; saved {min(SAVE_NUM, len(samples))} heatmaps.")

csv_fields = [
    "Category",
    "Index",
    "Image Path",
    "True Label",
    "Pred Label",
    "P(PNEUMONIA)",
    "Max Drop",
    "Max Y",
    "Max X",
    "Max Drop Is Edge",
    "Max Edge Drop",
    "Max Center Drop",
    "Edge Drop > Center Drop",
]

with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(results)

# Overall summary and TXT export
summary_lines.append("=" * 60)
summary_lines.append("Overall Summary")
summary_lines.append("=" * 60)
summary_lines.append(f"Total images analyzed: {len(results)}")

if results:
    all_max_drops = np.array([r["Max Drop"] for r in results], dtype=float)
    all_edge_drops = np.array([r["Max Edge Drop"] for r in results], dtype=float)
    all_center_drops = np.array([r["Max Center Drop"] for r in results], dtype=float)

    summary_lines.append(f"Overall mean max drop: {all_max_drops.mean():.6f}")
    summary_lines.append(f"Overall mean edge drop: {all_edge_drops.mean():.6f}")
    summary_lines.append(f"Overall mean center drop: {all_center_drops.mean():.6f}")

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")

print("\n========== Occlusion analysis finished ==========")
print(f"CSV:      {CSV_PATH.resolve()}")
print(f"Summary:  {SUMMARY_PATH.resolve()}")
print(f"Heatmaps: {HEATMAP_DIR.resolve()}")
print(f"Total analyzed images: {len(results)}")
print("=================================================")

# Summary statistics and visualization
if "results" not in globals() or len(results) == 0:
    raise RuntimeError("results is empty. Please run Part 2 successfully first.")

required_fields = {
    "Category",
    "Max Drop",
    "Max Edge Drop",
    "Max Center Drop",
    "Max Drop Is Edge",
    "Edge Drop > Center Drop",
}
missing_fields = required_fields - set(results[0].keys())

if missing_fields:
    raise KeyError(f"Missing fields in results: {sorted(missing_fields)}")

# Per-category statistics
summary_stats = []

for category in CATEGORY_NAMES:
    category_rows = [row for row in results if row["Category"] == category]

    if not category_rows:
        summary_stats.append({
            "Category": category,
            "N": 0,
            "Mean Max Drop": np.nan,
            "Median Max Drop": np.nan,
            "Std Max Drop": np.nan,
            "Mean Edge Drop": np.nan,
            "Mean Center Drop": np.nan,
            "Edge Max Location Percent": np.nan,
            "Edge Drop Greater Percent": np.nan,
        })
        continue

    max_drops = np.array([float(row["Max Drop"]) for row in category_rows])
    edge_drops = np.array([float(row["Max Edge Drop"]) for row in category_rows])
    center_drops = np.array([float(row["Max Center Drop"]) for row in category_rows])
    edge_location_flags = np.array([
        str(row["Max Drop Is Edge"]).strip().lower() in ("true", "1", "yes")
        for row in category_rows
    ])
    edge_greater_flags = np.array([
        str(row["Edge Drop > Center Drop"]).strip().lower() in ("true", "1", "yes")
        for row in category_rows
    ])
    summary_stats.append({
        "Category": category,
        "N": len(category_rows),
        "Mean Max Drop": float(np.mean(max_drops)),
        "Median Max Drop": float(np.median(max_drops)),
        "Std Max Drop": float(np.std(max_drops)),
        "Mean Edge Drop": float(np.mean(edge_drops)),
        "Mean Center Drop": float(np.mean(center_drops)),
        "Edge Max Location Percent": float(np.mean(edge_location_flags) * 100),
        "Edge Drop Greater Percent": float(np.mean(edge_greater_flags) * 100),
    })

print("\n" + "=" * 100)
print("OCCLUSION SUMMARY BY PREDICTION CATEGORY")
print("=" * 100)

header = (
    f"{'Category':<10}"
    f"{'N':>8}"
    f"{'Mean Max Drop':>18}"
    f"{'Median Max Drop':>19}"
    f"{'Mean Edge Drop':>18}"
    f"{'Mean Center Drop':>20}"
    f"{'Edge Max %':>13}"
)
print(header)
print("-" * len(header))

for row in summary_stats:
    if row["N"] == 0:
        print(f"{row['Category']:<10}{0:>8}")
        continue

    print(
        f"{row['Category']:<10}"
        f"{row['N']:>8}"
        f"{row['Mean Max Drop']:>18.6f}"
        f"{row['Median Max Drop']:>19.6f}"
        f"{row['Mean Edge Drop']:>18.6f}"
        f"{row['Mean Center Drop']:>20.6f}"
        f"{row['Edge Max Location Percent']:>12.2f}%"
    )

print("=" * 100)

stats_csv_path = OUTPUT_DIR / "occlusion_category_summary.csv"

summary_fields = [
    "Category",
    "N",
    "Mean Max Drop",
    "Median Max Drop",
    "Std Max Drop",
    "Mean Edge Drop",
    "Mean Center Drop",
    "Edge Max Location Percent",
    "Edge Drop Greater Percent",
]

with open(stats_csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_stats)

print(f"\nSaved summary CSV: {stats_csv_path.resolve()}")

# sample counts per category
counts = [sum(1 for row in results if row["Category"] == category) for category in CATEGORY_NAMES]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(CATEGORY_NAMES, counts)
ax.set_title("Number of Samples in Each Prediction Category")
ax.set_xlabel("Prediction Category")
ax.set_ylabel("Number of Images")

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_category_sample_counts.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# mean maximum probability drop per category
valid_stats = [row for row in summary_stats if row["N"] > 0]
labels = [row["Category"] for row in valid_stats]
means = [row["Mean Max Drop"] for row in valid_stats]
stds = [row["Std Max Drop"] for row in valid_stats]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, means, yerr=stds, capsize=5)

ax.set_title("Maximum Occlusion Probability Drop by Category")
ax.set_xlabel("Prediction Category")
ax.set_ylabel("Maximum Drop in P(PNEUMONIA)")
ax.axhline(0, linewidth=0.8)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_mean_max_drop.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# edge drop vs center drop
edge_means = [row["Mean Edge Drop"] for row in valid_stats]
center_means = [row["Mean Center Drop"] for row in valid_stats]
x = np.arange(len(valid_stats))
width = 0.36
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width / 2, edge_means, width, label="Edge Drop")
ax.bar(x + width / 2, center_means, width, label="Center Drop")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Prediction Category")
ax.set_ylabel("Mean Probability Drop")
ax.set_title("Edge vs Center Occlusion Sensitivity")
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_edge_vs_center.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# distribution of maximum drops
distribution_data = []

for category in CATEGORY_NAMES:
    values = [float(row["Max Drop"]) for row in results if row["Category"] == category]
    if values:
        distribution_data.append((category, values))

fig, ax = plt.subplots(figsize=(8, 5))

if distribution_data:
    ax.boxplot(
        [item[1] for item in distribution_data],
        tick_labels=[item[0] for item in distribution_data],
        showfliers=False,
    )

ax.set_title("Distribution of Maximum Occlusion Drops")
ax.set_xlabel("Prediction Category")
ax.set_ylabel("Maximum Drop in P(PNEUMONIA)")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_max_drop_distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Append final report
report_path = OUTPUT_DIR / "occlusion_summary.txt"

report_lines = [
    "",
    "",
    "=" * 60,
    "Part 3: Category-Level Summary",
    "=" * 60,
    f"Total analyzed images: {len(results)}",
    "",
]

for row in summary_stats:
    report_lines.append(f"Category: {row['Category']}")
    report_lines.append(f"  Sample count: {row['N']}")

    if row["N"] > 0:
        report_lines.append(f"  Mean max drop: {row['Mean Max Drop']:.6f}")
        report_lines.append(f"  Median max drop: {row['Median Max Drop']:.6f}")
        report_lines.append(f"  Std max drop: {row['Std Max Drop']:.6f}")
        report_lines.append(f"  Mean edge drop: {row['Mean Edge Drop']:.6f}")
        report_lines.append(f"  Mean center drop: {row['Mean Center Drop']:.6f}")
        report_lines.append(f"  Max-drop location classified as edge: {row['Edge Max Location Percent']:.2f}%")
        report_lines.append(f"  Edge drop greater than center drop: {row['Edge Drop Greater Percent']:.2f}%")

    report_lines.append("")

with open(report_path, "a", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"Summary CSV: {stats_csv_path.resolve()}")
print(f"Summary TXT: {report_path.resolve()}")
print("Generated plots:")

for filename in [
    "01_category_sample_counts.png",
    "02_mean_max_drop.png",
    "03_edge_vs_center.png",
    "04_max_drop_distribution.png",
]:
    print(f"  - {(OUTPUT_DIR / filename).resolve()}")

