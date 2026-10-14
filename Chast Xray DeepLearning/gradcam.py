import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "ResNet18"
MODEL_TYPE = "resnet18"  # "resnet18" | "densenet121" | "inception_v3"
MODEL_PATH = r"best_model_resnet18.pth"
TEST_DIR = r"chest_xray/test"
IMAGE_SIZE = 224

# Transform & Dataset
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
print(dataset.class_to_idx)

# Load Model
def build_model():
    if MODEL_TYPE == "resnet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(512, 2)
    elif MODEL_TYPE == "densenet121":
        model = torchvision.models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, 2)
    elif MODEL_TYPE == "inception_v3":
        model = torchvision.models.inception_v3(weights=None, aux_logits=False)
        model.fc = nn.Linear(2048, 2)
    return model

model = build_model()

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()
print("Model loaded")

# Target Layer for Grad-CAM
if MODEL_TYPE == "resnet18":
    TARGET_LAYER = model.layer4[-1]
elif MODEL_TYPE == "densenet121":
    TARGET_LAYER = model.features.denseblock4
elif MODEL_TYPE == "inception_v3":
    TARGET_LAYER = model.Mixed_7c

# 5. Grad-CAM
def generate_gradcam(model, image_tensor, target_layer, target_class):

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.clone().detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(image_tensor)
    score = output[0, target_class]
    score.backward()

    h1.remove()
    h2.remove()

    activation = activations[0]
    gradient = gradients[0]

    # Global-average-pooled gradient weights over the activation maps
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))

    return cam

# Find TP / TN / FP / FN
TP, TN, FP, FN = [], [], [], []

for i in range(len(dataset)):

    img, label = dataset[i]
    x = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()

    if label == 1 and pred == 1:
        TP.append(i)
    elif label == 0 and pred == 0:
        TN.append(i)
    elif label == 0 and pred == 1:
        FP.append(i)
    elif label == 1 and pred == 0:
        FN.append(i)

print("TP", len(TP))
print("TN", len(TN))
print("FP", len(FP))
print("FN", len(FN))

SAVE_DIR = os.path.join("GradCAM_results", MODEL_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_category(indices, category):

    if len(indices) == 0:
        print(category, "not available")
        return

    np.random.seed(42)
    indices = np.random.choice(indices, min(6, len(indices)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for ax, index in zip(axes, indices):

        img, label = dataset[index]
        x = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(x)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()

        cam = generate_gradcam(model, x, TARGET_LAYER, pred)

        image_path = dataset.samples[index][0]
        original = Image.open(image_path).convert("L")
        original = original.resize((IMAGE_SIZE, IMAGE_SIZE))
        original = np.array(original)

        ax.imshow(original, cmap="gray")

        heatmap = plt.cm.jet(cam)[:, :, :3]
        ax.imshow(heatmap, alpha=0.30)

        ax.set_title(f"True={label}\nPred={pred}\nP={prob[0, 1]:.3f}")
        ax.axis("off")

    plt.suptitle(MODEL_NAME + " - " + category)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, category + ".png"), dpi=300)
    plt.close()

plot_category(TP, "TP")
plot_category(TN, "TN")
plot_category(FP, "FP")
plot_category(FN, "FN")

print("Grad-CAM finished")