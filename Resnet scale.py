import os
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import requests

###############################################################################
# CUSTOM TRANSFORM CLASS FOR SCALING
###############################################################################
class ScaleTransform:
    def __init__(self, scale_factor=1.0, interpolation=Image.BICUBIC):
        self.scale_factor = scale_factor
        self.interpolation = interpolation

    def __call__(self, img):
        width, height = img.size
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        return img.resize((new_width, new_height), self.interpolation)

###############################################################################
# SETUP
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True

weights = ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights).to(device)
model.eval()

base_folder = r"C:\Users\goker\PycharmProjects\DiplomProject\ILSVRC2012_img_val_subset"

imagenet_classes = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.splitlines()

norm_mean = weights.meta.get("mean", (0.485, 0.456, 0.406))
norm_std = weights.meta.get("std", (0.229, 0.224, 0.225))

###############################################################################
# DATASET
###############################################################################
class ImageDataset(Dataset):
    def __init__(self, base_folder, transform):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for subfolder in os.listdir(base_folder):
            subfolder_path = os.path.join(base_folder, subfolder)
            if os.path.isdir(subfolder_path):
                true_label_idx = int(subfolder)
                for image_name in os.listdir(subfolder_path):
                    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(subfolder_path, image_name))
                        self.labels.append(true_label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, image_path

###############################################################################
# METRICS
###############################################################################
def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

###############################################################################
# PLOTTING
###############################################################################
def plot_metrics_vs_scale(df, save_path="ResNet_fixed_metrics_vs_scale.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(df["scale_factor"], df["accuracy"], marker='o', label="Accuracy")
    plt.plot(df["scale_factor"], df["precision"], marker='s', label="Precision")
    plt.plot(df["scale_factor"], df["recall"], marker='^', label="Recall")
    plt.plot(df["scale_factor"], df["f1_score"], marker='d', label="F1-Score")
    plt.title("ResNet-50 Performance vs. Scale Factor (Fixed)")
    plt.xlabel("Scale Factor")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to: {save_path}")

###############################################################################
# MAIN
###############################################################################
def main():
    scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]
    results_list = []

    for sf in scale_factors:
        print("=" * 50)
        print(f"Running inference with scale_factor={sf}")
        print("=" * 50)

        transform_with_scale = transforms.Compose([
            ScaleTransform(scale_factor=sf, interpolation=Image.BICUBIC),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

        dataset = ImageDataset(base_folder, transform_with_scale)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

        all_true, all_preds = [], []

        for images, labels, paths in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(images)

            _, preds = torch.max(outputs, 1)
            all_true.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        metrics = calculate_metrics(all_true, all_preds)
        print(f"\n--- Metrics for scale_factor={sf} ---")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}\n")

        results_list.append({
            "scale_factor": sf,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        })

    df = pd.DataFrame(results_list)
    df.to_csv("ResNet_scale_results.csv", index=False)
    plot_metrics_vs_scale(df)

if __name__ == "__main__":
    main()
