import os
import torch
from torchvision import models, transforms
# Import ResNeXt50_32x4d weights instead of DenseNet121 weights
from torchvision.models import ResNeXt50_32X4D_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image, ImageFilter
import requests
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

###############################################################################
# CUSTOM TRANSFORM CLASS (Picklable for multiprocessing)
###############################################################################
class GaussianBlurTransform:
    def __init__(self, radius=0.0):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))

###############################################################################
# SETUP
###############################################################################
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable cuDNN auto-tuner for performance optimization
torch.backends.cudnn.benchmark = True

# Load a pretrained ResNeXt50_32x4d model using its recommended weights and move it to GPU
weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1
model = models.resnext50_32x4d(weights=weights).to(device)
model.eval()

# Define the base folder path for the ImageNet images
base_folder = r"C:\\Users\\goker\\PycharmProjects\\DiplomProject\\ILSVRC2012_img_val_subset"

# Load ImageNet class index mapping (list of 1000 classes)
imagenet_classes = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.splitlines()

# Use normalization parameters from the ResNeXt50 weights metadata (fallback to defaults)
norm_mean = weights.meta.get("mean", (0.5, 0.5, 0.5))
norm_std = weights.meta.get("std", (0.5, 0.5, 0.5))

###############################################################################
# DATASET DEFINITION
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
                    if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(subfolder_path, image_name))
                        self.labels.append(true_label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, image_path

###############################################################################
# METRICS CALCULATION
###############################################################################
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }
    return metrics_dict

###############################################################################
# PLOTTING: PRETTY VISUALIZATION FOR METRICS
###############################################################################
def plot_metrics_vs_blur(df, save_path="ResNeXt_metrics_vs_blur.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(df["blur_radius"], df["accuracy"], marker='o', label="Accuracy")
    plt.plot(df["blur_radius"], df["precision"], marker='s', label="Precision")
    plt.plot(df["blur_radius"], df["recall"], marker='^', label="Recall")
    plt.plot(df["blur_radius"], df["f1_score"], marker='d', label="F1-Score")

    plt.title("Model Performance vs. Blur Radius")
    plt.xlabel("Blur Radius")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    print(f"Metrics plot saved to: {save_path}")
    plt.show()

###############################################################################
# VISUALIZATION
###############################################################################
def visualize_results(image_paths, true_labels, predicted_labels, num_samples=5):
    results = list(zip(image_paths, true_labels, predicted_labels))
    random_samples = random.sample(results, min(num_samples, len(results)))
    for image_path, true_label_idx, predicted_label_idx in random_samples:
        image = Image.open(image_path).convert("RGB")
        true_label = imagenet_classes[true_label_idx]
        predicted_label = imagenet_classes[predicted_label_idx]

        plt.imshow(image)
        title = f"True: {true_label} | Pred: {predicted_label}"
        plt.title(title)
        plt.axis("off")
        plt.show()

###############################################################################
# MAIN ROUTINE
###############################################################################
def main():
    # same kind of blur as sigma_levels = [0, 1, 2, 3, 4]
    blur_levels = [0, 0.6, 1.2, 1.8, 2.4]
    results_list = []

    for blur_radius in blur_levels:
        print("==================================================")
        print(f"Running inference with blur_radius={blur_radius}")
        print("==================================================")

        transform_with_blur = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            GaussianBlurTransform(blur_radius),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

        dataset = ImageDataset(base_folder, transform_with_blur)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

        all_true_labels = []
        all_predicted_labels = []

        for batch_images, batch_labels, image_paths in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(batch_images)

            _, predicted_indices = torch.max(outputs, 1)

            all_true_labels.extend(batch_labels.cpu().numpy())
            all_predicted_labels.extend(predicted_indices.cpu().numpy())

        metrics = calculate_metrics(all_true_labels, all_predicted_labels)

        print("\n--- Metrics for blur_radius={:.2f} ---".format(blur_radius))
        print(f"Accuracy:      {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision:     {metrics['precision']:.3f}")
        print(f"Recall:        {metrics['recall']:.3f}")
        print(f"F1-Score:      {metrics['f1_score']:.3f}")
        print("---------------------------------------\n")

        results_list.append({
            "blur_radius": blur_radius,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        })

    df = pd.DataFrame(results_list)
    csv_path = "ResNeXt_blur_results.csv"  # Updated filename for ResNeXt usage
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics for all blur levels to: {csv_path}")

    plot_metrics_vs_blur(df, save_path="ResNeXt_metrics_vs_blur.png")
    print("\nAll blur levels processed. End of script.")

if __name__ == "__main__":
    main()
