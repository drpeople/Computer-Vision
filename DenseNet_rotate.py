import os
import torch
from torchvision import models, transforms
# Import DenseNet121 weights instead of ConvNeXt weights
from torchvision.models import DenseNet121_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import requests
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd  # For CSV export

###############################################################################
# CUSTOM TRANSFORM CLASS (Picklable for multiprocessing on Windows)
###############################################################################
class RotationTransform:
    """
    Applies a rotation (in degrees) using PIL's built-in rotate.
    We define it at the module level so that it's picklable
    by the DataLoader workers (important for Windows).
    """
    def __init__(self, angle=0):
        self.angle = angle

    def __call__(self, img):
        return img.rotate(self.angle, expand=True)

###############################################################################
# SETUP
###############################################################################
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable cuDNN auto-tuner for performance optimization
torch.backends.cudnn.benchmark = True

# Load a pretrained DenseNet121 model using its recommended weights and move it to GPU
weights = DenseNet121_Weights.IMAGENET1K_V1
model = models.densenet121(weights=weights).to(device)
model.eval()

# Define the base folder path for the ImageNet images (subset or otherwise)
base_folder = r"C:\Users\goker\PycharmProjects\DiplomProject\ILSVRC2012_img_val_subset"

# Load ImageNet class index mapping
imagenet_classes = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.splitlines()

# Use normalization parameters from the DenseNet121 weights metadata (or fallback to defaults)
norm_mean = weights.meta.get("mean", (0.5, 0.5, 0.5))
norm_std = weights.meta.get("std", (0.5, 0.5, 0.5))

###############################################################################
# DATASET DEFINITION
###############################################################################
class ImageDataset(Dataset):
    """
    Loads images from subfolders under 'base_folder', where each subfolder name
    is the integer label. Applies the provided 'transform' to each image.
    """
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
def plot_metrics_vs_angle(df, save_path="DenseNet_metrics_vs_angle.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(df["angle"], df["accuracy"], marker='o', label="Accuracy")
    plt.plot(df["angle"], df["precision"], marker='s', label="Precision")
    plt.plot(df["angle"], df["recall"], marker='^', label="Recall")
    plt.plot(df["angle"], df["f1_score"], marker='d', label="F1-Score")

    plt.title("Model Performance vs. Rotation Angle")
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    print(f"Metrics plot saved to: {save_path}")
    plt.show()

###############################################################################
# OPTIONAL: FUNCTION FOR VISUALIZING A FEW SAMPLES
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
    angles = list(range(0, 181, 30))
    results_list = []

    for angle in angles:
        print("==================================================")
        print(f"Running inference with angle={angle}°")
        print("==================================================")

        transform_with_rotation = transforms.Compose([
            RotationTransform(angle=angle),
            transforms.Resize(300),  # Increased size to preserve more details
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

        dataset = ImageDataset(base_folder, transform_with_rotation)
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

        print(f"\n--- Metrics for angle={angle}° ---")
        print(f"Accuracy:      {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision:     {metrics['precision']:.3f}")
        print(f"Recall:        {metrics['recall']:.3f}")
        print(f"F1-Score:      {metrics['f1_score']:.3f}")
        print("---------------------------------------\n")

        results_list.append({
            "angle": angle,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        })

    df = pd.DataFrame(results_list)
    csv_path = "DenseNet_rotation_results.csv"  # Updated filename for DenseNet usage
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics for all rotation angles to: {csv_path}")

    plot_metrics_vs_angle(df, save_path="DenseNet_metrics_vs_angle.png")
    print("\nAll rotation angles processed. End of script.")

if __name__ == "__main__":
    main()
