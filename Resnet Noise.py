import os
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
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
class GaussianNoiseTransform:
    """
    Applies Gaussian noise with a given standard deviation (in normalized [0,1] domain)
    to a PIL image. Defined at the module level so it can be pickled by DataLoader workers.
    """
    def __init__(self, noise_std=0.0):
        self.noise_std = noise_std

    def __call__(self, img):
        # Convert PIL image to NumPy array (float32) and scale to [0, 1]
        np_img = np.array(img, dtype=np.float32) / 255.0
        # Generate Gaussian noise (normalized)
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=np_img.shape)
        # Add noise and clip to [0, 1]
        np_noisy = np.clip(np_img + noise, 0, 1)
        # Scale back to [0, 255] and convert to uint8
        np_noisy = (np_noisy * 255).astype(np.uint8)
        # Convert NumPy array back to PIL image
        noisy_img = Image.fromarray(np_noisy)
        return noisy_img


###############################################################################
# SETUP
###############################################################################
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable cuDNN auto-tuner for performance optimization
torch.backends.cudnn.benchmark = True

# Load a pretrained ResNet-50 model from torchvision and move it to GPU
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

# Define the base folder path for the ImageNet images (or a subset)
base_folder = r"C:\Users\goker\PycharmProjects\DiplomProject\ILSVRC2012_img_val_subset"

# Load ImageNet class index mapping
imagenet_classes = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.splitlines()


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
                # Convert folder name to integer label
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

        # Apply the transform (including noise)
        if self.transform is not None:
            image = self.transform(image)

        return image, label, image_path


###############################################################################
# METRICS CALCULATION
###############################################################################
def calculate_metrics(y_true, y_pred):
    """
    Returns a dictionary of computed metrics and the confusion matrix.
    """
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
def plot_metrics_vs_noise(df, save_path="metrics_vs_noise.png"):
    """
    Plots accuracy, precision, recall, and F1-score vs. noise level on a single plot.
    Saves the figure as 'metrics_vs_noise.png' by default.
    """
    plt.figure(figsize=(10, 6))

    # Plot each metric against the noise_std
    plt.plot(df["noise_std"], df["accuracy"], marker='o', label="Accuracy")
    plt.plot(df["noise_std"], df["precision"], marker='s', label="Precision")
    plt.plot(df["noise_std"], df["recall"], marker='^', label="Recall")
    plt.plot(df["noise_std"], df["f1_score"], marker='d', label="F1-Score")

    plt.title("Model Performance vs. Noise Standard Deviation")
    plt.xlabel("Noise Std (normalized)")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])  # All metrics range from 0 to 1
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save and show
    plt.savefig(save_path, dpi=150)
    print(f"Metrics plot saved to: {save_path}")
    plt.show()


###############################################################################
# MAIN ROUTINE
###############################################################################
def main():
    # Define the noise levels in the normalized domain (e.g., 0.0, 0.05, 0.1, etc.)
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # For storing each noise level's metrics in a table
    results_list = []

    # We'll iterate over each noise level and run inference on the entire dataset
    for std in noise_levels:
        print("==================================================")
        print(f"Running inference with noise_std={std}")
        print("==================================================")

        # Define a transform pipeline that includes Gaussian noise in-memory
        transform_with_noise = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            GaussianNoiseTransform(noise_std=std),  # custom noise transform operating in normalized domain
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Create a dataset and dataloader with this specific noise level
        dataset = ImageDataset(base_folder, transform_with_noise)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        # Variables to store ground truth and predictions for the whole dataset
        all_true_labels = []
        all_predicted_labels = []

        # Inference loop
        for batch_images, batch_labels, image_paths in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            with torch.no_grad():
                # Use mixed precision for faster inference (if supported)
                with torch.cuda.amp.autocast():
                    outputs = model(batch_images)

            # Get the predicted class indices
            _, predicted_indices = torch.max(outputs, 1)

            # Accumulate for final metrics
            all_true_labels.extend(batch_labels.cpu().numpy())
            all_predicted_labels.extend(predicted_indices.cpu().numpy())

        # Calculate metrics for this particular noise level
        metrics = calculate_metrics(all_true_labels, all_predicted_labels)

        # Print the metrics
        print("\n--- Metrics for noise_std={:.2f} ---".format(std))
        print(f"Accuracy:      {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision:     {metrics['precision']:.3f}")
        print(f"Recall:        {metrics['recall']:.3f}")
        print(f"F1-Score:      {metrics['f1_score']:.3f}")
        print("---------------------------------------\n")

        # Store metrics for CSV & plotting
        results_list.append({
            "noise_std": std,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        })

    # Save the summary metrics to CSV
    df = pd.DataFrame(results_list)
    csv_path = "Resnet-noise_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics for all noise levels to: {csv_path}")

    # Create a "pretty" line plot of metrics vs. noise
    plot_metrics_vs_noise(df, save_path="metrics_vs_noise.png")

    print("\nAll noise levels processed. End of script.")


###############################################################################
# ENTRY POINT
###############################################################################
if __name__ == "__main__":
    main()
