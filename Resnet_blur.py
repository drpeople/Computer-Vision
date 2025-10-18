import os
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image, ImageFilter
import requests
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd  # For CSV export


###############################################################################
# CUSTOM TRANSFORM CLASS (Picklable for multiprocessing)
###############################################################################
class GaussianBlurTransform:
    """
    Applies Gaussian Blur using a specified radius.
    Defined at the top level so it can be pickled by the DataLoader workers.
    """

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

# Load a pretrained ResNet-50 model from torchvision and move it to GPU
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

# Define the base folder path for the ImageNet images
base_folder = r"C:\Users\goker\PycharmProjects\DiplomProject\ILSVRC2012_img_val_subset"

# Load ImageNet class index mapping (list of 1000 classes)
imagenet_classes = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.splitlines()


###############################################################################
# DATASET DEFINITION
###############################################################################
class ImageDataset(Dataset):
    """
    Loads images from subfolders under 'base_folder', where each subfolder name
    is an integer label. Applies the provided 'transform' to each image.
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

        # Apply the transform
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
def plot_metrics_vs_blur(df, save_path="metrics_vs_blur.png"):
    """
    Plots accuracy, precision, recall, and F1-score vs. blur radius on a single plot.
    Saves the figure as 'metrics_vs_blur.png' by default.
    """
    plt.figure(figsize=(10, 6))

    # Plot each metric against the blur_radius
    plt.plot(df["blur_radius"], df["accuracy"], marker='o', label="Accuracy")
    plt.plot(df["blur_radius"], df["precision"], marker='s', label="Precision")
    plt.plot(df["blur_radius"], df["recall"], marker='^', label="Recall")
    plt.plot(df["blur_radius"], df["f1_score"], marker='d', label="F1-Score")

    plt.title("Model Performance vs. Blur Radius")
    plt.xlabel("Blur Radius")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])  # Metrics range from 0 to 1
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save and show
    plt.savefig(save_path, dpi=150)
    print(f"Metrics plot saved to: {save_path}")
    plt.show()


###############################################################################
# OPTIONAL: FUNCTION FOR VISUALIZING A FEW SAMPLES
###############################################################################
def visualize_results(image_paths, true_labels, predicted_labels, num_samples=5):
    """
    Randomly selects 'num_samples' images and shows them along with their
    true and predicted labels. This can help you inspect how the blur affects
    individual predictions.
    """
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
    # Define the different blur levels you want to test
    # same kind of blur as sigma_levels = [0, 1, 2, 3, 4]
    blur_levels = [0, 0.6, 1.2, 1.8, 2.4]

    # For storing each blur level's metrics in a table
    results_list = []

    # We'll iterate over each blur level and run inference on the entire dataset
    for blur_radius in blur_levels:
        print("==================================================")
        print(f"Running inference with blur_radius={blur_radius}")
        print("==================================================")

        # Define a transform pipeline that includes Gaussian blur in-memory
        transform_with_blur = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            GaussianBlurTransform(blur_radius),  # custom blur transform
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Create a dataset and dataloader with this specific blur level
        dataset = ImageDataset(base_folder, transform_with_blur)
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

        # Calculate metrics for this particular blur level
        metrics = calculate_metrics(all_true_labels, all_predicted_labels)

        # Print the metrics
        print("\n--- Metrics for blur_radius={:.2f} ---".format(blur_radius))
        print(f"Accuracy:      {metrics['accuracy'] * 100:.2f}%")
        print(f"Precision:     {metrics['precision']:.3f}")
        print(f"Recall:        {metrics['recall']:.3f}")
        print(f"F1-Score:      {metrics['f1_score']:.3f}")
        print("---------------------------------------\n")

        # Store metrics in results_list for CSV & plotting
        results_list.append({
            "blur_radius": blur_radius,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            # We won't store confusion_matrix in the CSV
            # If you want it, you can store it in a separate file or skip it for large class sets
        })

    # -------------------------------------------
    # 1) Save the summary metrics to CSV
    # -------------------------------------------
    df = pd.DataFrame(results_list)
    csv_path = "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics for all blur levels to: {csv_path}")

    # -------------------------------------------
    # 2) Create a "pretty" line plot of metrics vs. blur radius
    # -------------------------------------------
    plot_metrics_vs_blur(df, save_path="metrics_vs_blur.png")

    # Optional: If you want to visualize a few random predictions from
    # the *last* blur level, you can uncomment the following lines:
    #
    # visualize_results(image_paths=...,
    #                   true_labels=all_true_labels,
    #                   predicted_labels=all_predicted_labels,
    #                   num_samples=5)

    print("\nAll blur levels processed. End of script.")


###############################################################################
# ENTRY POINT in this code i inference resnet and pass it through blur and get results
###############################################################################
if __name__ == "__main__":
    main()
