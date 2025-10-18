import os
import torch
import timm  # Ensure timm is installed: pip install timm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # For CSV export

###############################################################################
# CUSTOM TRANSFORM CLASS: GAUSSIAN NOISE (normalized approach)
###############################################################################
class GaussianNoiseTransform:
    def __init__(self, noise_std=0.0):
        self.noise_std = noise_std

    def __call__(self, img):
        # Convert PIL image to numpy array and scale to [0, 1]
        np_img = np.array(img, dtype=np.float32) / 255.0
        # Generate Gaussian noise with mean 0 in the normalized domain
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=np_img.shape).astype(np.float32)
        np_noisy = np_img + noise
        # Clip the noisy image to [0, 1]
        np_noisy = np.clip(np_noisy, 0.0, 1.0)
        # Scale back to [0, 255] and convert to uint8
        np_noisy = (np_noisy * 255).astype(np.uint8)
        return Image.fromarray(np_noisy)

###############################################################################
# SETUP
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable cuDNN auto-tuner for performance optimization
torch.backends.cudnn.benchmark = True

# Load a pretrained DeiT model from timm and move it to GPU.
# Here we use 'deit_base_patch16_224' as an example.
model = timm.create_model('deit_base_patch16_224', pretrained=True).to(device)
model.eval()

# Define the base folder path for the ImageNet images (or a subset)
base_folder = r"C:\Users\goker\PycharmProjects\DiplomProject\ILSVRC2012_img_val_subset"

# Load ImageNet class index mapping (list of 1000 classes)
imagenet_classes = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.splitlines()

# Use standard ImageNet normalization parameters for DeiT
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

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
def plot_metrics_vs_noise(df, save_path="DeiT_metrics_vs_noise.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(df["noise_std"], df["accuracy"], marker='o', label="Accuracy")
    plt.plot(df["noise_std"], df["precision"], marker='s', label="Precision")
    plt.plot(df["noise_std"], df["recall"], marker='^', label="Recall")
    plt.plot(df["noise_std"], df["f1_score"], marker='d', label="F1-Score")
    plt.title("Model Performance vs. Noise Standard Deviation")
    plt.xlabel("Noise Std (in normalized [0,1] domain)")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Metrics plot saved to: {save_path}")
    plt.show()

###############################################################################
# MAIN ROUTINE
###############################################################################
def main():
    # Update noise levels to normalized values (for example, 0.0 to 0.3)
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results_list = []

    for std in noise_levels:
        print(f"Running inference with noise_std={std}")
        transform_with_noise = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            GaussianNoiseTransform(noise_std=std),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

        dataset = ImageDataset(base_folder, transform_with_noise)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

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
        print(f"Metrics for noise_std={std}: {metrics}")
        results_list.append({
            "noise_std": std,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        })

    df = pd.DataFrame(results_list)
    csv_path = "DeiT_noise_results.csv"  # Updated filename for DeiT usage
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics for all noise levels to: {csv_path}")
    plot_metrics_vs_noise(df)

if __name__ == "__main__":
    main()
