import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
from torchvision.transforms import functional as TF
import segmentation_models_pytorch as smp
import torch.nn as nn

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Path to your saved DeepLabV3+ model
model_path = "./finetuned_deeplabv3plus_leaf.pth"

# Load DeepLabV3+ Model with ResNet101 encoder
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    encoder_weights=None,  # Load custom-trained weights
    in_channels=3,
    classes=2
)

# Load the state dict
try:
    state_dict = torch.load(model_path, map_location=device)
except Exception as e:
    print(f"Error loading state dict: {e}")
    sys.exit(1)

model.load_state_dict(state_dict, strict=False)

# Ensure model is properly set up
model = model.float()
model.to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"


# ---------------------------
# 2) LEAF SEGMENTATION DATASET
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    """
    Loads the original PIL image (for plotting) and processes it for DeepLabV3+.
    Converts the ground truth mask to a binary mask (0 for background, 1 for leaf).
    """

    def __init__(self, images_dir, masks_dir, image_size=(352, 352), sigma=0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.image_size = image_size
        self.sigma = sigma  # Gaussian blur sigma

        # Transformations for the image
        self.image_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Transformations for the mask
        self.mask_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        # Apply resizing and optional blur
        image = image.resize(self.image_size, Image.BILINEAR)
        if self.sigma > 0:
            image = apply_blur(image, self.sigma)

        # Load mask; assume corresponding mask has same base name with .png extension
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.image_size, Image.NEAREST)

        # Process the image
        pixel_values = self.image_transform(image)

        # Convert mask to tensor and binarize (values > 0 become 1)
        label = self.mask_transform(mask).squeeze(0)
        label = (label > 0).float()

        return {"pixel_values": pixel_values, "labels": label}


# ---------------------------
# 3) TRANSFORM WITH BLUR OPTION
# ---------------------------
def apply_blur(image, sigma):
    """Applies Gaussian blur if sigma > 0."""
    if sigma > 0:
        return TF.gaussian_blur(image, kernel_size=5, sigma=sigma)
    return image


# ---------------------------
# 4) INFERENCE FUNCTION
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    """Runs inference with mixed precision for efficiency."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    pred_masks, gt_masks = [], []

    model.eval()
    for batch in loader:
        # Move pixel_values to device
        pixel_values = batch["pixel_values"].to(device)

        with autocast():
            outputs = model(pixel_values)  # No ["out"], DeepLabV3+ returns tensor directly
            pred_maps = torch.argmax(outputs, dim=1)  # Get predicted class (0 or 1)

        # For binary segmentation, assume label 1 corresponds to "leaf"
        for pred in pred_maps:
            pred_mask = (pred == 1).long()  # Binary mask: 1 if predicted class is leaf, else 0
            pred_masks.append(pred_mask.cpu().numpy())

        # Ground truth masks are already binarized (on CPU)
        for label in batch["labels"]:
            gt_masks.append(label.cpu().numpy())

    return pred_masks, gt_masks


# ---------------------------
# 5) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(pred_masks, gt_masks):
    pixel_accs = []
    ious = []
    # Compute per-image metrics and average
    for pred, gt in zip(pred_masks, gt_masks):
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
        pixel_acc = np.diag(cm).sum() / np.sum(cm)
        iou = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0] + 1e-10)
        pixel_accs.append(pixel_acc)
        ious.append(iou)
    return np.mean(pixel_accs), np.mean(ious)


# ---------------------------
# 6) PLOTTING RESULTS
# ---------------------------
def plot_blur_sweep(blur_levels, pixel_accs, mious, save_path="deeplabv3plus_blur_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(blur_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(blur_levels, mious, marker='s', label='Mean IoU')
    plt.title("DeepLabV3+ Segmentation vs. Gaussian Blur Sigma")
    plt.xlabel("Blur Sigma")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to: {save_path}")


# ---------------------------
# 7) MAIN SCRIPT
# ---------------------------
def main():
    blur_levels = [0, 1, 2, 3, 4]
    pixel_acc_list, miou_list = [], []

    for sigma in blur_levels:
        print(f"\n=== Evaluating with blur sigma={sigma} ===")
        # Create dataset with the current blur sigma
        dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, image_size=(352, 352), sigma=sigma)

        preds, gts = run_inference(dataset)
        pix_acc, miou = compute_metrics(preds, gts)

        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)

        print(f"Results (sigma={sigma}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    plot_blur_sweep(blur_levels, pixel_acc_list, miou_list)

    results_df = pd.DataFrame({
        "sigma": blur_levels,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    results_df.to_csv("deeplabv3plus_blur_results.csv", index=False)
    print("CSV saved: deeplabv3plus_blur_results.csv")


if __name__ == "__main__":
    main()
