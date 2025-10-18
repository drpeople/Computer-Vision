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
import segmentation_models_pytorch as smp
import torch.nn as nn

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Path to your saved U-Net model (trained earlier)
model_path = "./finetuned_unet_leaf.pth"

# Load U-Net Model with ResNet34 encoder (binary segmentation: classes=1)
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1  # Single-channel output (logits)
)

# Load the state dict
try:
    state_dict = torch.load(model_path, map_location=device)
except Exception as e:
    print(f"Error loading state dict: {e}")
    sys.exit(1)

model.load_state_dict(state_dict, strict=False)
model = model.float()
model.to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH SCALING
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    """
    Loads the original PIL image (for plotting) and processes it for U-Net.
    Applies scaling to the image and mask based on a scale factor.
    Converts the ground truth mask to a binary mask (0 for background, 1 for leaf).
    """
    def __init__(self, images_dir, masks_dir, base_image_size=(544, 544), scale_factor=1.0):
        """
        Args:
            images_dir: Directory with input images.
            masks_dir: Directory with segmentation masks.
            base_image_size: The size to which images are first resized.
            scale_factor: Factor to scale the image and mask dimensions.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.base_image_size = base_image_size
        self.scale_factor = scale_factor

        # Transformation that converts images back to base size, normalizes, and converts to tensor.
        self.image_transform = T.Compose([
            T.Resize(base_image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Transformation for the mask: resize back to base size and convert to tensor.
        self.mask_transform = T.Compose([
            T.Resize(base_image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Open image and resize to base size
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.base_image_size, Image.BILINEAR)

        # If scale_factor != 1, scale the image dimensions
        if self.scale_factor != 1.0:
            new_size = (int(self.base_image_size[0] * self.scale_factor),
                        int(self.base_image_size[1] * self.scale_factor))
            image = image.resize(new_size, Image.BILINEAR)

        # Process corresponding mask; assume mask has same base name with .png extension
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.base_image_size, Image.NEAREST)

        if self.scale_factor != 1.0:
            new_size = (int(self.base_image_size[0] * self.scale_factor),
                        int(self.base_image_size[1] * self.scale_factor))
            mask = mask.resize(new_size, Image.NEAREST)

        # Apply transforms: the resize in the transform brings the scaled image back to base_image_size.
        pixel_values = self.image_transform(image)

        # Process mask: squeeze extra channel and binarize (values > 0 become 1)
        label = self.mask_transform(mask).squeeze(0)
        label = (label > 0).float()

        return {"pixel_values": pixel_values, "labels": label}


# ---------------------------
# 3) INFERENCE FUNCTION
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    """Runs inference with mixed precision for efficiency."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    pred_masks, gt_masks = [], []

    model.eval()
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(pixel_values)  # Outputs shape: (B, 1, H, W)
            preds = torch.sigmoid(outputs)
            pred_maps = (preds > 0.5).long().squeeze(1)

        for pred in pred_maps:
            pred_masks.append(pred.cpu().numpy())
        for label in batch["labels"]:
            gt_masks.append(label.cpu().numpy())

    return pred_masks, gt_masks


# ---------------------------
# 4) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(pred_masks, gt_masks):
    pixel_accs = []
    ious = []
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
# 5) PLOTTING: Metrics vs. Scale Factor
# ---------------------------
def plot_scale_sweep(scale_factors, pixel_accs, mious, save_path="unet_scale_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(scale_factors, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(scale_factors, mious, marker='s', label='Mean IoU')
    plt.title("U-Net Segmentation vs. Scale Factor")
    plt.xlabel("Scale Factor")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to: {save_path}")


# ---------------------------
# 6) MAIN SCRIPT
# ---------------------------
def main():
    # Define the scale factors to test.
    scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]
    pixel_acc_list, miou_list = [], []

    for scale in scale_factors:
        print(f"\n=== Evaluating with scale_factor={scale} ===")
        # Create dataset with the current scale factor and base image size
        dataset = LeafSegFineTuneDataset(
            images_dir=test_images_dir,
            masks_dir=test_masks_dir,
            base_image_size=(544, 544),
            scale_factor=scale
        )

        preds, gts = run_inference(dataset)
        pix_acc, miou = compute_metrics(preds, gts)

        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)

        print(f"Results (scale_factor={scale}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    plot_scale_sweep(scale_factors, pixel_acc_list, miou_list)

    results_df = pd.DataFrame({
        "scale_factor": scale_factors,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    csv_path = "unet_scale_results.csv"
    results_df.to_csv(csv_path, index=False)
    print("CSV saved:", csv_path)


if __name__ == "__main__":
    main()
