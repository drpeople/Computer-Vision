import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Load SqueezeNet-based segmentation model from directory
model_path = "./squeezenet_segmentation_finetuned"  # Ensure this directory contains your saved model
class SqueezeNetSegmentation(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load SqueezeNet (no pretrained weights needed as we load the finetuned state)
        self.squeezenet = models.squeezenet1_1(pretrained=False)
        self.features = self.squeezenet.features  # Feature extractor
        # Segmentation head: 1x1 convolution to predict num_classes channels
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)  # shape: (B, num_classes, H_feat, W_feat)
        # Dynamically upsample to match input resolution
        up_logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return up_logits

model = SqueezeNetSegmentation(num_classes=2)
model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=device))
model.to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

# Ground truth threshold for mask binarization
GT_THRESHOLD = 0.05

# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH SCALING
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, base_image_size=(512, 512), scale_factor=1.0):
        """
        Args:
            images_dir: Directory with input images.
            masks_dir: Directory with corresponding segmentation masks.
            base_image_size: The base size to which images are first resized.
            scale_factor: Factor to further scale the image (and mask) dimensions.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.base_image_size = base_image_size
        self.scale_factor = scale_factor

        # Define transformations: first resize to base size then convert to tensor.
        self.image_transform = T.Compose([
            T.ToTensor()
        ])
        # For masks, we use nearest neighbor interpolation.
        self.mask_transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Open image and resize to the base size
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.base_image_size, Image.BILINEAR)
        # If scale_factor != 1.0, further resize the image
        if self.scale_factor != 1.0:
            new_size = (int(self.base_image_size[0] * self.scale_factor),
                        int(self.base_image_size[1] * self.scale_factor))
            image = image.resize(new_size, Image.BILINEAR)

        # Process corresponding mask
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.base_image_size, Image.NEAREST)
        if self.scale_factor != 1.0:
            new_size = (int(self.base_image_size[0] * self.scale_factor),
                        int(self.base_image_size[1] * self.scale_factor))
            mask = mask.resize(new_size, Image.NEAREST)

        # Convert mask to tensor and binarize
        label = self.mask_transform(mask).squeeze(0)
        label = (label > GT_THRESHOLD).float()

        # Convert image to tensor
        pixel_values = self.image_transform(image)

        return {"pixel_values": pixel_values, "labels": label}

# ---------------------------
# 3) INFERENCE FUNCTION
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    """Runs inference using mixed precision for efficiency."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    pred_masks, gt_masks = [], []

    model.eval()
    for batch in loader:
        # Move tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast():
            outputs = model(batch["pixel_values"])
            # Upsample logits to match ground truth mask dimensions if necessary
            logits = F.interpolate(outputs, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)  # shape: (B, H, W)

        gt_np = batch["labels"].cpu().numpy()
        for p, g in zip(preds.cpu().numpy(), gt_np):
            pred_masks.append(p)
            gt_masks.append(g)

    # Debug: print unique values of the first prediction and ground truth
    if len(pred_masks) > 0:
        print("Unique values in predicted mask (first image):", np.unique(pred_masks[0]))
        print("Unique values in ground truth mask (first image):", np.unique(gt_masks[0].astype(np.int32)))

    return pred_masks, gt_masks

# ---------------------------
# 4) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(pred_masks, gt_masks):
    pixel_accs = []
    ious = []
    # Compute per-image metrics and average them
    for pred, gt in zip(pred_masks, gt_masks):
        # Ensure binary masks (assuming class 1 is the leaf)
        pred_binary = (pred == 1).astype(np.int32)
        gt_binary = gt.astype(np.int32)

        # Pixel accuracy: fraction of matching pixels
        pixel_acc = np.mean(pred_binary == gt_binary)

        # Compute IoU: Intersection over Union for the leaf class
        intersection = np.sum((pred_binary == 1) & (gt_binary == 1))
        union = np.sum((pred_binary == 1) | (gt_binary == 1))
        iou = intersection / union if union > 0 else 0

        pixel_accs.append(pixel_acc)
        ious.append(iou)
    return np.mean(pixel_accs), np.mean(ious)

# ---------------------------
# 5) PLOTTING RESULTS
# ---------------------------
def plot_scale_sweep(scale_factors, pixel_accs, mious, save_path="squeezenet_scale_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(scale_factors, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(scale_factors, mious, marker='s', label='Mean IoU')
    plt.title("SqueezeNet Segmentation vs. Scale Factor")
    plt.xlabel("Scale Factor")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to: {save_path}")

# ---------------------------
# 6) MAIN SCRIPT
# ---------------------------
def main():
    # Define the scale factors to test (scaling the images up or down)
    scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]
    pixel_acc_list, miou_list = [], []

    for scale in scale_factors:
        print(f"\n=== Evaluating with scale_factor={scale} ===")
        dataset = LeafSegFineTuneDataset(
            images_dir=test_images_dir,
            masks_dir=test_masks_dir,
            base_image_size=(512, 512),
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
    results_df.to_csv("squeezenet_scale_results.csv", index=False)
    print("CSV saved: squeezenet_scale_results.csv")

if __name__ == "__main__":
    main()
