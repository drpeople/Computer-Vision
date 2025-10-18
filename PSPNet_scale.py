import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import autocast
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Load PSPNet fine-tuned model from directory.
model_dir = "./pspnet_finetuned_leaf"  # Ensure this path exists and contains your saved model weights
model_weights_path = os.path.join(model_dir, "model.pth")

# Recreate the PSPNet model with the same parameters used during training.
model = smp.PSPNet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    classes=2,
    activation=None  # Use raw logits
)
model.load_state_dict(torch.load(model_weights_path, map_location=device))
model.to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

# Ground truth threshold for mask binarization (adjust if needed)
GT_THRESHOLD = 0.05

# Define image transformation matching training: convert to tensor and normalize.
# Note: The resizing is handled in the dataset.
image_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225])  # ImageNet stds
])

# Base image size (width, height)
BASE_IMAGE_SIZE = (512, 512)


# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH SCALING
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform, base_image_size=BASE_IMAGE_SIZE, scale_factor=1.0):
        """
        Args:
            images_dir: Directory with input images.
            masks_dir: Directory with corresponding segmentation masks.
            image_transform: Transformations to apply to the image.
            base_image_size: The size to which images are first resized.
            scale_factor: Factor to scale the image (and mask) dimensions.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.image_transform = image_transform
        self.base_image_size = base_image_size
        self.scale_factor = scale_factor

        # Define a mask transformation: simply convert to tensor.
        self.mask_transform = T.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Open image and resize to the base size
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.base_image_size, Image.BILINEAR)

        # If scale_factor != 1, further resize image by the scale factor and adjust to be divisible by 8
        if self.scale_factor != 1.0:
            # Compute raw scaled size
            raw_width = self.base_image_size[0] * self.scale_factor
            raw_height = self.base_image_size[1] * self.scale_factor
            # Round up to the nearest multiple of 8
            new_width = int(math.ceil(raw_width / 8) * 8)
            new_height = int(math.ceil(raw_height / 8) * 8)
            new_size = (new_width, new_height)
            image = image.resize(new_size, Image.BILINEAR)
        else:
            new_size = self.base_image_size

        # Process corresponding mask
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.base_image_size, Image.NEAREST)
        if self.scale_factor != 1.0:
            mask = mask.resize(new_size, Image.NEAREST)

        # Convert mask to tensor and binarize (assuming nonzero pixels represent the leaf)
        label = self.mask_transform(mask).squeeze(0)
        label = (label > 0).float()

        # Apply the image transformation (normalization happens here)
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
            logits = model(batch["pixel_values"])  # shape: (B, num_labels, H_out, W_out)
            # Upsample logits to match ground truth mask dimensions
            logits = F.interpolate(logits, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)  # shape: (B, H, W)

        gt_np = batch["labels"].cpu().numpy()
        for p, g in zip(preds.cpu().numpy(), gt_np):
            pred_masks.append(p)
            gt_masks.append(g)

    # Debug: print unique values in the first prediction and ground truth
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
    # Calculate metrics for each image and then average
    for pred, gt in zip(pred_masks, gt_masks):
        # Ensure binary masks (assuming class 1 is the leaf)
        pred_binary = (pred == 1).astype(np.int32)
        gt_binary = gt.astype(np.int32)

        # Pixel accuracy: fraction of matching pixels
        pixel_acc = np.mean(pred_binary == gt_binary)

        # Compute Intersection over Union (IoU)
        intersection = np.sum((pred_binary == 1) & (gt_binary == 1))
        union = np.sum((pred_binary == 1) | (gt_binary == 1))
        iou = intersection / union if union > 0 else 0

        pixel_accs.append(pixel_acc)
        ious.append(iou)
    return np.mean(pixel_accs), np.mean(ious)


# ---------------------------
# 5) PLOTTING RESULTS
# ---------------------------
def plot_scale_sweep(scale_factors, pixel_accs, mious, save_path="pspnet_scale_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(scale_factors, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(scale_factors, mious, marker='s', label='Mean IoU')
    plt.title("PSPNet Segmentation vs. Scale Factor")
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
    # Define the scale factors to test (e.g., scaling down/up the images)
    scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]
    pixel_acc_list, miou_list = [], []

    for scale in scale_factors:
        print(f"\n=== Evaluating with scale_factor={scale} ===")
        dataset = LeafSegFineTuneDataset(
            images_dir=test_images_dir,
            masks_dir=test_masks_dir,
            image_transform=image_transform,
            base_image_size=BASE_IMAGE_SIZE,
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
    results_df.to_csv("pspnet_scale_results.csv", index=False)
    print("CSV saved: pspnet_scale_results.csv")


if __name__ == "__main__":
    main()
