import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights  # Updated for ResNet101
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # For speed optimizations if input sizes are consistent

# Load DeepLabv3 with ResNet101 backbone using new 'weights' parameter
model = models.segmentation.deeplabv3_resnet101(
    weights=DeepLabV3_ResNet101_Weights.DEFAULT
).to(device)
model.eval()

# Adjust these paths to your local setup
VOC_ROOT = r"C:\Users\goker\PycharmProjects\DiplomProject\voc"
IMAGES_FOLDER = os.path.join(VOC_ROOT, "JPEGImages")
MASKS_FOLDER = os.path.join(VOC_ROOT, "SegmentationClass")
VAL_TXT_PATH = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", "val.txt")


# ---------------------------
# 2) DATASET FOR PASCAL VOC SEGMENTATION
# ---------------------------
class VOCSegmentationDataset(Dataset):
    """
    Loads (image, segmentation mask) pairs from Pascal VOC.
    Reads image IDs from the specified text file (e.g., 'val.txt').
    """
    def __init__(self, images_dir, masks_dir, list_path, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Read all image IDs (e.g. "2007_000032") from val.txt
        with open(list_path, "r") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img_path = os.path.join(self.images_dir, image_id + ".jpg")
        mask_path = os.path.join(self.masks_dir, image_id + ".png")

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Apply transform if provided
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask, image_id


# ---------------------------
# 3) CUSTOM GAUSSIAN NOISE CLASS
# ---------------------------
class AddGaussianNoise(torch.nn.Module):
    """
    A custom transform that adds Gaussian noise to a tensor.
    mean: The mean of the Gaussian distribution (0 by default).
    std:  The standard deviation (noise intensity).
    """
    def __init__(self, mean=0.0, std=0.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        if self.std <= 0:
            # No noise applied if std <= 0
            return tensor
        # Add Gaussian noise
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


# ---------------------------
# 4) TRANSFORM BUILDER
# ---------------------------
def build_transform(resize=(520, 520), noise_std=0.0):
    """
    Returns a SegmentationTransform that resizes the image & mask,
    and adds Gaussian noise with std=noise_std. If noise_std=0,
    no noise is added.
    """
    return SegmentationTransform(
        resize=resize,
        noise_std=noise_std
    )


class SegmentationTransform:
    """
    1. Resizes the image & mask.
    2. Optionally adds Gaussian noise to the image.
    3. Converts image to tensor & normalizes.
    4. Converts mask to LongTensor (class IDs).
    """
    def __init__(self, resize=(520, 520), noise_std=0.0):
        self.resize = resize
        self.noise_std = noise_std

        # Build the image transform pipeline
        transforms_list = [
            T.Resize(self.resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            AddGaussianNoise(std=self.noise_std),  # Custom noise transform
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ]
        self.image_transform = T.Compose(transforms_list)

        # Mask transform: just resize with nearest neighbor, then convert to int tensor
        self.mask_transform = T.Compose([
            T.Resize(self.resize, interpolation=T.InterpolationMode.NEAREST)
        ])

    def __call__(self, image, mask):
        # Transform the image
        image = self.image_transform(image)

        # Transform the mask
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask


# ---------------------------
# 5) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(pred_masks, gt_masks, num_classes=21):
    """
    Compute Pixel Accuracy and Mean IoU, ignoring label=255 (the 'void' in VOC).
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for pred, gt in zip(pred_masks, gt_masks):
        pred = pred.flatten()
        gt = gt.flatten()

        # Exclude "void" label (255)
        valid = (gt != 255)
        pred = pred[valid]
        gt = gt[valid]

        cm += confusion_matrix(gt, pred, labels=range(num_classes))

    # Pixel Accuracy
    correct = np.diag(cm).sum()
    total = cm.sum()
    pixel_acc = correct / (total + 1e-10)

    # Mean IoU
    iou_list = []
    for c in range(num_classes):
        # If no pixels for class c, skip it
        if cm[c, :].sum() == 0 and cm[:, c].sum() == 0:
            continue
        iou = cm[c, c] / (cm[c, :].sum() + cm[:, c].sum() - cm[c, c] + 1e-10)
        iou_list.append(iou)

    mean_iou = np.mean(iou_list) if iou_list else 0.0
    return pixel_acc, mean_iou


# ---------------------------
# 6) INFERENCE FUNCTION (with Mixed Precision)
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=8):
    """
    Runs inference with mixed precision on a given dataset.
    Returns predicted masks & ground-truth masks for metric computation.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    pred_masks = []
    gt_masks = []

    model.eval()
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)

        # Mixed precision inference
        with torch.cuda.amp.autocast():
            outputs = model(images)['out']  # shape: (B, 21, H, W)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        gt_np = masks.numpy()
        for p, g in zip(preds, gt_np):
            pred_masks.append(p)
            gt_masks.append(g)

    return pred_masks, gt_masks


# ---------------------------
# 7) PLOTTING: PixelAcc / MeanIoU vs. Noise STD
# ---------------------------
def plot_noise_sweep(noise_levels, pixel_accs, mious):
    """
    Plots Pixel Accuracy and Mean IoU vs. noise level (std).
    """
    plt.figure(figsize=(8, 6))

    # Plot Pixel Accuracy
    plt.plot(noise_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    # Plot Mean IoU
    plt.plot(noise_levels, mious, marker='s', label='Mean IoU')

    plt.title("Segmentation Results vs. Gaussian Noise (std)")
    plt.xlabel("Noise Std")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])  # PixelAcc & IoU are between 0..1
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------
# 8) MAIN SCRIPT
# ---------------------------
def main():
    # You can test multiple noise levels (standard deviations)
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]

    pixel_acc_list = []
    miou_list = []

    for std in noise_levels:
        print(f"\n=== Evaluating with noise std={std} ===")
        transform = build_transform(resize=(520, 520), noise_std=std)

        dataset = VOCSegmentationDataset(
            images_dir=IMAGES_FOLDER,
            masks_dir=MASKS_FOLDER,
            list_path=VAL_TXT_PATH,
            transform=transform
        )

        preds, gts = run_inference(dataset, batch_size=4)
        pix_acc, miou = compute_metrics(preds, gts)

        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)

        print(f"Results (noise_std={std}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    # (A) Plot the collected results
    plot_noise_sweep(noise_levels, pixel_acc_list, miou_list)

    # (B) Print a final summary table
    print("\n===== Final Summary (Noise) =====")
    for s, acc, iou in zip(noise_levels, pixel_acc_list, miou_list):
        print(f"Noise={s}: PixelAcc={acc:.3f}, MeanIoU={iou:.3f}")

    # (C) Save results to CSV
    results_df = pd.DataFrame({
        "noise_std": noise_levels,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    csv_path = "deepLabv3_ResNet101_noise_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
