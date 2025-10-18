import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd  # NEW: for CSV export

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable cuDNN benchmark for potential speedups
torch.backends.cudnn.benchmark = True

# Load DeepLabv3 with new 'weights' parameter
model = models.segmentation.deeplabv3_resnet50(
    weights=DeepLabV3_ResNet50_Weights.DEFAULT
).to(device)
model.eval()

# Paths: adjust to match your setup
VOC_ROOT = r"C:\Users\goker\PycharmProjects\DiplomProject\voc"
IMAGES_FOLDER = os.path.join(VOC_ROOT, "JPEGImages")       # Original images
MASKS_FOLDER = os.path.join(VOC_ROOT, "SegmentationClass") # Ground-truth masks
VAL_TXT_PATH = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", "val.txt")

# ---------------------------
# 2) DATASET FOR PASCAL VOC SEGMENTATION
# ---------------------------
class VOCSegmentationDataset(Dataset):
    """
    Loads (image, segmentation mask) pairs from Pascal VOC,
    reading image IDs from the specified text file (e.g., 'val.txt').
    """

    def __init__(self, images_dir, masks_dir, list_path, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        with open(list_path, "r") as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img_path = os.path.join(self.images_dir, image_id + ".jpg")
        mask_path = os.path.join(self.masks_dir, image_id + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask, image_id

# ---------------------------
# 3) TRANSFORM BUILDER
# ---------------------------
def build_transform(resize=(520, 520), sigma=0):
    """
    Returns a SegmentationTransform that optionally applies Gaussian blur
    with the given sigma. If sigma=0, no blur is applied.
    """
    return SegmentationTransform(
        resize=resize,
        blur_sigma=sigma
    )

class SegmentationTransform:
    """
    Resizes and optionally applies Gaussian blur to the image, then
    normalizes it. Resizes the mask (w/o blur) and converts to LongTensor.
    """

    def __init__(self, resize=(520, 520), blur_sigma=0):
        self.resize = resize
        self.blur_sigma = blur_sigma

        # Build transform pipeline for the image
        transforms_list = [
            T.Resize(self.resize, interpolation=T.InterpolationMode.BILINEAR)
        ]
        if self.blur_sigma > 0:
            # e.g., kernel_size=(5,5) can be tuned further if desired
            transforms_list.append(
                T.GaussianBlur(kernel_size=(5, 5), sigma=(blur_sigma, blur_sigma))
            )
        transforms_list += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ]
        self.image_transform = T.Compose(transforms_list)

        # Mask transform (no blur, just resize + convert to LongTensor)
        self.mask_transform = T.Compose([
            T.Resize(self.resize, interpolation=T.InterpolationMode.NEAREST)
        ])

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask

# ---------------------------
# 4) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(pred_masks, gt_masks):
    """
    Computes Pixel Accuracy & Mean IoU, ignoring label=255 ('void' in VOC).
    Computes per-image metrics before averaging for improved accuracy.
    Dynamically infers number of classes from the input data.
    """
    pixel_accs = []
    ious = []

    # Dynamically determine num_classes from data
    all_labels = np.unique(np.concatenate([gt.flatten() for gt in gt_masks]))
    all_labels = all_labels[all_labels != 255]  # Remove ignored label
    if len(all_labels) == 0:
        return 0.0, 0.0  # No valid labels in dataset
    num_classes = int(all_labels.max() + 1)  # Ensure range includes all labels

    for pred, gt in zip(pred_masks, gt_masks):
        pred = pred.flatten()
        gt = gt.flatten()

        valid = (gt != 255)  # Ignore void pixels
        pred = pred[valid]
        gt = gt[valid]

        if len(gt) == 0:  # Skip if no valid pixels
            continue

        cm = confusion_matrix(gt, pred, labels=list(range(num_classes)))

        # Pixel Accuracy
        pixel_acc = np.diag(cm).sum() / (cm.sum() + 1e-10)
        pixel_accs.append(pixel_acc)

        # Mean IoU
        iou_list = []
        for c in range(num_classes):  # Iterate only over actual class range
            if cm[c, :].sum() == 0 and cm[:, c].sum() == 0:
                continue  # Ignore classes not present
            iou = cm[c, c] / (cm[c, :].sum() + cm[:, c].sum() - cm[c, c] + 1e-10)
            iou_list.append(iou)

        mean_iou = np.mean(iou_list) if iou_list else 0.0
        ious.append(mean_iou)

    return (np.mean(pixel_accs) if pixel_accs else 0.0,
            np.mean(ious) if ious else 0.0)
# ---------------------------
# 5) INFERENCE FUNCTION (with Mixed Precision)
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=8):
    """
    Runs inference with mixed precision for a speed-up on CUDA-enabled GPUs.
    Returns predicted masks & ground-truth masks.
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

        # Mixed precision context
        with torch.cuda.amp.autocast():
            outputs = model(images)['out']  # shape (B, 21, H, W)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        gt_np = masks.numpy()
        for p, g in zip(preds, gt_np):
            pred_masks.append(p)
            gt_masks.append(g)

    return pred_masks, gt_masks

# ---------------------------
# 6) PLOTTING RESULTS
# ---------------------------
def plot_blur_sweep(blur_levels, pixel_accs, mious):
    """
    Plots Pixel Accuracy and Mean IoU vs. blur level (sigma).
    """
    plt.figure(figsize=(8, 6))
    # Plot Pixel Accuracy
    plt.plot(blur_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    # Plot Mean IoU
    plt.plot(blur_levels, mious, marker='s', label='Mean IoU')

    plt.title("Segmentation Results vs. Gaussian Blur Sigma")
    plt.xlabel("Blur Sigma")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])  # both PixelAcc and IoU in [0..1]
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# 7) MAIN SCRIPT
# ---------------------------
def main():
    # Various blur sigma levels; adjust as desired
    blur_levels = [0, 1, 2, 3, 4]

    pixel_acc_list = []
    miou_list = []

    for sigma in blur_levels:
        print(f"\n=== Evaluating with blur sigma={sigma} ===")
        transform = build_transform(resize=(520, 520), sigma=sigma)

        dataset = VOCSegmentationDataset(
            images_dir=IMAGES_FOLDER,
            masks_dir=MASKS_FOLDER,
            list_path=VAL_TXT_PATH,
            transform=transform
        )

        preds, gts = run_inference(dataset, batch_size=8)
        pix_acc, miou = compute_metrics(preds, gts)

        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)

        print(f"Results (sigma={sigma}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    # (A) Plot the collected results
    plot_blur_sweep(blur_levels, pixel_acc_list, miou_list)

    # (B) Print final summary
    print("\n===== Final Summary =====")
    for s, acc, iou in zip(blur_levels, pixel_acc_list, miou_list):
        print(f"Sigma={s}: PixelAcc={acc:.3f}, MeanIoU={iou:.3f}")

    # (C) NEW: Save results to CSV
    # Construct a DataFrame (requires 'import pandas as pd')
    results_df = pd.DataFrame({
        "sigma": blur_levels,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    csv_path = "deepLabv3_blur_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
