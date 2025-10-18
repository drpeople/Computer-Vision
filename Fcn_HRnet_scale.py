import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models.segmentation import FCN_ResNet50_Weights  # FCN model weights
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

# Use FCN_ResNet50 for segmentation
model = models.segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT).to(device)
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
# 3) CUSTOM SCALING TRANSFORMS
# ---------------------------
class ScaleTransform:
    """
    Scales a PIL image by a specified factor.
    scale_factor: multiplier to scale the image dimensions.
    interpolation: interpolation method for resizing (default: bicubic for images).
    """

    def __init__(self, scale_factor=1.0, interpolation=Image.BICUBIC):
        self.scale_factor = scale_factor
        self.interpolation = interpolation

    def __call__(self, img):
        width, height = img.size
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        return img.resize((new_width, new_height), self.interpolation)


class SegmentationScaleTransform:
    """
    1. Scales the image and mask by a specified factor.
    2. Converts the image to a tensor and normalizes it.
    3. Converts the mask to a LongTensor.

    Note: No additional fixed-size resizing is applied—images will have variable sizes
    according to the scale factor. Therefore, we use a DataLoader with batch_size=1.
    """

    def __init__(self, scale_factor=1.0):
        self.scale_factor = scale_factor
        self.image_transform = T.Compose([
            ScaleTransform(scale_factor=self.scale_factor, interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = T.Compose([
            ScaleTransform(scale_factor=self.scale_factor, interpolation=Image.NEAREST)
        ])

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


def build_scale_transform(scale_factor=1.0):
    """
    Returns a SegmentationScaleTransform that scales both the image and the mask.
    """
    return SegmentationScaleTransform(scale_factor=scale_factor)


# ---------------------------
# 4) METRICS (Pixel Accuracy & Mean IoU)
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
# 5) INFERENCE FUNCTION (with Mixed Precision)
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=1):
    """
    Runs inference with mixed precision on a given dataset.
    Returns predicted masks & ground-truth masks for metric computation.
    Note: batch_size is set to 1 because the images (after scaling) have variable sizes.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Using 0 workers to simplify variable image sizes
        pin_memory=False
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
# 6) PLOTTING: PixelAcc / MeanIoU vs. Scale Factor
# ---------------------------
def plot_scale_sweep(scale_levels, pixel_accs, mious):
    """
    Plots Pixel Accuracy and Mean IoU vs. scale factor.
    """
    plt.figure(figsize=(8, 6))

    # Plot Pixel Accuracy
    plt.plot(scale_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    # Plot Mean IoU
    plt.plot(scale_levels, mious, marker='s', label='Mean IoU')

    plt.title("Segmentation Results vs. Scale Factor")
    plt.xlabel("Scale Factor")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------
# 7) MAIN SCRIPT
# ---------------------------
def main():
    # Define the scale factors to test.
    scale_levels = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]

    pixel_acc_list = []
    miou_list = []

    for scale in scale_levels:
        print(f"\n=== Evaluating with scale_factor={scale} ===")
        transform = build_scale_transform(scale_factor=scale)

        dataset = VOCSegmentationDataset(
            images_dir=IMAGES_FOLDER,
            masks_dir=MASKS_FOLDER,
            list_path=VAL_TXT_PATH,
            transform=transform
        )

        preds, gts = run_inference(dataset, batch_size=1)
        pix_acc, miou = compute_metrics(preds, gts)

        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)

        print(f"Results (scale_factor={scale}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    # (A) Plot the collected results
    plot_scale_sweep(scale_levels, pixel_acc_list, miou_list)

    # (B) Print a final summary table
    print("\n===== Final Summary (Scale Factor) =====")
    for s, acc, iou in zip(scale_levels, pixel_acc_list, miou_list):
        print(f"Scale={s}: PixelAcc={acc:.3f}, MeanIoU={iou:.3f}")

    # (C) Save results to CSV
    results_df = pd.DataFrame({
        "scale_factor": scale_levels,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    csv_path = "fcn_resnet50_scale_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
