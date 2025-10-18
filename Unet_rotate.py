# import os
# import sys
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# from torch.cuda.amp import autocast
# from sklearn.metrics import confusion_matrix
# import segmentation_models_pytorch as smp
# import torch.nn as nn
#
# # ---------------------------
# # 1) CONFIG + SETUP
# # ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# torch.backends.cudnn.benchmark = True  # Performance optimization
#
# # Path to your saved U-Net model (trained earlier)
# model_path = "./finetuned_unet_leaf.pth"
#
# # Load U-Net Model with ResNet34 encoder (binary segmentation: classes=1)
# model = smp.Unet(
#     encoder_name="resnet34",
#     encoder_weights="imagenet",
#     in_channels=3,
#     classes=1  # Single-channel output (logits)
# )
#
# # Load the state dict
# try:
#     state_dict = torch.load(model_path, map_location=device)
# except Exception as e:
#     print(f"Error loading state dict: {e}")
#     sys.exit(1)
#
# model.load_state_dict(state_dict, strict=False)
# model = model.float()
# model.to(device)
# model.eval()
#
# # Paths to test dataset
# test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
# # ---------------------------
# # 2) LEAF SEGMENTATION DATASET WITH ROTATION
# # ---------------------------
# class LeafSegFineTuneDataset(Dataset):
#     """
#     Loads the original PIL image and its mask, applies a rotation transformation,
#     and processes them for U-Net. The mask is binarized (0 for background, 1 for leaf).
#     """
#     def __init__(self, images_dir, masks_dir, image_size=(544, 544), angle=0):
#         """
#         Args:
#             images_dir: Directory containing input images.
#             masks_dir: Directory containing segmentation masks.
#             image_size: Target size for the images after rotation.
#             angle: Rotation angle in degrees.
#         """
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.image_files = sorted(os.listdir(images_dir))
#         self.image_size = image_size
#         self.angle = angle
#
#         # Transformation for the image (after rotation, we resize and normalize)
#         self.image_transform = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         # Transformation for the mask (resize and convert to tensor)
#         self.mask_transform = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
#             T.ToTensor()
#         ])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#
#         # Open image, apply rotation, and resize
#         image = Image.open(img_path).convert("RGB")
#         image = image.rotate(self.angle, resample=Image.BILINEAR, expand=True)
#         image = image.resize(self.image_size, Image.BILINEAR)
#
#         # Process corresponding mask; assumes mask has the same base name with .png extension
#         mask_name = os.path.splitext(img_name)[0] + ".png"
#         mask_path = os.path.join(self.masks_dir, mask_name)
#         mask = Image.open(mask_path).convert("L")
#         mask = mask.rotate(self.angle, resample=Image.NEAREST, expand=True)
#         mask = mask.resize(self.image_size, Image.NEAREST)
#
#         # Convert image and mask to tensor
#         pixel_values = self.image_transform(image)
#         label = self.mask_transform(mask).squeeze(0)
#         label = (label > 0).float()
#
#         return {"pixel_values": pixel_values, "labels": label}
#
#
# # ---------------------------
# # 3) INFERENCE FUNCTION
# # ---------------------------
# @torch.no_grad()
# def run_inference(dataset, batch_size=4):
#     """Runs inference with mixed precision for efficiency."""
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
#     pred_masks, gt_masks = [], []
#
#     model.eval()
#     for batch in loader:
#         pixel_values = batch["pixel_values"].to(device)
#
#         with autocast():
#             outputs = model(pixel_values)  # Outputs shape: (B, 1, H, W)
#             preds = torch.sigmoid(outputs)
#             pred_maps = (preds > 0.5).long().squeeze(1)
#
#         for pred in pred_maps:
#             pred_masks.append(pred.cpu().numpy())
#         for label in batch["labels"]:
#             gt_masks.append(label.cpu().numpy())
#
#     return pred_masks, gt_masks
#
#
# # ---------------------------
# # 4) METRICS (Pixel Accuracy & Mean IoU)
# # ---------------------------
# def compute_metrics(pred_masks, gt_masks):
#     pixel_accs = []
#     ious = []
#     for pred, gt in zip(pred_masks, gt_masks):
#         pred_flat = pred.flatten()
#         gt_flat = gt.flatten()
#         cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
#         pixel_acc = np.diag(cm).sum() / np.sum(cm)
#         iou = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0] + 1e-10)
#         pixel_accs.append(pixel_acc)
#         ious.append(iou)
#     return np.mean(pixel_accs), np.mean(ious)
#
#
# # ---------------------------
# # 5) PLOTTING RESULTS
# # ---------------------------
# def plot_rotation_sweep(angles, pixel_accs, mious, save_path="unet_rotation_results.png"):
#     plt.figure(figsize=(8, 6))
#     plt.plot(angles, pixel_accs, marker='o', label='Pixel Accuracy')
#     plt.plot(angles, mious, marker='s', label='Mean IoU')
#     plt.title("U-Net Segmentation vs. Rotation Angle")
#     plt.xlabel("Rotation Angle (degrees)")
#     plt.ylabel("Metric Value")
#     plt.ylim([0, 1])
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()
#     print(f"Plot saved to: {save_path}")
#
#
# # ---------------------------
# # 6) MAIN SCRIPT
# # ---------------------------
# def main():
#     # Define a list of rotation angles (in degrees) to evaluate.
#     angles = list(range(0, 361, 30))  # For example: 0°, 30°, 60°, ... 180°
#     pixel_acc_list, miou_list = [], []
#
#     for angle in angles:
#         print(f"\n=== Evaluating with rotation angle={angle}° ===")
#         # Create dataset with the current rotation angle
#         dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, image_size=(544, 544), angle=angle)
#
#         preds, gts = run_inference(dataset)
#         pix_acc, miou = compute_metrics(preds, gts)
#
#         pixel_acc_list.append(pix_acc)
#         miou_list.append(miou)
#
#         print(f"Results (angle={angle}°): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")
#
#     plot_rotation_sweep(angles, pixel_acc_list, miou_list)
#
#     results_df = pd.DataFrame({
#         "rotation_angle": angles,
#         "pixel_accuracy": pixel_acc_list,
#         "mean_iou": miou_list
#     })
#     csv_path = "unet_rotation_results.csv"
#     results_df.to_csv(csv_path, index=False)
#     print(f"CSV saved: {csv_path}")
#
#
# if __name__ == "__main__":
#     main()

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
model = model.float().to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH ROTATION
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    """
    Loads the original PIL image and its mask, applies a rotation transformation,
    and processes them for U-Net. The mask is binarized (0 for background, 1 for leaf).
    """
    def __init__(self, images_dir, masks_dir, image_size=(544, 544), angle=0):
        self.images_dir  = images_dir
        self.masks_dir   = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.image_size  = image_size
        self.angle       = angle

        self.image_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

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
        image = image.rotate(self.angle, resample=Image.BILINEAR, expand=True)
        image = image.resize(self.image_size, Image.BILINEAR)

        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.rotate(self.angle, resample=Image.NEAREST, expand=True)
        mask = mask.resize(self.image_size, Image.NEAREST)

        pixel_values = self.image_transform(image)
        label = self.mask_transform(mask).squeeze(0)
        label = (label > 0).float()

        return {"pixel_values": pixel_values, "labels": label}

# ---------------------------
# 3) INFERENCE FUNCTION
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    pred_masks, gt_masks = [], []

    model.eval()
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)

        with autocast():
            outputs = model(pixel_values)
            preds = torch.sigmoid(outputs)
            pred_maps = (preds > 0.5).long().squeeze(1)

        for pred in pred_maps:
            pred_masks.append(pred.cpu().numpy())
        for label in batch["labels"]:
            gt_masks.append(label.cpu().numpy())

    return pred_masks, gt_masks

# ---------------------------
# 4) METRICS: Pixel Accuracy & Mean IoU
# ---------------------------
def compute_metrics(pred_masks, ref_masks):
    """
    Compute pixel accuracy & mean IoU between two sets of binary masks.
    """
    cm = np.zeros((2, 2), dtype=np.int64)
    for pred, ref in zip(pred_masks, ref_masks):
        pred_flat = pred.flatten()
        ref_flat  = ref.flatten()
        cm += confusion_matrix(ref_flat, pred_flat, labels=[0, 1])

    correct   = np.diag(cm).sum()
    total     = cm.sum()
    pixel_acc = correct / (total + 1e-10)

    intersection = cm[1, 1]
    union        = cm[1, :].sum() + cm[:, 1].sum() - intersection
    mean_iou     = intersection / (union + 1e-10)

    return pixel_acc, mean_iou

# ---------------------------
# 5) HELPER: Inverse Rotate Mask
# ---------------------------
def inverse_rotate_mask(mask, angle, image_size=(544, 544)):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.rotate(-angle, resample=Image.NEAREST, expand=False)
    mask_np  = (np.array(mask_img) > 128).astype(np.uint8)
    return mask_np

# ---------------------------
# 6) PLOTTING RESULTS
# ---------------------------
def plot_rotation_sweep(angles, pixel_accs, mious,
                        save_path="unet_rotation_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(angles, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(angles, mious,       marker='s', label='Mean IoU')
    plt.title("U-Net Segmentation Consistency vs. Rotation Angle")
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to: {save_path}")

# ---------------------------
# 7) MAIN SCRIPT
# ---------------------------
def main():
    print("Running baseline inference (angle=0)...")
    baseline_ds    = LeafSegFineTuneDataset(
        test_images_dir, test_masks_dir,
        image_size=(544, 544), angle=0
    )
    baseline_preds, baseline_gts = run_inference(baseline_ds)

    angles          = list(range(0, 361, 30))
    pixel_acc_list  = []
    miou_list       = []

    for angle in angles:
        print(f"\n=== Evaluating with rotation angle = {angle}° ===")
        ds, _ = LeafSegFineTuneDataset(
            test_images_dir, test_masks_dir,
            image_size=(544, 544), angle=angle
        ), None
        rotated_preds, _ = run_inference(ds)

        adjusted_preds = []
        for pred in rotated_preds:
            if angle != 0:
                adj = inverse_rotate_mask(pred, angle)
            else:
                adj = pred
            adjusted_preds.append(adj)

        pa, miou = compute_metrics(adjusted_preds, baseline_preds)
        pixel_acc_list.append(pa)
        miou_list.append(miou)
        print(f"Angle {angle}° → PixelAcc={pa:.4f}, MeanIoU={miou:.4f}")

    plot_rotation_sweep(angles, pixel_acc_list, miou_list)

    results_df = pd.DataFrame({
        "rotation_angle": angles,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    csv_path = "unet_rotation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

if __name__ == "__main__":
    main()

