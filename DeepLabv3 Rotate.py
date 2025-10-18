# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import torchvision.models as models
# from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import pandas as pd
#
# # ---------------------------
# # 1) CONFIG + SETUP
# # ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# torch.backends.cudnn.benchmark = True  # Speed optimization for fixed-size inputs
#
# model = models.segmentation.deeplabv3_resnet50(
#     weights=DeepLabV3_ResNet50_Weights.DEFAULT
# ).to(device)
# model.eval()
#
# # Adjust to your local Pascal VOC paths
# VOC_ROOT = r"C:\Users\goker\PycharmProjects\DiplomProject\voc"
# IMAGES_FOLDER = os.path.join(VOC_ROOT, "JPEGImages")
# MASKS_FOLDER = os.path.join(VOC_ROOT, "SegmentationClass")
# VAL_TXT_PATH = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", "val.txt")
#
# # ---------------------------
# # 2) DATASET FOR PASCAL VOC SEGMENTATION
# # ---------------------------
# class VOCSegmentationDataset(Dataset):
#     """
#     Loads (image, segmentation mask) pairs from Pascal VOC,
#     reading image IDs from the specified text file (e.g., 'val.txt').
#     """
#
#     def __init__(self, images_dir, masks_dir, list_path, transform=None):
#         super().__init__()
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.transform = transform
#
#         with open(list_path, "r") as f:
#             self.image_ids = [line.strip() for line in f if line.strip()]
#
#     def __len__(self):
#         return len(self.image_ids)
#
#     def __getitem__(self, index):
#         image_id = self.image_ids[index]
#         img_path = os.path.join(self.images_dir, image_id + ".jpg")
#         mask_path = os.path.join(self.masks_dir, image_id + ".png")
#
#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path)
#
#         if self.transform:
#             image, mask = self.transform(image, mask)
#
#         return image, mask, image_id
#
#
# # ---------------------------
# # 3) ROTATION WITH EXPAND + RESIZE
# #    AND LABELING BLACK CORNERS AS IGNORE (255)
# # ---------------------------
# class RotatePairExpand:
#     """
#     Rotates both image (bilinear) and mask (nearest) with 'expand=True'
#     so no original content is lost.
#     """
#     def __init__(self, angle=0):
#         self.angle = angle
#
#     def __call__(self, image, mask):
#         # Rotate image & mask with expand=True
#         image_rot = image.rotate(self.angle, resample=Image.BILINEAR, expand=True)
#         mask_rot  = mask.rotate(self.angle, resample=Image.NEAREST,  expand=True)
#         return image_rot, mask_rot
#
#
# class SegmentationTransform:
#     """
#     1. Rotate (image, mask) with expand=True.
#     2. Resizes them to (520, 520).
#     3. Creates a 'valid region' mask to track black corners and set them to 255 in the final mask.
#     4. Converts the image to tensor & normalizes.
#     5. Converts the mask to LongTensor, with black corners set as IGNORE (255).
#     """
#
#     def __init__(self, angle=0, resize=(520, 520)):
#         self.angle = angle
#         self.resize = resize
#         self.rotate_pair = RotatePairExpand(angle=self.angle)
#
#         # The transforms for final image
#         self.image_after_rotate = T.Compose([
#             T.Resize(self.resize, interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])
#         ])
#
#         # The transforms for final mask
#         self.mask_after_rotate = T.Compose([
#             T.Resize(self.resize, interpolation=T.InterpolationMode.NEAREST)
#         ])
#
#     def __call__(self, image, mask):
#         # 1) Rotate image & mask
#         image_rot, mask_rot = self.rotate_pair(image, mask)
#
#         # 2) Also rotate a "valid region" map to find black corners
#         #    - Create a white (value=1) image the same size as 'image' originally
#         w, h = image.size
#         valid_map = Image.new("L", (w, h), color=1)  # all '1'
#         valid_map_rot = valid_map.rotate(self.angle, resample=Image.NEAREST, expand=True)
#
#         # 3) Resize all
#         image_out = self.image_after_rotate(image_rot)
#         mask_out  = self.mask_after_rotate(mask_rot)
#         valid_map_resized = valid_map_rot.resize(self.resize, Image.NEAREST)
#
#         # Convert mask to numpy int
#         mask_out = torch.from_numpy(np.array(mask_out, dtype=np.int64))
#
#         # 4) Convert valid_map to numpy to see which pixels were "originally valid"
#         valid_np = np.array(valid_map_resized, dtype=np.uint8)  # 1 where original, 0 where new/black
#
#         # 5) Mark black corners as IGNORE (255) in the final mask
#         #    Wherever valid_np == 0, set mask_out to 255
#         #    But be careful to handle the mask_out as a NumPy array for indexing.
#         mask_np = mask_out.numpy()  # shape: (H, W)
#         mask_np[valid_np == 0] = 255  # label=255 means 'ignore' in VOC
#         mask_out = torch.from_numpy(mask_np)
#
#         return image_out, mask_out
#
#
# def build_transform(angle=0, resize=(520, 520)):
#     """
#     Build a transform that:
#       - rotates with expand=True
#       - resizes to (520, 520)
#       - sets newly introduced black corners as ignore=255 in the mask
#     """
#     return SegmentationTransform(angle=angle, resize=resize)
#
# # ---------------------------
# # 4) METRICS (Pixel Accuracy & Mean IoU)
# # ---------------------------
# def compute_metrics(pred_masks, gt_masks, num_classes=21):
#     """
#     Compute Pixel Accuracy & Mean IoU, ignoring label=255 (the 'void' in VOC).
#     """
#     cm = np.zeros((num_classes, num_classes), dtype=np.int64)
#
#     for pred, gt in zip(pred_masks, gt_masks):
#         pred = pred.flatten()
#         gt = gt.flatten()
#
#         # Exclude "void"/ignore label
#         valid = (gt != 255)
#         pred = pred[valid]
#         gt = gt[valid]
#
#         cm += confusion_matrix(gt, pred, labels=range(num_classes))
#
#     # Pixel Accuracy
#     correct = np.diag(cm).sum()
#     total = cm.sum()
#     pixel_acc = correct / (total + 1e-10)
#
#     # Mean IoU
#     iou_list = []
#     for c in range(num_classes):
#         if cm[c, :].sum() == 0 and cm[:, c].sum() == 0:
#             continue  # Class c not present at all
#         iou = cm[c, c] / (cm[c, :].sum() + cm[:, c].sum() - cm[c, c] + 1e-10)
#         iou_list.append(iou)
#
#     mean_iou = np.mean(iou_list) if iou_list else 0.0
#     return pixel_acc, mean_iou
#
#
# # ---------------------------
# # 5) INFERENCE FUNCTION (with Mixed Precision)
# # ---------------------------
# @torch.no_grad()
# def run_inference(dataset, batch_size=8):
#     """
#     Runs inference with mixed precision on a given dataset.
#     Returns predicted masks & ground-truth masks for metric computation.
#     """
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
#
#     pred_masks = []
#     gt_masks = []
#
#     model.eval()
#     for images, masks, _ in loader:
#         images = images.to(device, non_blocking=True)
#
#         with torch.cuda.amp.autocast():
#             outputs = model(images)['out']  # shape: (B, 21, H, W)
#             preds = torch.argmax(outputs, dim=1).cpu().numpy()
#
#         gt_np = masks.numpy()
#         for p, g in zip(preds, gt_np):
#             pred_masks.append(p)
#             gt_masks.append(g)
#
#     return pred_masks, gt_masks
#
#
# # ---------------------------
# # 6) PLOTTING & SAVING RESULTS
# # ---------------------------
# def plot_rotation_sweep(angles, pixel_accs, mious):
#     """
#     Plots Pixel Accuracy and Mean IoU vs. rotation angle.
#     """
#     plt.figure(figsize=(8, 6))
#
#     # Plot Pixel Accuracy
#     plt.plot(angles, pixel_accs, marker='o', label='Pixel Accuracy')
#     # Plot Mean IoU
#     plt.plot(angles, mious, marker='s', label='Mean IoU')
#
#     plt.title("Segmentation Results vs. Rotation Angle (Expand=True + Ignore Black Corners)")
#     plt.xlabel("Rotation Angle (degrees)")
#     plt.ylabel("Metric Value")
#     plt.ylim([0, 1])  # PixelAcc & IoU in [0..1]
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#
#
# # ---------------------------
# # 7) MAIN SCRIPT
# # ---------------------------
# def main():
#     # Rotate images in 30-degree increments, up to 180
#     angles = list(range(0, 181, 30))
#
#     pixel_acc_list = []
#     miou_list = []
#
#     for angle in angles:
#         print(f"\n=== Evaluating with rotation angle={angle}° (expand=True, corners=ignore) ===")
#         transform = build_transform(angle=angle, resize=(520, 520))
#
#         dataset = VOCSegmentationDataset(
#             images_dir=IMAGES_FOLDER,
#             masks_dir=MASKS_FOLDER,
#             list_path=VAL_TXT_PATH,
#             transform=transform
#         )
#
#         preds, gts = run_inference(dataset, batch_size=8)
#         pix_acc, miou = compute_metrics(preds, gts)
#
#         pixel_acc_list.append(pix_acc)
#         miou_list.append(miou)
#
#         print(f"Results (angle={angle}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")
#
#     # (A) Plot results
#     plot_rotation_sweep(angles, pixel_acc_list, miou_list)
#
#     # (B) Print final summary table
#     print("\n===== Final Summary (Rotation + expand=True + ignore corners) =====")
#     for a, acc, iou in zip(angles, pixel_acc_list, miou_list):
#         print(f"Angle={a}°: PixelAcc={acc:.3f}, MeanIoU={iou:.3f}")
#
#     # (C) Save to CSV
#     results_df = pd.DataFrame({
#         "rotation_angle": angles,
#         "pixel_accuracy": pixel_acc_list,
#         "mean_iou": miou_list
#     })
#     csv_path = "deepLabv3_rotation_ignore_results.csv"
#     results_df.to_csv(csv_path, index=False)
#     print(f"\nCSV saved to: {csv_path}")
#
#
# if __name__ == "__main__":
#     main()
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
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

torch.backends.cudnn.benchmark = True  # Speed optimization for fixed-size inputs

# Load DeepLabv3 with ResNet50 backbone using the new 'weights' parameter
model = models.segmentation.deeplabv3_resnet50(
    weights=DeepLabV3_ResNet50_Weights.DEFAULT
).to(device)
model.eval()

# Adjust to your local Pascal VOC paths
VOC_ROOT = r"C:\Users\goker\PycharmProjects\DiplomProject\voc"
IMAGES_FOLDER = os.path.join(VOC_ROOT, "JPEGImages")
MASKS_FOLDER = os.path.join(VOC_ROOT, "SegmentationClass")
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
# 3) ROTATION WITH EXPAND + RESIZE
#    AND LABELING BLACK CORNERS AS IGNORE (255)
# ---------------------------
class RotatePairExpand:
    """
    Rotates both image (using bilinear interpolation) and mask (using nearest)
    with expand=True so that no original content is lost.
    """
    def __init__(self, angle=0):
        self.angle = angle

    def __call__(self, image, mask):
        image_rot = image.rotate(self.angle, resample=Image.BILINEAR, expand=True)
        mask_rot  = mask.rotate(self.angle, resample=Image.NEAREST, expand=True)
        return image_rot, mask_rot

class SegmentationTransform:
    """
    1. Rotates (image, mask) with expand=True.
    2. Resizes them to a fixed size (520, 520).
    3. Creates a 'valid region' map to find newly introduced black corners and sets those pixels in the mask to 255 (ignore).
    4. Converts the image to tensor and normalizes.
    5. Converts the mask to a LongTensor.
    """
    def __init__(self, angle=0, resize=(520, 520)):
        self.angle = angle
        self.resize = resize
        self.rotate_pair = RotatePairExpand(angle=self.angle)

        self.image_after_rotate = T.Compose([
            T.Resize(self.resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.mask_after_rotate = T.Compose([
            T.Resize(self.resize, interpolation=T.InterpolationMode.NEAREST)
        ])

    def __call__(self, image, mask):
        # 1) Rotate image & mask
        image_rot, mask_rot = self.rotate_pair(image, mask)
        # 2) Create a valid region map that is white (all ones) and rotate it
        w, h = image.size
        valid_map = Image.new("L", (w, h), color=1)  # all pixels 1
        valid_map_rot = valid_map.rotate(self.angle, resample=Image.NEAREST, expand=True)
        # 3) Resize all outputs
        image_out = self.image_after_rotate(image_rot)
        mask_out  = self.mask_after_rotate(mask_rot)
        valid_map_resized = valid_map_rot.resize(self.resize, Image.NEAREST)
        # 4) Process the mask: convert to numpy and set new black corners to ignore (255)
        mask_out = torch.from_numpy(np.array(mask_out, dtype=np.int64))
        valid_np = np.array(valid_map_resized, dtype=np.uint8)  # 1 where originally valid, 0 where black/introduced
        mask_np = mask_out.numpy()
        mask_np[valid_np == 0] = 255  # Set ignore label (255)
        mask_out = torch.from_numpy(mask_np)
        return image_out, mask_out

def build_transform(angle=0, resize=(520, 520)):
    """
    Returns a SegmentationTransform with the specified rotation angle and resize dimensions.
    """
    return SegmentationTransform(angle=angle, resize=resize)

# ---------------------------
# 4) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(pred_masks, gt_masks, num_classes=21):
    """
    Compute Pixel Accuracy & Mean IoU, ignoring label 255 (void in VOC).
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, gt in zip(pred_masks, gt_masks):
        pred = pred.flatten()
        gt = gt.flatten()
        valid = (gt != 255)
        pred = pred[valid]
        gt = gt[valid]
        cm += confusion_matrix(gt, pred, labels=range(num_classes))
    correct = np.diag(cm).sum()
    total = cm.sum()
    pixel_acc = correct / (total + 1e-10)
    iou_list = []
    for c in range(num_classes):
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
def run_inference(dataset, batch_size=8):
    """
    Runs inference with mixed precision on a given dataset.
    Returns predicted masks and ground-truth masks.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    pred_masks = []
    gt_masks = []
    model.eval()
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(images)['out']  # shape: (B, 21, H, W)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        gt_np = masks.numpy()
        for p, g in zip(preds, gt_np):
            pred_masks.append(p)
            gt_masks.append(g)
    return pred_masks, gt_masks

# ---------------------------
# 6) HELPER FUNCTION: Inverse Rotate Mask
# ---------------------------
def inverse_rotate_mask(mask, angle, resize=(520,520)):
    """
    Converts a mask (numpy array) to a PIL image, rotates it by -angle,
    and returns the adjusted mask as a numpy array.
    """
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img = mask_img.rotate(-angle, resample=Image.NEAREST, expand=False)
    mask_img = mask_img.resize(resize, Image.NEAREST)
    return np.array(mask_img)

# ---------------------------
# 7) PLOTTING & SAVING RESULTS
# ---------------------------
def plot_rotation_sweep(angles, pixel_accs, mious):
    """
    Plots Pixel Accuracy and Mean IoU vs. rotation angle.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(angles, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(angles, mious, marker='s', label='Mean IoU')
    plt.title("Segmentation Results vs. Rotation Angle\n(Expand=True + Ignore Black Corners)")
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------------
# 8) MAIN SCRIPT
# ---------------------------
def main():
    # First, run baseline inference on original images (angle=0) as reference.
    print("Running baseline inference (angle=0)...")
    baseline_transform = build_transform(angle=0, resize=(520,520))
    baseline_dataset = VOCSegmentationDataset(
        images_dir=IMAGES_FOLDER,
        masks_dir=MASKS_FOLDER,
        list_path=VAL_TXT_PATH,
        transform=baseline_transform
    )
    baseline_preds, _ = run_inference(baseline_dataset, batch_size=8)

    # Define rotation angles to evaluate.
    angles = list(range(0, 361, 30))
    pixel_acc_list = []
    miou_list = []

    for angle in angles:
        print(f"\n=== Evaluating with rotation angle={angle}° (expand=True, ignore corners) ===")
        transform = build_transform(angle=angle, resize=(520,520))
        dataset = VOCSegmentationDataset(
            images_dir=IMAGES_FOLDER,
            masks_dir=MASKS_FOLDER,
            list_path=VAL_TXT_PATH,
            transform=transform
        )
        preds, _ = run_inference(dataset, batch_size=8)
        # Inverse rotate predictions (if angle != 0) so they align with the baseline.
        adjusted_preds = []
        for pred in preds:
            if angle != 0:
                adjusted_pred = inverse_rotate_mask(pred, angle, resize=(520,520))
            else:
                adjusted_pred = pred
            adjusted_preds.append(adjusted_pred)
        # Compute metrics comparing the adjusted predictions with the baseline predictions.
        pix_acc, miou = compute_metrics(adjusted_preds, baseline_preds, num_classes=21)
        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)
        print(f"Results (angle={angle}°): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    # Plot results.
    plot_rotation_sweep(angles, pixel_acc_list, miou_list)

    # Print final summary and save to CSV.
    print("\n===== Final Summary (Rotation + Expand + Ignore Corners) =====")
    for a, acc, iou in zip(angles, pixel_acc_list, miou_list):
        print(f"Angle={a}°: PixelAcc={acc:.3f}, MeanIoU={iou:.3f}")
    results_df = pd.DataFrame({
        "rotation_angle": angles,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    csv_path = "deepLabv3_rotation_ignore_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
