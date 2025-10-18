# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
# from torch.cuda.amp import autocast
# from sklearn.metrics import confusion_matrix
# from torchvision.transforms import functional as TF
#
# # ---------------------------
# # 1) CONFIG + SETUP
# # ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# torch.backends.cudnn.benchmark = True  # Performance optimization
#
# # Load Mask2Former model and processor
# model_path = "./mask2former_finetuned_leaf"  # Ensure this path exists
# processor = AutoImageProcessor.from_pretrained(model_path)
# model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
# model.eval()
#
# # Paths to test dataset
# test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
#
# # ---------------------------
# # 2) LEAF SEGMENTATION DATASET WITH ROTATION
# # ---------------------------
# class LeafSegFineTuneDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, processor, image_size=(352, 352), angle=0):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.image_files = sorted(os.listdir(images_dir))
#         self.processor = processor
#         self.image_size = image_size
#         self.angle = angle  # Rotation angle in degrees
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#
#         # Open image and rotate it; expand so that the full rotated image is kept
#         image = Image.open(img_path).convert("RGB")
#         image = image.rotate(self.angle, resample=Image.BILINEAR, expand=True)
#         # Resize to desired size
#         image = image.resize(self.image_size, Image.BILINEAR)
#
#         mask_name = img_name.replace('.jpg', '.png')
#         mask_path = os.path.join(self.masks_dir, mask_name)
#         mask = Image.open(mask_path).convert("L")
#         # Rotate mask using nearest neighbor interpolation, then resize
#         mask = mask.rotate(self.angle, resample=Image.NEAREST, expand=True)
#         mask = mask.resize(self.image_size, Image.NEAREST)
#
#         # Process image using the Mask2Former processor (no text prompt is needed)
#         inputs = self.processor(images=image, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].squeeze(0)  # shape: (3, H, W)
#
#         # Convert mask to tensor and binarize (all nonzero values become 1)
#         label = T.ToTensor()(mask).squeeze(0)
#         label = (label > 0).float()
#
#         return {"pixel_values": pixel_values,
#                 "labels": label}
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
#         pixel_values = batch["pixel_values"].to(device)  # shape: (B, 3, H, W)
#
#         with autocast():
#             outputs = model(pixel_values=pixel_values, return_dict=True)
#             # Post-process the semantic segmentation output
#             # For each image, set target size as (H, W) using image_size reversed (i.e., (height, width))
#             target_sizes = [dataset.image_size[::-1]] * pixel_values.size(0)
#             pred_maps_list = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
#
#         # For binary segmentation, assume that class 1 corresponds to the "leaf" class
#         for pred_map in pred_maps_list:
#             pred_mask = (pred_map == 1).long()  # Binary mask: 1 if predicted class is leaf, else 0
#             pred_masks.append(pred_mask.cpu().numpy())
#
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
# def plot_rotation_sweep(angles, pixel_accs, mious, save_path="mask2former_rotation_results.png"):
#     plt.figure(figsize=(8, 6))
#     plt.plot(angles, pixel_accs, marker='o', label='Pixel Accuracy')
#     plt.plot(angles, mious, marker='s', label='Mean IoU')
#     plt.title("Mask2Former Segmentation vs. Rotation Angle")
#     plt.xlabel("Rotation Angle (degrees)")
#     plt.ylabel("Metric Value")
#     plt.ylim([0, 1])
#     plt.grid(True)
#     plt.legend()
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
#     angles = list(range(0, 361, 30))  # 0°, 30°, 60°, ..., 180°
#     pixel_acc_list, miou_list = [], []
#
#     for angle in angles:
#         print(f"\n=== Evaluating with rotation angle={angle}° ===")
#         # Create dataset with the current rotation angle
#         dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, processor, image_size=(352, 352), angle=angle)
#
#         preds, gts = run_inference(dataset)
#         pix_acc, miou = compute_metrics(preds, gts)
#         pixel_acc_list.append(pix_acc)
#         miou_list.append(miou)
#         print(f"Results (angle={angle}°): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")
#
#     plot_rotation_sweep(angles, pixel_acc_list, miou_list)
#
#     results_df = pd.DataFrame({
#         "rotation_angle": angles,
#         "pixel_accuracy": pixel_acc_list,
#         "mean_iou": miou_list
#     })
#     csv_path = "mask2former_rotation_results.csv"
#     results_df.to_csv(csv_path, index=False)
#     print(f"CSV saved: {csv_path}")
#
#
# if __name__ == "__main__":
#     main()
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Load Mask2Former model and processor
model_path = "./mask2former_finetuned_leaf"
processor = AutoImageProcessor.from_pretrained(model_path)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH ROTATION
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor, image_size=(352, 352), angle=0):
        self.images_dir  = images_dir
        self.masks_dir   = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor   = processor
        self.image_size  = image_size
        self.angle       = angle

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Open image & rotate
        image = Image.open(img_path).convert("RGB")
        image = image.rotate(self.angle, resample=Image.BILINEAR, expand=True)
        image = image.resize(self.image_size, Image.BILINEAR)

        # Open mask & rotate
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.rotate(self.angle, resample=Image.NEAREST, expand=True)
        mask = mask.resize(self.image_size, Image.NEAREST)

        # Prepare model input
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        # Binarize mask
        label = T.ToTensor()(mask).squeeze(0)
        label = (label > 0).float()

        return {"pixel_values": pixel_values, "labels": label}

# ---------------------------
# 3) INFERENCE FUNCTION (Predictions Only)
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    pred_masks = []

    model.eval()
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        with autocast():
            outputs = model(pixel_values=pixel_values, return_dict=True)
            target_sizes = [dataset.image_size[::-1]] * pixel_values.size(0)
            preds_list = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )

        for pred_map in preds_list:
            # binary mask: leaf class == 1
            pred_mask = (pred_map == 1).long()
            pred_masks.append(pred_mask.cpu().numpy())

    return pred_masks

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

    # pixel accuracy
    correct = np.diag(cm).sum()
    total   = cm.sum()
    pixel_acc = correct / (total + 1e-10)

    # IoU for leaf class (1)
    intersection = cm[1, 1]
    union = cm[1, :].sum() + cm[:, 1].sum() - intersection
    mean_iou = intersection / (union + 1e-10)

    return pixel_acc, mean_iou

# ---------------------------
# 5) HELPER: Inverse Rotate Mask
# ---------------------------
def inverse_rotate_mask(mask, angle, image_size=(352, 352)):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.rotate(-angle, resample=Image.NEAREST, expand=False)
    mask_np  = (np.array(mask_img) > 128).astype(np.uint8)
    return mask_np

# ---------------------------
# 6) PLOTTING RESULTS
# ---------------------------
def plot_rotation_sweep(angles, pixel_accs, mious, save_path="mask2former_rotation_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(angles, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(angles, mious,       marker='s', label='Mean IoU')
    plt.title("Segmentation Consistency vs. Rotation Angle")
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
    # Baseline (angle=0)
    print("Running baseline inference (angle=0)...")
    baseline_ds   = LeafSegFineTuneDataset(
        test_images_dir, test_masks_dir, processor,
        image_size=(352, 352), angle=0
    )
    baseline_preds = run_inference(baseline_ds)

    angles         = list(range(0, 361, 30))
    pixel_acc_list = []
    miou_list      = []

    for angle in angles:
        print(f"\n=== Evaluating at rotation angle = {angle}° ===")
        ds = LeafSegFineTuneDataset(
            test_images_dir, test_masks_dir, processor,
            image_size=(352, 352), angle=angle
        )
        rotated_preds = run_inference(ds)

        # inverse-rotate back
        adjusted_preds = []
        for pred in rotated_preds:
            if angle != 0:
                adjusted = inverse_rotate_mask(pred, angle)
            else:
                adjusted = pred
            adjusted_preds.append(adjusted)

        pa, miou = compute_metrics(adjusted_preds, baseline_preds)
        pixel_acc_list.append(pa)
        miou_list.append(miou)
        print(f"Angle {angle}° → PixelAcc={pa:.4f}, MeanIoU={miou:.4f}")

    # Plot both metrics
    plot_rotation_sweep(angles, pixel_acc_list, miou_list)

    # Save to CSV
    results_df = pd.DataFrame({
        "rotation_angle": angles,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    csv_path = "mask2former_rotation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

if __name__ == "__main__":
    main()


