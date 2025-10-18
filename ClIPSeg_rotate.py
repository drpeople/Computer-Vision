# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
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
# # Load CLIPSeg model
# model_path = "./clipseg_finetuned_leaf"  # Ensure this path exists
# processor = CLIPSegProcessor.from_pretrained(model_path)
# model = CLIPSegForImageSegmentation.from_pretrained(model_path).to(device)
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
#     def __init__(self, images_dir, masks_dir, processor, image_size=(352, 352), prompt="a photo of a leaf", angle=0):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.image_files = sorted(os.listdir(images_dir))
#         self.processor = processor
#         self.image_size = image_size
#         self.prompt = prompt
#         self.angle = angle  # Rotation angle in degrees
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#
#         image = Image.open(img_path).convert("RGB")
#         # Rotate image with expand=True and then resize to desired image size
#         image = image.rotate(self.angle, resample=Image.BILINEAR, expand=True)
#         image = image.resize(self.image_size, Image.BILINEAR)
#
#         mask_name = img_name.replace('.jpg', '.png')
#         mask_path = os.path.join(self.masks_dir, mask_name)
#         mask = Image.open(mask_path).convert("L")
#         # Rotate mask using nearest neighbor interpolation and then resize
#         mask = mask.rotate(self.angle, resample=Image.NEAREST, expand=True)
#         mask = mask.resize(self.image_size, Image.NEAREST)
#
#         inputs = self.processor(text=[self.prompt], images=[image], padding="max_length", return_tensors="pt")
#         pixel_values = inputs["pixel_values"].squeeze(0)
#         input_ids = inputs["input_ids"].squeeze(0)
#         attention_mask = inputs["attention_mask"].squeeze(0)
#
#         # Convert mask to tensor and binarize (assuming non-zero is leaf)
#         label = T.ToTensor()(mask).squeeze(0)
#         label = (label > 0).float()
#
#         return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": label}
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
#         batch = {k: v.to(device) for k, v in batch.items()}
#
#         with autocast():
#             outputs = model(pixel_values=batch["pixel_values"],
#                             input_ids=batch["input_ids"],
#                             attention_mask=batch["attention_mask"],
#                             return_dict=True)
#             logits = outputs.logits
#             preds = torch.sigmoid(logits) > 0.5
#             preds = preds.cpu().numpy()
#
#         gt_np = batch["labels"].cpu().numpy()
#
#         for p, g in zip(preds, gt_np):
#             pred_masks.append(p)
#             gt_masks.append(g)
#
#     return pred_masks, gt_masks
#
# # ---------------------------
# # 4) METRICS (Pixel Accuracy & Mean IoU)
# # ---------------------------
# def compute_metrics(pred_masks, gt_masks):
#     pixel_accs = []
#     ious = []
#     # Compute per-image metrics and average
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
# # ---------------------------
# # 5) PLOTTING RESULTS
# # ---------------------------
# def plot_rotation_sweep(angles, pixel_accs, mious, save_path="clipseg_rotation_results.png"):
#     plt.figure(figsize=(8, 6))
#     plt.plot(angles, pixel_accs, marker='o', label='Pixel Accuracy')
#     plt.plot(angles, mious, marker='s', label='Mean IoU')
#     plt.title("CLIPSeg Segmentation vs. Rotation Angle")
#     plt.xlabel("Rotation Angle (degrees)")
#     plt.ylabel("Metric Value")
#     plt.ylim([0, 1])
#     plt.grid(True)
#     plt.legend()
#     plt.savefig(save_path)
#     plt.show()
#     print(f"Plot saved to: {save_path}")
#
# # ---------------------------
# # 6) MAIN SCRIPT
# # ---------------------------
# def main():
#     # Define a list of rotation angles (in degrees) to evaluate.
#     angles = list(range(0, 181, 30))  # 0°, 30°, 60°, ..., 180°
#     pixel_acc_list, miou_list = [], []
#
#     for angle in angles:
#         print(f"\n=== Evaluating with rotation angle={angle}° ===")
#         # Pass the current rotation angle to the dataset so that images are rotated accordingly.
#         dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, processor, image_size=(352, 352), angle=angle)
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
#     csv_path = "clipseg_rotation_results.csv"
#     results_df.to_csv(csv_path, index=False)
#     print(f"CSV saved: {csv_path}")
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
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
from torchvision.transforms import functional as TF

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Load CLIPSeg model
model_path = "./clipseg_finetuned_leaf"  # Ensure this path exists
processor = CLIPSegProcessor.from_pretrained(model_path)
model = CLIPSegForImageSegmentation.from_pretrained(model_path).to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH ROTATION
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor, image_size=(352, 352), prompt="a photo of a leaf", angle=0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor = processor
        self.image_size = image_size
        self.prompt = prompt
        self.angle = angle

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = image.rotate(self.angle, resample=Image.BILINEAR, expand=True)
        image = image.resize(self.image_size, Image.BILINEAR)

        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.rotate(self.angle, resample=Image.NEAREST, expand=True)
        mask = mask.resize(self.image_size, Image.NEAREST)

        inputs = self.processor(text=[self.prompt], images=[image], padding="max_length", return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        label = T.ToTensor()(mask).squeeze(0)
        label = (label > 0).float()

        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": label}

# ---------------------------
# 3) INFERENCE FUNCTION
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    pred_masks = []
    gt_masks = []

    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast():
            outputs = model(pixel_values=batch["pixel_values"],
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            return_dict=True)
            logits = outputs.logits
            preds = torch.sigmoid(logits) > 0.5
            preds = preds.cpu().numpy()
            labels = batch["labels"].cpu().numpy()

        for p, l in zip(preds, labels):
            pred_masks.append(p)
            gt_masks.append(l)

    return pred_masks, gt_masks

# ---------------------------
# 4) METRICS: Mean IoU and Pixel Accuracy
# ---------------------------
def compute_metrics(pred_masks, gt_masks):
    ious = []
    correct = 0
    total = 0

    for pred, gt in zip(pred_masks, gt_masks):
        pred_flat = pred.flatten().astype(int)
        gt_flat = gt.flatten().astype(int)

        cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
        correct += (pred_flat == gt_flat).sum()
        total += gt_flat.size

        iou = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0] + 1e-10)
        ious.append(iou)

    pixel_acc = correct / (total + 1e-10)
    mean_iou = np.mean(ious)
    return pixel_acc, mean_iou

# ---------------------------
# 5) HELPER FUNCTION: Inverse Rotate Mask
# ---------------------------
def inverse_rotate_mask(mask, angle, image_size=(352, 352)):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.rotate(-angle, resample=Image.NEAREST, expand=False)
    mask_img = mask_img.resize(image_size, Image.NEAREST)
    mask_np = (np.array(mask_img) > 128).astype(np.uint8)
    return mask_np

# ---------------------------
# 6) PLOTTING RESULTS
# ---------------------------
def plot_rotation_sweep(angles, pixel_accs, mious, save_path="clipseg_rotation_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(angles, pixel_accs, marker='s', label='Pixel Accuracy')
    plt.plot(angles, mious, marker='o', label='Mean IoU')
    plt.title("CLIPSeg Segmentation Performance vs. Rotation Angle")
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
    baseline_dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, processor, image_size=(352, 352), prompt="a photo of a leaf", angle=0)
    baseline_preds, baseline_gts = run_inference(baseline_dataset)

    angles = list(range(0, 361, 30))
    pixel_acc_list = []
    miou_list = []

    for angle in angles:
        print(f"\n=== Evaluating with rotation angle = {angle}° ===")
        dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, processor, image_size=(352, 352), prompt="a photo of a leaf", angle=angle)
        rotated_preds, rotated_gts = run_inference(dataset)

        adjusted_preds = []
        for pred in rotated_preds:
            if angle != 0:
                adjusted_pred = inverse_rotate_mask(pred, angle)
            else:
                adjusted_pred = pred
            adjusted_preds.append(adjusted_pred)

        pixel_acc, miou = compute_metrics(adjusted_preds, baseline_preds)
        pixel_acc_list.append(pixel_acc)
        miou_list.append(miou)
        print(f"Angle {angle}°: Pixel Accuracy = {pixel_acc:.4f}, Mean IoU = {miou:.4f}")

    plot_rotation_sweep(angles, pixel_acc_list, miou_list)

    results_df = pd.DataFrame({
        "rotation_angle": angles,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    csv_path = "clipseg_rotation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

if __name__ == "__main__":
    main()
