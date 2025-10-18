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
from torchvision.transforms import functional as TF

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Load Mask2Former model and processor
model_path = "./mask2former_finetuned_leaf"  # Update this path to your saved model
processor = AutoImageProcessor.from_pretrained(model_path)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"


# ---------------------------
# 2) LEAF SEGMENTATION DATASET
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    """
    Loads the original PIL image (for plotting) and processes it for the Mask2Former model.
    Converts the ground truth mask to a binary mask (0 for background, 1 for leaf).
    """

    def __init__(self, images_dir, masks_dir, processor, image_size=(352, 352), sigma=0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor = processor
        self.image_size = image_size
        self.sigma = sigma  # Gaussian blur sigma

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        # Apply resizing and optional blur
        image = image.resize(self.image_size, Image.BILINEAR)
        if self.sigma > 0:
            image = apply_blur(image, self.sigma)

        # Load mask; assume corresponding mask has same base name with .png extension
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.image_size, Image.NEAREST)

        # Process the image using the Mask2Former processor (text is not used)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # shape: (3, H, W)

        # Convert mask to tensor and binarize (values > 0 become 1)
        label = T.ToTensor()(mask).squeeze(0)
        label = (label > 0).float()

        return {"pixel_values": pixel_values, "labels": label}


# ---------------------------
# 3) TRANSFORM WITH BLUR OPTION
# ---------------------------
def apply_blur(image, sigma):
    """Applies Gaussian blur if sigma > 0."""
    if sigma > 0:
        return TF.gaussian_blur(image, kernel_size=5, sigma=sigma)
    return image


# ---------------------------
# 4) INFERENCE FUNCTION
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    """Runs inference with mixed precision for efficiency."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    pred_masks, gt_masks = [], []

    model.eval()
    for batch in loader:
        # Move pixel_values to device
        pixel_values = batch["pixel_values"].to(device)

        with autocast():
            outputs = model(pixel_values=pixel_values, return_dict=True)
            # Use the processor to post-process the semantic segmentation output.
            # Provide target_sizes for each image (here, using image_size reversed: (H, W))
            target_sizes = [dataset.image_size[::-1]] * pixel_values.size(0)
            pred_maps_list = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        # For binary segmentation, assume label 1 corresponds to "leaf"
        for pred_map in pred_maps_list:
            pred_mask = (pred_map == 1).long()  # Binary mask: 1 if predicted class is leaf, else 0
            pred_masks.append(pred_mask.cpu().numpy())

        # Ground truth masks are already binarized (on CPU)
        for label in batch["labels"]:
            gt_masks.append(label.cpu().numpy())

    return pred_masks, gt_masks


# ---------------------------
# 5) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(pred_masks, gt_masks):
    pixel_accs = []
    ious = []
    # Compute per-image metrics and average
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
# 6) PLOTTING RESULTS
# ---------------------------
def plot_blur_sweep(blur_levels, pixel_accs, mious, save_path="mask2former_blur_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(blur_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(blur_levels, mious, marker='s', label='Mean IoU')
    plt.title("Mask2Former Segmentation vs. Gaussian Blur Sigma")
    plt.xlabel("Blur Sigma")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to: {save_path}")


# ---------------------------
# 7) MAIN SCRIPT
# ---------------------------
def main():
    blur_levels = [0, 1, 2, 3, 4]
    pixel_acc_list, miou_list = [], []

    for sigma in blur_levels:
        print(f"\n=== Evaluating with blur sigma={sigma} ===")
        # Create dataset with the current blur sigma
        dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, processor, image_size=(352, 352), sigma=sigma)

        preds, gts = run_inference(dataset)
        pix_acc, miou = compute_metrics(preds, gts)

        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)

        print(f"Results (sigma={sigma}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    plot_blur_sweep(blur_levels, pixel_acc_list, miou_list)

    results_df = pd.DataFrame({
        "sigma": blur_levels,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    results_df.to_csv("mask2former_blur_results.csv", index=False)
    print("CSV saved: mask2former_blur_results.csv")


if __name__ == "__main__":
    main()
