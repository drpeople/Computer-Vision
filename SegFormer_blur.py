import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch.cuda.amp import autocast
import torch.nn.functional as F

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Load SegFormer fine-tuned model from directory
model_path = "./segformer_finetuned_leaf"  # Ensure this path exists and contains your fine-tuned model & processor
processor = SegformerImageProcessor.from_pretrained(model_path)
model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

# Ground truth threshold for mask binarization (lower than 128/255)
GT_THRESHOLD = 0.05


# ---------------------------
# 2) LEAF SEGMENTATION DATASET
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor, image_size=(512, 512), sigma=0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor = processor
        self.image_size = image_size  # (width, height)
        self.sigma = sigma  # Gaussian blur sigma

        # Define a mask transformation: resize (using nearest neighbor) and convert to tensor.
        self.mask_transform = T.Compose([
            T.Resize(self.image_size, interpolation=Image.NEAREST),
            T.ToTensor()  # Converts mask to [0,1] float (dividing by 255 if image is uint8)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)
        if self.sigma > 0:
            image = apply_blur(image, self.sigma)

        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")

        # Apply mask transformation and lower threshold for binarization
        mask = self.mask_transform(mask)  # shape: (1, H, W)
        mask = (mask > GT_THRESHOLD).float().squeeze(0)  # shape: (H, W)

        # Use the SegFormer processor to preprocess the image.
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return {"pixel_values": pixel_values, "labels": mask}


# ---------------------------
# 3) TRANSFORM WITH BLUR OPTION
# ---------------------------
def apply_blur(image, sigma):
    """Applies Gaussian blur if sigma > 0."""
    if sigma > 0:
        return T.functional.gaussian_blur(image, kernel_size=5, sigma=sigma)
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
        # Move tensor to device
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast():
            outputs = model(pixel_values=batch["pixel_values"], return_dict=True)
            logits = outputs.logits  # shape: (B, num_labels, H_out, W_out)
            # Upsample logits to match ground truth size
            logits = F.interpolate(logits, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)  # shape: (B, H, W)

        # Get ground truth masks
        gt_np = batch["labels"].cpu().numpy()
        for p, g in zip(preds.cpu().numpy(), gt_np):
            pred_masks.append(p)
            gt_masks.append(g)

    # Debug: print unique values of first prediction and ground truth
    if len(pred_masks) > 0:
        print("Unique values in predicted mask (first image):", np.unique(pred_masks[0]))
        print("Unique values in ground truth mask (first image):", np.unique(gt_masks[0].astype(np.int32)))

    return pred_masks, gt_masks


# ---------------------------
# 5) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(pred_masks, gt_masks):
    pixel_accs = []
    ious = []
    # Compute per-image metrics and average
    for pred, gt in zip(pred_masks, gt_masks):
        # Ensure binary masks (assuming class 1 is the leaf)
        pred_binary = (pred == 1).astype(np.int32)
        gt_binary = gt.astype(np.int32)

        # Pixel accuracy: proportion of matching pixels
        pixel_acc = np.mean(pred_binary == gt_binary)

        # Compute IoU: Intersection over Union for the leaf class
        intersection = np.sum((pred_binary == 1) & (gt_binary == 1))
        union = np.sum((pred_binary == 1) | (gt_binary == 1))
        iou = intersection / union if union > 0 else 0

        pixel_accs.append(pixel_acc)
        ious.append(iou)
    return np.mean(pixel_accs), np.mean(ious)


# ---------------------------
# 6) PLOTTING RESULTS
# ---------------------------
def plot_blur_sweep(blur_levels, pixel_accs, mious, save_path="segformer_blur_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(blur_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(blur_levels, mious, marker='s', label='Mean IoU')
    plt.title("SegFormer Segmentation vs. Gaussian Blur Sigma")
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
    # Define blur levels to test
    blur_levels = [0, 1, 2, 3, 4]
    pixel_acc_list, miou_list = [], []

    for sigma in blur_levels:
        print(f"\n=== Evaluating with blur sigma={sigma} ===")
        dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, processor, image_size=(512, 512), sigma=sigma)

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
    results_df.to_csv("segformer_blur_results.csv", index=False)
    print("CSV saved: segformer_blur_results.csv")


if __name__ == "__main__":
    main()
