import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Load SqueezeNet-based segmentation model from directory
model_path = "./squeezenet_segmentation_finetuned"  # Ensure this directory contains your saved model


class SqueezeNetSegmentation(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load SqueezeNet (no pretrained weights needed as we load the finetuned state)
        self.squeezenet = models.squeezenet1_1(pretrained=False)
        self.features = self.squeezenet.features  # Feature extractor
        # Segmentation head: 1x1 convolution to predict num_classes channels
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)  # shape: (B, num_classes, H_feat, W_feat)
        # Dynamically upsample to match input resolution
        up_logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return up_logits


model = SqueezeNetSegmentation(num_classes=2)
model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=device))
model.to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

# Ground truth threshold for mask binarization
GT_THRESHOLD = 0.05


# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH NOISE
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(512, 512), noise_std=0.0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.image_size = image_size  # (width, height)
        self.noise_std = noise_std  # Standard deviation for Gaussian noise

        # Define a mask transformation: resize (using nearest neighbor) and convert to tensor.
        self.mask_transform = T.Compose([
            T.Resize(self.image_size, interpolation=Image.NEAREST),
            T.ToTensor()  # Converts mask to [0,1] float
        ])
        # Define an image transformation: resize and convert to tensor.
        self.image_transform = T.Compose([
            T.Resize(self.image_size, interpolation=Image.BILINEAR),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)

        # If noise_std > 0, add Gaussian noise to the image.
        if self.noise_std > 0:
            # Convert PIL image to tensor in [0,1]
            tensor_img = TF.to_tensor(image)
            # Generate Gaussian noise and add it to the image
            noise = torch.randn(tensor_img.size()) * self.noise_std
            noisy_img = tensor_img + noise
            noisy_img = torch.clamp(noisy_img, 0, 1)
            # Convert tensor back to PIL image
            image = TF.to_pil_image(noisy_img)

        # Apply image transformation to get a tensor input for the model.
        image = self.image_transform(image)

        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.image_size, Image.NEAREST)
        mask = self.mask_transform(mask)  # shape: (1, H, W)
        mask = (mask > GT_THRESHOLD).float().squeeze(0)  # shape: (H, W)

        return {"pixel_values": image, "labels": mask}


# ---------------------------
# 3) INFERENCE FUNCTION
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    """Runs inference with mixed precision for efficiency."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    pred_masks, gt_masks = [], []

    model.eval()
    for batch in loader:
        # Move tensors to device
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast():
            outputs = model(batch["pixel_values"])
            # Model output should match the input resolution; if not, interpolate:
            logits = F.interpolate(outputs, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1)  # shape: (B, H, W)

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
# 4) METRICS (Pixel Accuracy & Mean IoU)
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
# 5) PLOTTING RESULTS
# ---------------------------
def plot_noise_sweep(noise_levels, pixel_accs, mious, save_path="squeezenet_noise_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(noise_levels, mious, marker='s', label='Mean IoU')
    plt.title("SqueezeNet Segmentation vs. Gaussian Noise Std")
    plt.xlabel("Noise Std")
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
    # Define noise levels to test; adjust these values as needed.
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    pixel_acc_list, miou_list = [], []

    for noise_std in noise_levels:
        print(f"\n=== Evaluating with noise_std={noise_std} ===")
        dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, image_size=(512, 512), noise_std=noise_std)

        preds, gts = run_inference(dataset)
        pix_acc, miou = compute_metrics(preds, gts)

        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)

        print(f"Results (noise_std={noise_std}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    plot_noise_sweep(noise_levels, pixel_acc_list, miou_list)

    results_df = pd.DataFrame({
        "noise_std": noise_levels,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    results_df.to_csv("squeezenet_noise_results.csv", index=False)
    print("CSV saved: squeezenet_noise_results.csv")


if __name__ == "__main__":
    main()
