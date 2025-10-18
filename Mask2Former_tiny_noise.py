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
model_path = "./mask2former_finetuned_leaf"  # Update this path accordingly
processor = AutoImageProcessor.from_pretrained(model_path)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"


# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH NOISE
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor, image_size=(352, 352), noise_std=0.0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor = processor
        self.image_size = image_size
        self.noise_std = noise_std  # Standard deviation for Gaussian noise

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Open and resize the image
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)

        # Optionally add Gaussian noise
        if self.noise_std > 0:
            tensor_img = TF.to_tensor(image)  # values in [0,1]
            noise = torch.randn(tensor_img.size()) * self.noise_std
            noisy_img = tensor_img + noise
            noisy_img = torch.clamp(noisy_img, 0, 1)
            image = TF.to_pil_image(noisy_img)

        # Load and resize mask (assumes mask file has same base name with .png extension)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.image_size, Image.NEAREST)

        # Process the image using the Mask2Former processor (no text prompt needed)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # shape: (3, H, W)

        # Convert mask to tensor and binarize: any value > 0 becomes 1
        label = T.ToTensor()(mask).squeeze(0)
        label = (label > 0).float()

        return {"pixel_values": pixel_values, "labels": label}


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
        pixel_values = batch["pixel_values"].to(device)  # shape: (B, 3, H, W)

        with autocast():
            outputs = model(pixel_values=pixel_values, return_dict=True)
            # Post-process to get semantic segmentation maps.
            # Provide target_sizes for each image (using image_size reversed: (H, W)).
            target_sizes = [dataset.image_size[::-1]] * pixel_values.size(0)
            pred_maps_list = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        # For binary segmentation, assume label 1 corresponds to the "leaf" class.
        for pred_map in pred_maps_list:
            pred_mask = (pred_map == 1).long()  # binary mask: 1 if class==1, else 0
            pred_masks.append(pred_mask.cpu().numpy())

        for label in batch["labels"]:
            gt_masks.append(label.cpu().numpy())

    return pred_masks, gt_masks


# ---------------------------
# 4) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(pred_masks, gt_masks):
    pixel_accs = []
    ious = []
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
# 5) PLOTTING RESULTS
# ---------------------------
def plot_noise_sweep(noise_levels, pixel_accs, mious, save_path="mask2former_noise_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(noise_levels, mious, marker='s', label='Mean IoU')
    plt.title("Mask2Former Segmentation vs. Gaussian Noise Std")
    plt.xlabel("Noise Std")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()


# ---------------------------
# 6) MAIN SCRIPT
# ---------------------------
def main():
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    pixel_acc_list, miou_list = [], []

    for std in noise_levels:
        print(f"\n=== Evaluating with noise std={std} ===")
        dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, processor, image_size=(352, 352),
                                         noise_std=std)
        preds, gts = run_inference(dataset)
        pix_acc, miou = compute_metrics(preds, gts)
        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)
        print(f"Results (noise_std={std}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    plot_noise_sweep(noise_levels, pixel_acc_list, miou_list)

    results_df = pd.DataFrame({
        "noise_std": noise_levels,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    results_df.to_csv("mask2former_noise_results.csv", index=False)
    print("CSV saved: mask2former_noise_results.csv")


if __name__ == "__main__":
    main()
