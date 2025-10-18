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
import torch.nn.functional as F

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Load Mask2Former model and processor
model_path = "./mask2former_finetuned_leaf"  # Ensure this path exists
processor = AutoImageProcessor.from_pretrained(model_path)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"


# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH SCALING
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor, base_image_size=(352, 352), scale_factor=1.0):
        """
        Args:
            images_dir: Directory containing input images.
            masks_dir: Directory containing corresponding segmentation masks.
            processor: The Mask2Former processor to prepare model inputs.
            base_image_size: The size to which images are first resized.
            scale_factor: Factor to scale the image (and mask) dimensions.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor = processor
        self.base_image_size = base_image_size
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Open and resize image to base size
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.base_image_size, Image.BILINEAR)
        # Apply scaling if needed
        if self.scale_factor != 1.0:
            new_size = (int(self.base_image_size[0] * self.scale_factor),
                        int(self.base_image_size[1] * self.scale_factor))
            image = image.resize(new_size, Image.BILINEAR)

        # Process corresponding mask
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.base_image_size, Image.NEAREST)
        if self.scale_factor != 1.0:
            new_size = (int(self.base_image_size[0] * self.scale_factor),
                        int(self.base_image_size[1] * self.scale_factor))
            mask = mask.resize(new_size, Image.NEAREST)

        # Use the processor to prepare model inputs.
        # Note: Mask2Former processor does its own resizing; text prompt is not required.
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # shape: (3, H, W)

        # Convert mask to tensor and binarize (nonzero becomes 1)
        label = T.ToTensor()(mask).squeeze(0)
        label = (label > 0).float()

        return {"pixel_values": pixel_values, "labels": label}


# ---------------------------
# 3) INFERENCE FUNCTION
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    """Runs inference with mixed precision for efficiency."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    pred_masks, gt_masks = [], []

    model.eval()
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)  # shape: (B, 3, H, W)

        with autocast():
            outputs = model(pixel_values=pixel_values, return_dict=True)
            # Post-process semantic segmentation output
            # Provide target sizes for each image: (height, width) derived from the dataset's scaled size.
            target_sizes = [dataset.base_image_size[::-1] if dataset.scale_factor == 1.0
                            else (int(dataset.base_image_size[0] * dataset.scale_factor),
                                  int(dataset.base_image_size[1] * dataset.scale_factor))
                            for _ in range(pixel_values.size(0))]
            pred_maps_list = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

        # For binary segmentation, assume class 1 corresponds to "leaf"
        for pred_map in pred_maps_list:
            pred_mask = (pred_map == 1).long()  # Binary mask: 1 if predicted class is leaf, else 0
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
        # If shapes don't match, resize gt to match pred
        if pred.shape != gt.shape:
            gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float()
            gt_tensor = F.interpolate(gt_tensor, size=pred.shape, mode='nearest')
            gt = gt_tensor.squeeze().numpy()
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
        pixel_acc = np.diag(cm).sum() / np.sum(cm)
        iou = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0] + 1e-10)
        pixel_accs.append(pixel_acc)
        ious.append(iou)
    return np.mean(pixel_accs), np.mean(ious)


# ---------------------------
# 5) PLOTTING: Metrics vs. Scale Factor
# ---------------------------
def plot_scale_sweep(scale_factors, pixel_accs, mious, save_path="mask2former_scale_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(scale_factors, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(scale_factors, mious, marker='s', label='Mean IoU')
    plt.title("Mask2Former Segmentation vs. Scale Factor")
    plt.xlabel("Scale Factor")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to: {save_path}")


# ---------------------------
# 6) MAIN SCRIPT
# ---------------------------
def main():
    # Define the scale factors to test.
    scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]
    pixel_acc_list, miou_list = [], []

    for scale in scale_factors:
        print(f"\n=== Evaluating with scale_factor={scale} ===")
        dataset = LeafSegFineTuneDataset(
            images_dir=test_images_dir,
            masks_dir=test_masks_dir,
            processor=processor,
            base_image_size=(352, 352),
            scale_factor=scale
        )

        preds, gts = run_inference(dataset)
        pix_acc, miou = compute_metrics(preds, gts)

        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)

        print(f"Results (scale_factor={scale}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    plot_scale_sweep(scale_factors, pixel_acc_list, miou_list)

    results_df = pd.DataFrame({
        "scale_factor": scale_factors,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    csv_path = "mask2former_scale_results.csv"
    results_df.to_csv(csv_path, index=False)
    print("CSV saved:", csv_path)


if __name__ == "__main__":
    main()
