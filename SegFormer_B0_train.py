import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from torch.cuda.amp import autocast, GradScaler

# Fixed image size – SegFormer typically expects images to be resized (width, height)
IMAGE_SIZE = (512, 512)

# Lower threshold to accommodate mask values up to ~38/255 ~= 0.15
GT_THRESHOLD = 0.05

# --- Custom Dataset for Fine-Tuning with SegFormer ---
class LeafSegFineTuneDataset(Dataset):
    """
    Dataset for leaf segmentation fine-tuning.
    Loads an image and its corresponding binary segmentation mask.
    The mask is assumed to be a grayscale image where nonzero values indicate the leaf.
    """

    def __init__(self, images_dir, masks_dir, image_size=IMAGE_SIZE):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.image_size = image_size

        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            # Normalization can be done by the processor if desired.
        ])

        # Mask transforms
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),  # Converts mask to [0,1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        # Load mask (assumes same base name + ".png")
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = self.mask_transform(mask)

        # Binarize mask at GT_THRESHOLD
        mask = (mask > GT_THRESHOLD).float().squeeze(0)  # shape: (H, W)

        return {
            "pixel_values": image,  # (3, H, W)
            "labels": mask          # (H, W)
        }

# --- Training and Validation Functions ---
def train_one_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device).long()  # LongTensor for segmentation loss

        optimizer.zero_grad()
        with autocast():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device).long()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(dataloader)

# --- Main Fine-Tuning Script ---
def main():
    # Update these paths to your training and validation data directories
    train_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\images"
    train_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\masks"
    val_images_dir   = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
    val_masks_dir    = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    batch_size = 4
    num_epochs = 6
    learning_rate = 1e-4

    # Create training and validation datasets and dataloaders
    train_dataset = LeafSegFineTuneDataset(train_images_dir, train_masks_dir, image_size=IMAGE_SIZE)
    val_dataset   = LeafSegFineTuneDataset(val_images_dir, val_masks_dir, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Load SegFormer model and image processor.
    # We set num_labels=2 (background and leaf). Adjust id2label and label2id as needed.
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=2,
        id2label={0: "background", 1: "leaf"},
        label2id={"background": 0, "leaf": 1},
        ignore_mismatched_sizes=True
    )
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss   = validate_one_epoch(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the fine-tuned model and image processor
    save_dir = "./segformer_finetuned_leaf"
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print("Fine-tuning complete. Model and processor saved to:", save_dir)

if __name__ == "__main__":
    main()

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F

# Settings
IMAGE_SIZE = (512, 512)
VIS_THRESHOLD = 0.1  # Threshold for model predictions
# In training/evaluation, the model outputs probabilities. Here, we lower the threshold for binarizing GT masks.
GT_THRESHOLD = 0.05  # Lower threshold for binarizing ground-truth masks

def custom_collate_fn(batch):
    """
    Custom collate function to keep original PIL images in a list,
    while default-collating tensor items.
    """
    original_pils = [item.pop("original_pil") for item in batch]
    collated = default_collate(batch)
    collated["original_pil"] = original_pils
    return collated

# --- Dataset Definition ---
class LeafSegDataset(Dataset):
    """
    Loads the original PIL image (for plotting) and processes it for the SegFormer model.
    Converts the ground truth mask to a binary mask (0 for background, 1 for leaf).
    """
    def __init__(self, images_dir, masks_dir, processor, image_size=IMAGE_SIZE):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor = processor
        self.image_size = image_size
        # Define a transformation for the mask (resize and convert to tensor)
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST),
            transforms.ToTensor()  # Converts mask to [0,1] float (dividing by 255 if image is uint8)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Load original PIL image (for plotting)
        original_pil = Image.open(img_path).convert("RGB")
        # Resize the image for model input (the processor expects images of a given size)
        pil_image = original_pil.resize(self.image_size, Image.BILINEAR)

        # Load mask; assume corresponding mask has same base name with .png extension
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask_img = Image.open(mask_path).convert("L")

        # Debug: print unique pixel values in the raw mask
        mask_vals = set(mask_img.getdata())
        print(f"[DEBUG] Mask file: {mask_name}, unique pixel values: {mask_vals}")

        # Transform mask and apply lower threshold for binarization
        mask = self.mask_transform(mask_img)  # shape: (1, H, W), values in [0,1]
        mask = (mask > GT_THRESHOLD).float().squeeze(0)  # shape: (H, W)

        # Debug: print unique values after thresholding
        unique_vals_after_thresh = torch.unique(mask)
        print(f"[DEBUG] Mask file: {mask_name}, unique values after threshold: {unique_vals_after_thresh.tolist()}")

        # Process the image using the SegformerImageProcessor
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # shape: (3, H, W)

        return {
            "original_pil": original_pil,  # for plotting
            "pixel_values": pixel_values,  # for model input
            "labels": mask                 # binary mask
        }

# --- Evaluation and Plotting ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the fine-tuned SegFormer model and its image processor
    save_dir = "./segformer_finetuned_leaf"
    processor = SegformerImageProcessor.from_pretrained(save_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(save_dir)
    model.to(device)
    model.eval()

    # Create test dataset and DataLoader with the custom collate function
    test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
    test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
    test_dataset = LeafSegDataset(test_images_dir, test_masks_dir, processor, image_size=IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    def compute_iou(pred_mask, gt_mask):
        """Compute IoU for binary masks of shape (H, W)."""
        intersection = (pred_mask & gt_mask).float().sum()
        union = (pred_mask | gt_mask).float().sum()
        if union == 0:
            return 1.0
        return (intersection / union).item()

    total_iou = 0.0
    num_samples = 0
    plot_samples = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)  # shape: (B, 3, H, W)
            labels = batch["labels"]  # (B, H, W) on CPU

            # Forward pass
            outputs = model(pixel_values=pixel_values, return_dict=True)
            logits = outputs.logits  # shape: (B, 2, H_model, W_model)

            # Convert logits to probabilities via softmax
            probs = torch.softmax(logits, dim=1)
            # Extract the probability map for the "leaf" class (channel = 1)
            leaf_probs = probs[:, 1, :, :]  # shape: (B, H_model, W_model)

            # Upsample predicted probability map to match the ground-truth resolution
            leaf_probs = F.interpolate(
                leaf_probs.unsqueeze(1),  # shape: (B,1,H_model,W_model)
                size=IMAGE_SIZE,
                mode="bilinear",
                align_corners=False
            ).squeeze(1)  # shape: (B,H,W)

            # Debug: print mean & max of leaf_probs
            print("[DEBUG] Leaf probability stats:",
                  f"mean={leaf_probs.mean().item():.6f},",
                  f"max={leaf_probs.max().item():.6f}")

            # Apply threshold to obtain binary prediction mask
            pred_masks = (leaf_probs > VIS_THRESHOLD).long()

            for i in range(pred_masks.size(0)):
                gt = labels[i].long()  # shape: (H, W), binary 0 or 1
                pred = pred_masks[i].cpu()  # shape: (H, W)
                iou = compute_iou(pred, gt)
                total_iou += iou
                num_samples += 1

                plot_samples.append({
                    "original_pil": batch["original_pil"][i],
                    "gt_mask": gt,
                    "pred_mask": pred,
                    "prob_map": leaf_probs[i].cpu()
                })

    mean_iou = total_iou / num_samples if num_samples > 0 else 0.0
    print(f"Mean IoU over test data: {mean_iou:.4f}")

    # Plot a few samples
    num_to_plot = min(4, len(plot_samples))
    for idx in range(num_to_plot):
        sample = plot_samples[idx]
        image = sample["original_pil"]  # original PIL image
        gt_mask = sample["gt_mask"]
        pred_mask = sample["pred_mask"]
        prob_map = sample["prob_map"]

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(gt_mask, cmap="gray")
        plt.title("GT Mask")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(pred_mask, cmap="gray")
        plt.title(f"Pred Mask (thresh > {VIS_THRESHOLD})")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(prob_map, cmap="viridis")
        plt.title("Probability Heatmap")
        plt.axis("off")

        plt.show()
