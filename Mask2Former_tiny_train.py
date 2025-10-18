# import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import MaskFormerConfig, MaskFormerModel
from torch.cuda.amp import autocast, GradScaler

# Fixed image size – Mask2Former typically expects images to be resized (width, height)
IMAGE_SIZE = (512, 512)

# Lower threshold to accommodate mask values up to ~38/255 ~= 0.15
GT_THRESHOLD = 0.05

# --- Helper function to move data to device ---
def move_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [move_to_device(item, device) for item in x]
    else:
        return x

# --- Custom Dataset for Fine-Tuning with Mask2Former ---
class LeafSegFineTuneDataset(Dataset):
    """
    Dataset for leaf segmentation fine-tuning.
    Loads an image and its corresponding segmentation map.
    The segmentation map is assumed to be a grayscale image where nonzero values indicate the leaf.
    """
    def __init__(self, images_dir, masks_dir, image_size=IMAGE_SIZE):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.image_size = image_size

        # For Mask2Former, we let the processor handle normalization and tensor conversion.
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BILINEAR),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST),
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

        # Binarize mask at GT_THRESHOLD.
        # Convert mask to a numpy array so that the processor can later handle it properly.
        mask_np = np.array(mask)
        mask_np = (mask_np > (GT_THRESHOLD * 255)).astype(np.int64)

        # Return the raw image and segmentation map (key renamed to "segmentation_map")
        return {
            "image": image,  # PIL Image
            "segmentation_map": mask_np  # numpy array with values 0 or 1
        }

# --- Custom collate function ---
def collate_fn(batch):
    images = [item["image"] for item in batch]
    segmentation_maps = [item["segmentation_map"] for item in batch]
    return {"images": images, "segmentation_maps": segmentation_maps}

# --- Training and Validation Functions ---
def train_one_epoch(model, dataloader, optimizer, device, scaler, processor):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        # Use the processor to prepare inputs (this will convert images and segmentation maps
        # into a dictionary with "pixel_values", "mask_labels", and "class_labels")
        inputs = processor(
            images=batch["images"],
            segmentation_maps=batch["segmentation_maps"],
            return_tensors="pt"
        )
        # Move all tensors (and lists of tensors) to device using our helper
        inputs = {k: move_to_device(v, device) for k, v in inputs.items()}

        optimizer.zero_grad()
        with autocast():
            outputs = model(**inputs)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device, processor):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = processor(
                images=batch["images"],
                segmentation_maps=batch["segmentation_maps"],
                return_tensors="pt"
            )
            inputs = {k: move_to_device(v, device) for k, v in inputs.items()}
            outputs = model(**inputs)
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
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Load Mask2Former model and processor.
    # We set num_labels=2 (background and leaf) and provide label mappings.
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-ade-semantic",
        num_labels=2,
        id2label={0: "background", 1: "leaf"},
        label2id={"background": 0, "leaf": 1},
        ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, processor)
        val_loss = validate_one_epoch(model, val_loader, device, processor)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the fine-tuned model and processor
    save_dir = "./mask2former_finetuned_leaf"
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
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F

# Settings
IMAGE_SIZE = (512, 512)
# For visualization, we use the processor’s post-processing, so VIS_THRESHOLD is not used.
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
    Loads the original PIL image (for plotting) and processes it for the Mask2Former model.
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
            transforms.ToTensor()  # Converts mask to [0,1] float
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Load original PIL image for plotting
        original_pil = Image.open(img_path).convert("RGB")
        # Resize image for model input (the processor expects a specific size)
        pil_image = original_pil.resize(self.image_size, Image.BILINEAR)

        # Load mask; assume corresponding mask has same base name with .png extension
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask_img = Image.open(mask_path).convert("L")

        # Debug: print unique pixel values in the raw mask
        mask_vals = set(mask_img.getdata())
        print(f"[DEBUG] Mask file: {mask_name}, unique pixel values: {mask_vals}")

        # Transform mask and apply threshold for binarization
        mask = self.mask_transform(mask_img)  # shape: (1, H, W)
        mask = (mask > GT_THRESHOLD).float().squeeze(0)  # shape: (H, W)
        print(f"[DEBUG] Mask file: {mask_name}, unique values after threshold: {torch.unique(mask).tolist()}")

        # Process the image using the Mask2Former processor
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # shape: (3, H, W)

        return {
            "original_pil": original_pil,  # for plotting
            "pixel_values": pixel_values,  # model input
            "labels": mask                 # binary ground truth mask
        }

# --- Evaluation and Plotting ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the fine-tuned Mask2Former model and its processor
    save_dir = "./mask2former_finetuned_leaf"  # update to your saved model directory
    processor = AutoImageProcessor.from_pretrained(save_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(save_dir)
    model.to(device)
    model.eval()

    # Create test dataset and DataLoader using the custom collate function
    test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
    test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
    test_dataset = LeafSegDataset(test_images_dir, test_masks_dir, processor, image_size=IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    def compute_iou(pred_mask, gt_mask):
        """Compute IoU for binary masks of shape (H, W)."""
        intersection = ((pred_mask == 1) & (gt_mask == 1)).float().sum()
        union = ((pred_mask == 1) | (gt_mask == 1)).float().sum()
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

            # Forward pass through Mask2Former model
            outputs = model(pixel_values=pixel_values, return_dict=True)
            # Use the processor to post-process the semantic segmentation output.
            # Provide a target size for each image (here, using IMAGE_SIZE reversed to (H, W))
            target_sizes = [IMAGE_SIZE[::-1]] * pixel_values.size(0)
            pred_maps_list = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            # Each element in pred_maps_list is a tensor of shape (H, W) with predicted class IDs.
            # For binary segmentation, assume label 1 corresponds to "leaf".
            for i, pred_map in enumerate(pred_maps_list):
                pred_mask = (pred_map == 1).long()
                gt = labels[i].long()
                iou = compute_iou(pred_mask, gt)
                total_iou += iou
                num_samples += 1

                plot_samples.append({
                    "original_pil": batch["original_pil"][i],
                    "gt_mask": gt,
                    "pred_mask": pred_mask,
                    "pred_map": pred_map
                })

    mean_iou = total_iou / num_samples if num_samples > 0 else 0.0
    print(f"Mean IoU over test data: {mean_iou:.4f}")

    # Plot a few samples
    num_to_plot = min(4, len(plot_samples))
    for idx in range(num_to_plot):
        sample = plot_samples[idx]
        image = sample["original_pil"]
        gt_mask = sample["gt_mask"]
        pred_mask = sample["pred_mask"]
        pred_map = sample["pred_map"]

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
        plt.title("Pred Mask (Leaf class)")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(pred_map, cmap="viridis")
        plt.title("Predicted Semantic Map")
        plt.axis("off")

        plt.show()
