import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Fixed image size – must match what the model expects (CIDAS/clipseg-rd64-refined uses 352×352)
IMAGE_SIZE = (352, 352)

# --- Custom Dataset for Fine-Tuning (without data augmentation) ---
class LeafSegFineTuneDataset(Dataset):
    """
    Dataset for leaf segmentation fine-tuning.
    For each sample, it loads the image and its ground-truth mask,
    resizes both to IMAGE_SIZE, and processes the image using the CLIPSegProcessor.
    """
    def __init__(self, images_dir, masks_dir, processor, image_size=IMAGE_SIZE, prompt="a photo of a leaf"):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor = processor
        self.image_size = image_size
        self.prompt = prompt

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        # Load and resize image
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)

        # Load and resize mask (assumes mask file has same name but with .png extension)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.image_size, Image.NEAREST)

        # Process image using the processor with the fixed text prompt
        inputs = self.processor(text=[self.prompt], images=[image], padding="max_length", return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Convert mask to tensor and remove extra channel so shape becomes (H, W)
        label = transforms.ToTensor()(mask).squeeze(0)
        # Convert any nonzero value to 1 (binary mask)
        label = (label > 0).float()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }

# --- Custom Loss Functions ---
def dice_loss(pred_logits, target, smooth=1.0):
    """
    Computes Dice loss.
    pred_logits: raw logits of shape (B, H, W)
    target: binary tensor of shape (B, H, W)
    """
    pred = torch.sigmoid(pred_logits)
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + smooth)
    return 1 - dice.mean()

bce_loss_fn = nn.BCEWithLogitsLoss()

def combined_loss(logits, target, alpha=1.0):
    loss_bce = bce_loss_fn(logits, target)
    loss_dice = dice_loss(logits, target)
    return loss_bce + alpha * loss_dice

# --- Main Fine-Tuning Script ---
if __name__ == '__main__':
    # Enable cuDNN benchmark for fixed-size inputs
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Directories for training and validation data (update paths as needed)
    train_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\images"
    train_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\masks"
    val_images_dir   = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
    val_masks_dir    = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

    # Load the CLIPSeg processor and model
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)

    # Create training and validation datasets and DataLoaders with optimization settings
    train_dataset = LeafSegFineTuneDataset(train_images_dir, train_masks_dir, processor, image_size=IMAGE_SIZE)
    val_dataset = LeafSegFineTuneDataset(val_images_dir, val_masks_dir, processor, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Set up optimizer, mixed precision training, and a cosine annealing scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader)*5, eta_min=1e-6)

    num_epochs = 6 # Increase epochs to allow better convergence
    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast():
                outputs = model(**batch, return_dict=True)
                logits = outputs.logits  # (B, H, W)
                loss = combined_loss(logits, batch["labels"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast():
                    outputs = model(**batch, return_dict=True)
                    logits = outputs.logits
                    loss = combined_loss(logits, batch["labels"])
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    # Save the fine-tuned model and processor
    save_dir = "./clipseg_finetuned_leaf"
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print("Fine-tuning complete. Model and processor saved to:", save_dir)

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# Settings
IMAGE_SIZE = (352, 352)
VIS_THRESHOLD = 0.1  # Threshold for converting predicted probabilities to binary mask

def custom_collate_fn(batch):
    """
    Custom collate function to keep original PIL images in a list,
    while default_collating tensor items.
    """
    original_pils = [item.pop("original_pil") for item in batch]
    from torch.utils.data._utils.collate import default_collate
    collated = default_collate(batch)
    collated["original_pil"] = original_pils
    return collated

# --- Dataset Definition ---
class LeafSegDataset(Dataset):
    """
    Loads the original PIL image (for plotting) and processes it for the model.
    Converts the ground truth mask to a binary mask (0 for background, 1 for leaf).
    """
    def __init__(self, images_dir, masks_dir, processor, image_size=IMAGE_SIZE, prompt="a photo of a leaf"):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor = processor
        self.image_size = image_size
        self.prompt = prompt

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        # Load original PIL image (for plotting)
        pil_image = Image.open(img_path).convert("RGB")
        pil_image = pil_image.resize(self.image_size, Image.BILINEAR)

        # Load mask; assume corresponding mask has same name but with .png extension
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.image_size, Image.NEAREST)


        # Process the image with the CLIPSeg processor
        inputs = self.processor(text=[self.prompt], images=[pil_image], padding="max_length", return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Convert mask to tensor; then threshold so that any nonzero becomes 1
        label = transforms.ToTensor()(mask).squeeze(0)
        label = (label > 0).float()  # Now binary: 0 or 1

        return {
            "original_pil": pil_image,       # For plotting (PIL image)
            "pixel_values": pixel_values,      # For model input (tensor)
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label                    # Binary mask (tensor of shape (H, W))
        }

# --- Evaluation and Plotting ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the fine-tuned model and processor
    save_dir = "./clipseg_finetuned_leaf"
    processor = CLIPSegProcessor.from_pretrained(save_dir)
    model = CLIPSegForImageSegmentation.from_pretrained(save_dir)
    model.to(device)
    model.eval()

    # Create test dataset and DataLoader with custom collate to handle PIL images
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
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]  # (B, H, W) on CPU

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits  # (B, H, W)
            probs = torch.sigmoid(logits)
            # Print probability range for debugging
            print(f"Prob range: min={probs.min().item():.4f}, max={probs.max().item():.4f}")
            # Lower threshold for visualization
            pred_masks = (probs > VIS_THRESHOLD).long()

            for i in range(pred_masks.size(0)):
                gt = labels[i].long()  # Binary mask (0 or 1)
                pred = pred_masks[i].cpu().squeeze(0)  # (H, W)
                iou = compute_iou(pred, gt)
                total_iou += iou
                num_samples += 1

                plot_samples.append({
                    "original_pil": batch["original_pil"][i],
                    "gt_mask": gt,
                    "pred_mask": pred,
                    "prob_map": probs[i].cpu()
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
