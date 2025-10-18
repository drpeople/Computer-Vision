import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# Fixed image size – SqueezeNet will expect the image to be at a fixed resolution.
IMAGE_SIZE = (512, 512)
GT_THRESHOLD = 0.05

# --- Custom Dataset for Fine-Tuning ---
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

        # Binarize mask at GT_THRESHOLD and convert to long (for CrossEntropyLoss)
        mask = (mask > GT_THRESHOLD).float().squeeze(0)
        return {
            "pixel_values": image,  # (3, H, W)
            "labels": mask.long()   # (H, W) with values 0 or 1
        }

# --- Define SqueezeNet-based Segmentation Model ---
class SqueezeNetSegmentation(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Load pretrained SqueezeNet
        self.squeezenet = models.squeezenet1_1(pretrained=True)
        # Use the feature extractor part (the convolutional layers)
        self.features = self.squeezenet.features  # e.g., outputs shape (B, 512, H_feat, W_feat)
        # Segmentation head: 1x1 convolution to predict num_classes channels
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)  # shape: (B, num_classes, H_feat, W_feat)
        # Dynamically upsample to match input resolution
        up_logits = torch.nn.functional.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return up_logits

# --- Training and Validation Functions ---
def train_one_epoch(model, dataloader, optimizer, device, scaler, loss_fn):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)  # shape: (B, H, W)

        optimizer.zero_grad()
        with autocast():
            outputs = model(pixel_values)  # shape: (B, 2, H, W)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values)
            loss = loss_fn(outputs, labels)
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
    num_epochs = 40
    learning_rate = 1e-4

    # Create training and validation datasets and dataloaders
    train_dataset = LeafSegFineTuneDataset(train_images_dir, train_masks_dir, image_size=IMAGE_SIZE)
    val_dataset   = LeafSegFineTuneDataset(val_images_dir, val_masks_dir, image_size=IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Initialize the SqueezeNet segmentation model
    model = SqueezeNetSegmentation(num_classes=2)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    loss_fn = nn.CrossEntropyLoss()

    # Directory to save models
    save_dir = "./squeezenet_segmentation_finetuned"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize best validation loss
    best_val_loss = float('inf')

    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, loss_fn)
        val_loss   = validate_one_epoch(model, val_loader, device, loss_fn)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save model if current validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best_val.pth"))
            print("Saved new best validation model!")

    print("Fine-tuning complete. Best model saved to:", save_dir)

if __name__ == "__main__":
    main()
