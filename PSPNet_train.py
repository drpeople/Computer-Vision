import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

import segmentation_models_pytorch as smp  # Using PSPNet from smp

# Fixed image size – PSPNet typically expects images to be resized (width, height)
IMAGE_SIZE = (512, 512)

# Lower threshold to accommodate mask values up to ~38/255 ~= 0.15
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

        # Image transforms: Resize, convert to tensor, and normalize using ImageNet stats.
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                                 std=[0.229, 0.224, 0.225])   # ImageNet stds
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
            "labels": mask          # (H, W) with values 0 or 1
        }

# --- Training and Validation Functions ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device).long()  # CrossEntropyLoss expects LongTensor for targets

        optimizer.zero_grad()
        with autocast():
            outputs = model(pixel_values)  # outputs shape: (B, num_classes, H, W)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device).long()
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
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
    num_epochs = 30
    learning_rate = 1e-4

    # Create training and validation datasets and dataloaders
    train_dataset = LeafSegFineTuneDataset(train_images_dir, train_masks_dir, image_size=IMAGE_SIZE)
    val_dataset   = LeafSegFineTuneDataset(val_images_dir, val_masks_dir, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # Create PSPNet model using smp.
    # Here, we use the 'resnet50' encoder with ImageNet pretraining.
    # Since we're doing binary segmentation (background and leaf), set classes=2.
    model = smp.PSPNet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        classes=2,
        activation=None  # No activation since we'll compute loss on logits
    )
    model.to(device)

    # Define optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss   = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save the fine-tuned model
    save_dir = "./pspnet_finetuned_leaf"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    print("Fine-tuning complete. Model saved to:", save_dir)

    #
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.CrossEntropyLoss()
    # scaler = GradScaler()
    #
    # save_dir = "./pspnet_finetuned_leaf"
    # best_val_loss = float('inf')
    # best_model_path = os.path.join(save_dir, "best_model.pth")
    # print("Starting fine-tuning...")
    # for epoch in range(num_epochs):
    #     train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
    #     val_loss   = validate_one_epoch(model, val_loader, criterion, device)
    #     print(f"Epoch {epoch + 1}/{num_epochs} - "
    #           f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save(model.state_dict(), best_model_path)
    #         print(f"Saved best model at epoch {epoch + 1} with val loss: {val_loss:.4f}")
    #
    # # Save the fine-tuned model
    # save_dir = "./pspnet_finetuned_leaf"
    # os.makedirs(save_dir, exist_ok=True)
    # torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    # print("Fine-tuning complete. Model saved to:", save_dir)

if __name__ == "__main__":
    main()