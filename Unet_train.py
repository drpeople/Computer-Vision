import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# 1) CONFIG + PATHS
# ---------------------------
train_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\images"
train_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\masks"
val_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
val_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# ---------------------------
# 2) DATASET DEFINITION
# ---------------------------
class LeafDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.image_names = [f for f in sorted(os.listdir(images_dir)) if f.lower().endswith(valid_exts)]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, os.path.splitext(img_name)[0] + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


# ---------------------------
# 3) TRANSFORMATIONS
# ---------------------------
class LeafSegmentationTransform:
    def __init__(self, resize=(544, 544), pad=(0, 0, 0, 0), sigma=0, is_train=False):
        self.resize = resize
        self.pad = pad
        self.sigma = sigma
        self.is_train = is_train

        image_transforms = [T.Resize(self.resize, interpolation=T.InterpolationMode.BILINEAR)]
        if self.is_train:
            image_transforms.append(T.RandomHorizontalFlip())
        if self.sigma > 0:
            image_transforms.append(T.GaussianBlur(kernel_size=(5, 5), sigma=(self.sigma, self.sigma)))
        image_transforms.append(T.Pad(self.pad))
        image_transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_transform = T.Compose(image_transforms)

        mask_transforms = [T.Resize(self.resize, interpolation=T.InterpolationMode.NEAREST), T.Pad(self.pad, fill=0)]
        self.mask_transform = T.Compose(mask_transforms)

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np > 0).astype(np.float32)  # Convert to binary mask
        mask = torch.from_numpy(mask_np).unsqueeze(0)  # Add channel dimension
        return image, mask


train_transform = LeafSegmentationTransform(resize=(544, 544), pad=(0, 0, 0, 0), sigma=0, is_train=True)
val_transform = LeafSegmentationTransform(resize=(544, 544), pad=(0, 0, 0, 0), sigma=0, is_train=False)


# ---------------------------
# 4) DATALOADERS
# ---------------------------
def get_dataloaders():
    train_dataset = LeafDataset(train_images_dir, train_masks_dir, transform=train_transform)
    val_dataset = LeafDataset(val_images_dir, val_masks_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
    return train_loader, val_loader


# ---------------------------
# 5) MODEL, LOSS, OPTIMIZER
# ---------------------------
def get_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1  # Binary segmentation (output single-channel probability map)
    ).to(device)
    return model


criterion = nn.BCEWithLogitsLoss()


# ---------------------------
# 6) TRAINING & EVALUATION FUNCTIONS
# ---------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device).float()
        optimizer.zero_grad()
        outputs = model(images)  # shape: (B, 1, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


# ---------------------------
# 7) TRAINING LOOP
# ---------------------------
def main():
    train_loader, val_loader = get_dataloaders()
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 40
    best_val_loss = float('inf')
    save_path = "finetuned_unet_leaf.pth"

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved model at epoch {epoch + 1} with val loss {val_loss:.4f}")

    print("Training complete. Best validation loss:", best_val_loss)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()

