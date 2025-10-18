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


# ---------------------------
# 2) DATASET DEFINITION
# ---------------------------
class LeafDataset(Dataset):
    """
    Dataset for leaf segmentation.
    Assumes that each image in the images folder has a corresponding mask with the same basename
    (mask extension is .png) in the masks folder.
    Filters out non-image files.
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.image_names = [
            f for f in sorted(os.listdir(images_dir))
            if f.lower().endswith(valid_exts)
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.masks_dir, base_name + ".png")

        image = Image.open(img_path).convert("RGB")
        # Convert mask to grayscale to ensure a single channel output
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask


# ---------------------------
# 3) TRANSFORMATION CLASS
# ---------------------------
class LeafSegmentationTransform:
    """
    Transformation class to resize and pad images and masks.
    This class is picklable, so it works with DataLoader's multiprocessing.
    """

    def __init__(self, resize=(520, 520), pad=(4, 4, 4, 4), sigma=0, is_train=False):
        self.resize = resize
        self.pad = pad
        self.sigma = sigma
        self.is_train = is_train

        # Build transformation pipeline for the image.
        image_transforms = [T.Resize(self.resize, interpolation=T.InterpolationMode.BILINEAR)]
        if self.is_train:
            image_transforms.append(T.RandomHorizontalFlip())
        if self.sigma > 0:
            image_transforms.append(T.GaussianBlur(kernel_size=(5, 5), sigma=(self.sigma, self.sigma)))
        image_transforms.append(T.Pad(self.pad))
        image_transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.image_transform = T.Compose(image_transforms)

        # Build transformation pipeline for the mask.
        mask_transforms = [
            T.Resize(self.resize, interpolation=T.InterpolationMode.NEAREST),
            T.Pad(self.pad, fill=0)  # We'll remap nonzero values to 1 below.
        ]
        self.mask_transform = T.Compose(mask_transforms)

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        # Convert mask to a NumPy array and map all nonzero pixels to 1 (binary segmentation)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np > 0).astype(np.int64)
        mask = torch.from_numpy(mask_np)
        return image, mask


# Create transforms for training and validation.
train_transform = LeafSegmentationTransform(resize=(520, 520), pad=(4, 4, 4, 4), sigma=0, is_train=True)
val_transform = LeafSegmentationTransform(resize=(520, 520), pad=(4, 4, 4, 4), sigma=0, is_train=False)


# ---------------------------
# 4) DATALOADERS
# ---------------------------
def get_dataloaders():
    train_dataset = LeafDataset(train_images_dir, train_masks_dir, transform=train_transform)
    val_dataset = LeafDataset(val_images_dir, val_masks_dir, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    return train_loader, val_loader


# ---------------------------
# 5) MODEL, LOSS, OPTIMIZER
# ---------------------------
def get_model():
    # Assuming binary segmentation (2 classes). Adjust if necessary.
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2
    ).to(device)
    return model


criterion = nn.CrossEntropyLoss()


# ---------------------------
# 6) TRAINING & EVALUATION FUNCTIONS
# ---------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # shape: (B, num_classes, H, W)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


# ---------------------------
# 7) TRAINING LOOP
# ---------------------------
def main():
    train_loader, val_loader = get_dataloaders()
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 30
    best_val_loss = float('inf')
    save_path = "finetuned_deeplabv3plus_leaf.pth"

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save model if validation loss improved.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved model at epoch {epoch + 1} with val loss {val_loss:.4f}")

    print("Training complete. Best validation loss:", best_val_loss)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()

import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

# ---------------------------
# 1) CONFIG + PATHS
# ---------------------------
val_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
val_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
save_path = "finetuned_deeplabv3plus_leaf.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# 2) DATASET DEFINITION
# ---------------------------
class LeafDataset(torch.utils.data.Dataset):
    """
    Dataset for leaf segmentation.
    Assumes each image in the images folder has a corresponding mask with the same basename
    (mask extension is .png) in the masks folder.
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        valid_exts = ('.jpg', '.jpeg', '.png')
        self.image_names = [
            f for f in sorted(os.listdir(images_dir))
            if f.lower().endswith(valid_exts)
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.masks_dir, base_name + ".png")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, img_name  # return img_name for reference if needed

# ---------------------------
# 3) TRANSFORMATIONS
# ---------------------------
class LeafSegmentationTransform:
    """
    Transformation class to resize and pad images and masks.
    """
    def __init__(self, resize=(520, 520), pad=(4, 4, 4, 4), sigma=0, is_train=False):
        self.resize = resize
        self.pad = pad
        self.sigma = sigma
        self.is_train = is_train

        # Build transformation pipeline for the image.
        image_transforms = [T.Resize(self.resize, interpolation=T.InterpolationMode.BILINEAR)]
        # No random flip in validation
        if self.sigma > 0:
            image_transforms.append(T.GaussianBlur(kernel_size=(5, 5), sigma=(self.sigma, self.sigma)))
        image_transforms.append(T.Pad(self.pad))
        image_transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.image_transform = T.Compose(image_transforms)

        # Build transformation pipeline for the mask.
        mask_transforms = [
            T.Resize(self.resize, interpolation=T.InterpolationMode.NEAREST),
            T.Pad(self.pad, fill=0)
        ]
        self.mask_transform = T.Compose(mask_transforms)

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        # Convert mask to a NumPy array and map all nonzero pixels to 1 (binary segmentation)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_np = (mask_np > 0).astype(np.int64)
        mask = torch.from_numpy(mask_np)
        return image, mask

# Create validation transform and dataset.
val_transform = LeafSegmentationTransform(resize=(520, 520), pad=(4, 4, 4, 4), sigma=0, is_train=False)
val_dataset = LeafDataset(val_images_dir, val_masks_dir, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

# ---------------------------
# 4) MODEL DEFINITION
# ---------------------------
def get_model():
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2  # binary segmentation: background and leaf
    ).to(device)
    return model

# Instantiate the model and load saved weights.
model = get_model()
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
print("Loaded trained model.")

# ---------------------------
# 5) INFERENCE & PLOTTING
# ---------------------------
def infer_and_plot(model, loader, num_samples=5):
    plt.figure(figsize=(15, num_samples * 5))
    sample_count = 0

    for image, mask, img_name in loader:
        if sample_count >= num_samples:
            break

        image = image.to(device)
        # Forward pass.
        with torch.no_grad():
            outputs = model(image)
        # Get predicted segmentation mask (choose the class with highest score)
        # outputs shape: (1, 2, H, W)
        pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        # Get ground truth mask.
        gt_mask = mask.squeeze(0).cpu().numpy()
        # Get original image for visualization: unnormalize and convert to numpy.
        # Reverse normalization.
        image_cpu = image.squeeze(0).cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image_cpu = image_cpu * std + mean
        image_np = image_cpu.permute(1,2,0).numpy()
        image_np = np.clip(image_np, 0, 1)

        # Plot the original image, ground truth, and predicted mask.
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_np)
        axes[0].set_title(f"Image: {img_name[0]}")
        axes[0].axis('off')

        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')

        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        sample_count += 1

if __name__ == '__main__':
    infer_and_plot(model, val_loader, num_samples=5)

