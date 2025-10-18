"""
Leaf-detection training script (Faster R-CNN ResNet50 FPN)

Assumes exactly the same folder layout you already have:
    C:\...\data\images   – RGB images
    C:\...\data\masks    – single-channel PNG masks (leaf = white/255, background = black/0)
A single bounding-box is derived from every mask by taking the tight
rectangle that encloses all foreground pixels (you can extend to multi-object
masks later).

⚠️  Differences vs. segmentation version
• Dataset now yields (image, target_dict) where target_dict has keys
  boxes / labels / area / iscrowd / image_id.
• No explicit criterion – Faster R-CNN returns its own loss dictionary.
• Need a custom collate_fn for the dataloader (variable-length targets).
• Inference helper draws predicted boxes on the image.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FastRCNNPredictor,
)
from torchvision.ops import masks_to_boxes
from torch.cuda.amp import autocast, GradScaler

# ---------------------------
# 1) CONFIG + PATHS
# ---------------------------
train_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\images"
train_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\masks"
val_images_dir   = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
val_masks_dir    = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"Using device: {device}")

# ---------------------------
# 2) DATASET & TRANSFORMS
# ---------------------------
class LeafDetectionDataset(Dataset):
    """
    Builds a single bounding-box per mask.
    Label '1'  →  leaf
    Background →  0 (handled internally by Faster R-CNN)
    """

    def __init__(self, images_dir, masks_dir, transforms=None):
        self.images = sorted(
            [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def _mask_to_targets(self, mask, img_id):
        """Convert single-channel mask → boxes/labels/area dict."""
        mask_np = np.array(mask)
        obj_ids = np.unique(mask_np)
        # background id = 0 ⇒ keep only foreground
        obj_ids = obj_ids[obj_ids != 0]
        masks = mask_np == obj_ids[:, None, None]  # shape [N,H,W]

        if masks.shape[0] == 0:  # safety – shouldn’t really happen
            masks = np.zeros((1, *mask_np.shape), dtype=np.uint8)
        masks_t = torch.as_tensor(masks, dtype=torch.uint8)

        boxes = masks_to_boxes(masks_t)  # [N,4] (x1,y1,x2,y2)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # all leaves → class 1
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        return {
            "boxes": boxes,
            "labels": labels,
            "masks": masks_t,  # optional but nice for future use
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([img_id]),
        }

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(
            self.masks_dir, os.path.splitext(img_name)[0] + ".png"
        )

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transforms:
            img = self.transforms(img)

        target = self._mask_to_targets(mask, idx)
        return img, target


def get_transforms(train=True):
    trans = []
    trans.append(T.ToTensor())  # converts [0,255] PIL → float [0,1] tensor + channels-first
    if train:
        # fliplr is applied to the image; boxes are auto-flipped by built-in util in torchvision >=0.15
        trans.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(trans)


# ---------------------------
# 3) DATALOADERS
# ---------------------------
def collate_fn(batch):
    return tuple(zip(*batch))  # required for detection models


def get_dataloaders(batch_size=4, num_workers=8):
    train_ds = LeafDetectionDataset(
        train_images_dir, train_masks_dir, transforms=get_transforms(train=True)
    )
    val_ds = LeafDetectionDataset(
        val_images_dir, val_masks_dir, transforms=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
        persistent_workers=True,
    )
    return train_loader, val_loader


# ---------------------------
# 4) MODEL, OPTIMIZER
# ---------------------------
def get_model(num_classes=2):
    """
    num_classes includes background, so 2 = {background, leaf}
    """
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # replace box predictor head
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    return model.to(device)


# ---------------------------
# 5) TRAIN/EVAL w/AMP
# ---------------------------
def train_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast():
            loss_dict = model(imgs, targets)  # returns dict of losses
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # During eval we can still get loss values by passing targets
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


# ---------------------------
# 6) MAIN with checkpointing
# ---------------------------
def main():
    train_loader, val_loader = get_dataloaders()
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()
    best_loss = float("inf")
    best_path = "best_fasterrcnn_resnet50_leaf.pth"

    # Optionally freeze backbone for a few warm-up epochs
    freeze_backbone_epochs = 5
    for param in model.backbone.parameters():
        param.requires_grad = False

    try:
        for epoch in range(1, 31):
            if epoch == freeze_backbone_epochs + 1:  # unfreeze
                for param in model.backbone.parameters():
                    param.requires_grad = True
                print(f"Backbone unfrozen at epoch {epoch}")

            train_loss = train_epoch(model, train_loader, optimizer, scaler)
            val_loss = eval_epoch(model, val_loader)
            print(
                f"Epoch {epoch:2d}/30 — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Checkpoint each epoch
            torch.save(model.state_dict(), f"checkpoint_frcnn_epoch{epoch}.pth")

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"✓ Best model updated (Val Loss: {val_loss:.4f}) at epoch {epoch}")
    except KeyboardInterrupt:
        print("Interrupted — saving current weights...")
        torch.save(model.state_dict(), "interrupted_frcnn_leaf.pth")
    finally:
        print(f"Training complete — Best Val Loss: {best_loss:.4f}")


# ---------------------------
# 7) INFERENCE helper
# ---------------------------
@torch.no_grad()
def infer_and_plot(model, loader, score_thr=0.5, num_samples=5):
    model.eval()
    cmap = plt.get_cmap("tab20")
    count = 0
    for imgs, _ in loader:
        if count >= num_samples:
            break
        img = imgs[0].to(device)
        preds = model([img])[0]

        # Move to CPU for vis
        img_np = img.cpu().permute(1, 2, 0).numpy()
        boxes = preds["boxes"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(img_np)
        for b, s in zip(boxes, scores):
            if s < score_thr:
                continue
            x1, y1, x2, y2 = b
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=cmap(0),
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 3,
                f"leaf {s:.2f}",
                fontsize=10,
                bbox=dict(facecolor="yellow", alpha=0.4, pad=0.2),
            )
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        count += 1


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
