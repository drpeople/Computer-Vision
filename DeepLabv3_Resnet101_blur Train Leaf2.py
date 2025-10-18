# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import numpy as np
# from PIL import Image
# from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
# import torch.nn as nn
# import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler
# import matplotlib.pyplot as plt
#
# # ---------------------------
# # 1) CONFIG + PATHS
# # ---------------------------
# train_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\images"
# train_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\masks"
# val_images_dir   = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# val_masks_dir    = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True
# print(f"Using device: {device}")
#
# # ---------------------------
# # 2) DATASET & TRANSFORMS
# # ---------------------------
# class LeafDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, transform=None):
#         self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         img = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
#         mask = Image.open(os.path.join(self.masks_dir, os.path.splitext(img_name)[0] + ".png")).convert("L")
#         if self.transform:
#             img, mask = self.transform(img, mask)
#         return img, mask
#
# class LeafSegmentationTransform:
#     def __init__(self, resize=(520,520), pad=(4,4,4,4), sigma=0, is_train=False):
#         img_t = [T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR)]
#         if is_train:
#             img_t.append(T.RandomHorizontalFlip())
#         if sigma > 0:
#             img_t.append(T.GaussianBlur((5,5), (sigma, sigma)))
#         img_t += [T.Pad(pad), T.ToTensor(),
#                   T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
#         self.img_t = T.Compose(img_t)
#         self.mask_t = T.Compose([
#             T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
#             T.Pad(pad, fill=0)
#         ])
#
#     def __call__(self, img, mask):
#         img = self.img_t(img)
#         mask = self.mask_t(mask)
#         mask_np = (np.array(mask) > 0).astype(np.int64)
#         return img, torch.from_numpy(mask_np)
#
# train_transform = LeafSegmentationTransform(is_train=True)
# val_transform   = LeafSegmentationTransform(is_train=False)
#
# # ---------------------------
# # 3) DATALOADERS
# # ---------------------------
# def get_dataloaders(batch_size=8, num_workers=8):
#     train_ds = LeafDataset(train_images_dir, train_masks_dir, transform=train_transform)
#     val_ds   = LeafDataset(val_images_dir,   val_masks_dir,   transform=val_transform)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
#                               num_workers=num_workers, pin_memory=True,
#                               prefetch_factor=2, persistent_workers=True)
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
#                               num_workers=num_workers, pin_memory=True,
#                               prefetch_factor=2, persistent_workers=True)
#     return train_loader, val_loader
#
# # ---------------------------
# # 4) MODEL, LOSS, OPTIMIZER
# # ---------------------------
# def get_model(num_classes=2):
#     # Load DeepLabV3-ResNet101 with pretrained weights
#     weights = DeepLabV3_ResNet101_Weights.DEFAULT
#     model = deeplabv3_resnet101(weights=weights).to(device)
#     # Replace classifier head for binary segmentation
#     in_ch = model.classifier[-1].in_channels
#     model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=False).to(device)
#     return model
#
# criterion = nn.CrossEntropyLoss()
#
# # ---------------------------
# # 5) TRAIN/EVAL w/AMP
# # ---------------------------
# def train_epoch(model, loader, optimizer, scaler):
#     model.train()
#     total_loss = 0.0
#     for imgs, masks in loader:
#         imgs = imgs.to(device, non_blocking=True)
#         masks = masks.to(device, non_blocking=True)
#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(imgs)['out']
#             loss = criterion(outputs, masks)
#         scaler.scale(loss).backward()
#         torch.cuda.synchronize()
#         scaler.step(optimizer)
#         scaler.update()
#         total_loss += loss.item() * imgs.size(0)
#     torch.cuda.synchronize()
#     return total_loss / len(loader.dataset)
#
# def eval_epoch(model, loader):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for imgs, masks in loader:
#             imgs = imgs.to(device)
#             masks = masks.to(device)
#             outputs = model(imgs)['out']
#             total_loss += criterion(outputs, masks).item() * imgs.size(0)
#     return total_loss / len(loader.dataset)
#
# # ---------------------------
# # 6) MAIN with checkpointing
# # ---------------------------
# def main():
#     train_loader, val_loader = get_dataloaders()
#     model = get_model()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scaler = GradScaler()
#     best_loss = float('inf')
#     best_path = "finetune_deeplabv3_resnet101_leaf.pth"
#
#     # Freeze backbone for first few epochs
#     for param in model.backbone.parameters():
#         param.requires_grad = False
#     unfreeze_epoch = 6
#
#     try:
#         for epoch in range(1, 31):
#             if epoch == unfreeze_epoch:
#                 for param in model.backbone.parameters():
#                     param.requires_grad = True
#                 print("Backbone unfrozen at epoch", epoch)
#
#             train_loss = train_epoch(model, train_loader, optimizer, scaler)
#             val_loss = eval_epoch(model, val_loader)
#             print(f"Epoch {epoch}/30 — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#
#             # Checkpoint each epoch
#             ckpt_name = f"checkpoint_epoch{epoch}.pth"
#             torch.save(model.state_dict(), ckpt_name)
#
#             # Save best model
#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 torch.save(model.state_dict(), best_path)
#                 print(f"Best model updated (Val Loss: {val_loss:.4f}) at epoch {epoch}")
#     except KeyboardInterrupt:
#         print("Training interrupted — saving current model...")
#         torch.save(model.state_dict(), "interrupted_deeplabv3v3_leaf.pth")
#     finally:
#         print(f"Done — Best Val Loss: {best_loss:.4f}")
#
# # ---------------------------
# # 7) INFERENCE helper
# # ---------------------------
# def infer_and_plot(model, loader, num_samples=5):
#     mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1).to(device)
#     std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1).to(device)
#     count = 0
#     for imgs, masks in loader:
#         if count >= num_samples: break
#         imgs = imgs.to(device)
#         with torch.no_grad(): out = model(imgs)['out']
#         preds = torch.argmax(out,1).cpu().numpy()
#         gts = masks.numpy()
#
#         img_vis = (imgs.squeeze()*std + mean).permute(1,2,0).cpu().numpy().clip(0,1)
#         fig, axs = plt.subplots(1,3,figsize=(15,5))
#         axs[0].imshow(img_vis); axs[0].axis('off'); axs[0].set_title("Image")
#         axs[1].imshow(gts[0], cmap='gray'); axs[1].axis('off'); axs[1].set_title("GT")
#         axs[2].imshow(preds[0], cmap='gray'); axs[2].axis('off'); axs[2].set_title("Pred")
#         plt.tight_layout(); plt.show()
#         count += 1
#
# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()


### blur
# import os
# import sys
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# from torch.cuda.amp import autocast
# from sklearn.metrics import confusion_matrix
# from torchvision.transforms import functional as TF
# from torchvision.models.segmentation import deeplabv3_resnet101
# import torch.nn as nn
#
# # ---------------------------
# # 1) CONFIG + SETUP
# # ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# torch.backends.cudnn.benchmark = True  # Performance optimization
#
# model_path = "./finetune_deeplabv3_resnet101_leaf.pth"
#
# # ---------------------------
# # 2) MODEL LOADING
# # ---------------------------
# model = deeplabv3_resnet101(weights=None, progress=True)
# in_ch = model.classifier[-1].in_channels
# model.classifier[-1] = nn.Conv2d(in_ch, 2, kernel_size=1, bias=False)
# if getattr(model, 'aux_classifier', None) is not None:
#     aux_in = model.aux_classifier[-1].in_channels
#     model.aux_classifier[-1] = nn.Conv2d(aux_in, 2, kernel_size=1, bias=False)
#
# try:
#     state_dict = torch.load(model_path, map_location=device)
#     load_info  = model.load_state_dict(state_dict, strict=False)
#     print("Loaded weights (with ignored keys):", load_info)
# except Exception as e:
#     print(f"Error loading state dict: {e}")
#     sys.exit(1)
#
# model = model.to(device).eval().float()
#
# # ---------------------------
# # 3) TEST DATASET & TRANSFORMS
# # ---------------------------
# test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
# class LeafSegInferenceDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, image_size=(520, 520), sigma=0):
#         self.images_dir = images_dir
#         self.masks_dir  = masks_dir
#         self.files      = sorted([
#             f for f in os.listdir(images_dir)
#             if f.lower().endswith(('.jpg','.jpeg','.png'))
#         ])
#         self.sigma      = sigma
#         self.img_tf = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#         ])
#         self.mask_tf = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)
#         ])
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         fn  = self.files[idx]
#         img = Image.open(os.path.join(self.images_dir, fn)).convert("RGB")
#         if self.sigma > 0:
#             img = TF.gaussian_blur(img, kernel_size=5, sigma=self.sigma)
#
#         mask_name = os.path.splitext(fn)[0] + ".png"
#         mask = Image.open(os.path.join(self.masks_dir, mask_name)).convert("L")
#
#         img_t  = self.img_tf(img)
#         mask_t = torch.from_numpy(
#             (np.array(self.mask_tf(mask)) > 0).astype(np.int64)
#         )
#
#         return {"image": img_t, "mask": mask_t}
#
# # ---------------------------
# # 4) INFERENCE
# # ---------------------------
# @torch.no_grad()
# def run_inference(dataset, batch_size=4):
#     loader = DataLoader(dataset, batch_size=batch_size,
#                         shuffle=False, num_workers=4, pin_memory=True)
#     preds, gts = [], []
#     for batch in loader:
#         imgs = batch["image"].to(device)
#         with autocast():
#             out = model(imgs)["out"]
#         preds.extend(torch.argmax(out, dim=1).cpu().numpy())
#         gts.extend(batch["mask"].cpu().numpy())
#     return preds, gts
#
# # ---------------------------
# # 5) METRICS (GLOBAL CM)
# # ---------------------------
# def compute_metrics(preds, gts):
#     # build one global confusion matrix
#     cm = np.zeros((2,2), dtype=np.int64)
#     for p, g in zip(preds, gts):
#         cm += confusion_matrix(g.flatten(), p.flatten(), labels=[0,1])
#     pixel_acc = cm.diagonal().sum() / (cm.sum() + 1e-10)
#     mean_iou  = cm[1,1] / (cm[1,1] + cm[0,1] + cm[1,0] + 1e-10)
#     return pixel_acc, mean_iou
#
# # ---------------------------
# # 6) SWEEP & PLOT
# # ---------------------------
# def plot_blur_sweep(sigmas, accs, mious):
#     plt.figure(figsize=(8,6))
#     plt.plot(sigmas, accs, marker='o', label='Pixel Accuracy')
#     plt.plot(sigmas, mious, marker='s', label='Mean IoU')
#     plt.title("Segmentation vs Gaussian Blur")
#     plt.xlabel("Blur Sigma")
#     plt.ylabel("Score")
#     plt.ylim(0,1)
#     plt.grid(True)
#     plt.legend()
#     plt.savefig("blur_vs_metrics.png")
#     plt.show()
#
# # ---------------------------
# # 7) MAIN
# # ---------------------------
# def main():
#     sigmas = [0,1,2,3,4]
#     acc_list, miou_list = [], []
#     for s in sigmas:
#         print(f"\n--> Evaluating sigma={s}")
#         ds = LeafSegInferenceDataset(test_images_dir, test_masks_dir, sigma=s)
#         preds, gts = run_inference(ds)
#         acc, miou = compute_metrics(preds, gts)
#         acc_list.append(acc)
#         miou_list.append(miou)
#         print(f"Sigma={s}: PixelAcc={acc:.4f}, MeanIoU={miou:.4f}")
#
#     plot_blur_sweep(sigmas, acc_list, miou_list)
#
#     df = pd.DataFrame({
#         "sigma": sigmas,
#         "pixel_accuracy": acc_list,
#         "mean_iou": miou_list
#     })
#     df.to_csv("deeplabv3_resnet101_blur_results.csv", index=False)
#     print("Saved CSV: deeplabv3_resnet101_blur_results.csv")
#
# if __name__ == "__main__":
#     main()


### NOISE
"""
Evaluate DeepLabV3-ResNet101 (fine-tuned for leaves) on the leaf test-set
while sweeping Gaussian-noise levels.

Author: you 😊
Date  : 12 May 2025
"""
# ---------------------------#
# 0) IMPORTS
# ---------------------------#
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

# ---------------------------#
# 1) CONFIG + SETUP
# ---------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True

LEAF_ROOT       = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf"
TEST_IMAGES_DIR = os.path.join(LEAF_ROOT, "test", "images")
TEST_MASKS_DIR  = os.path.join(LEAF_ROOT, "test", "masks")
WEIGHTS_PATH    = r".\finetune_deeplabv3_resnet101_leaf.pth"
CSV_NAME        = "deeplabv3_resnet101_leaf_noise_results.csv"

NUM_CLASSES = 2  # background, leaf

# ---------------------------#
# 2) MODEL: LOAD & PATCH HEAD
# ---------------------------#
model = deeplabv3_resnet101(weights=None, progress=True)
# Patch classifier & aux head to NUM_CLASSES channels
in_ch = model.classifier[-1].in_channels
model.classifier[-1] = nn.Conv2d(in_ch, NUM_CLASSES, kernel_size=1, bias=False)
if getattr(model, "aux_classifier", None) is not None:
    aux_in = model.aux_classifier[-1].in_channels
    model.aux_classifier[-1] = nn.Conv2d(aux_in, NUM_CLASSES, kernel_size=1, bias=False)

try:
    load = model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
    print("Weights loaded. Ignored keys:", load.unexpected_keys)
except Exception as e:
    print("❌ Could not load weights:", e)
    sys.exit(1)

model = model.to(device).eval()

# ---------------------------#
# 3) DATASET
# ---------------------------#
class LeafSegDataset(Dataset):
    """Loads (image, mask, filename) tuples from the leaf test split."""
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.files     = sorted(
            [f for f in os.listdir(img_dir)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn   = self.files[idx]
        img  = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        mask = Image.open(
            os.path.join(self.mask_dir, os.path.splitext(fn)[0] + ".png")
        ).convert("L")

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask, fn

# ---------------------------#
# 4) CUSTOM GAUSSIAN-NOISE TF
# ---------------------------#
class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.0):
        super().__init__()
        self.mean, self.std = mean, std

    def forward(self, x):
        if self.std <= 0:
            return x
        return x + torch.randn_like(x) * self.std + self.mean

# ---------------------------#
# 5) TRANSFORM PIPELINE
# ---------------------------#
class SegmentationTransform:
    """Resize → tensor → noise → normalize   (mask: resize→(>0)→long)."""
    def __init__(self, resize=(520, 520), noise_std=0.0):
        self.img_tf = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            AddGaussianNoise(std=noise_std),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        self.mask_tf = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.NEAREST)
        ])

    def __call__(self, img, mask):
        img  = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.int64))
        return img, mask

def build_transform(resize=(520, 520), noise_std=0.0):
    return SegmentationTransform(resize, noise_std)

# ---------------------------#
# 6) METRICS (PixelAcc & IoU)
# ---------------------------#
def compute_metrics(preds, gts):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for p, g in zip(preds, gts):
        cm += confusion_matrix(g.flatten(), p.flatten(), labels=list(range(NUM_CLASSES)))
    pixel_acc = cm.diagonal().sum() / (cm.sum() + 1e-10)
    iou       = cm[1,1] / (cm[1,1] + cm[1,0] + cm[0,1] + 1e-10)
    return pixel_acc, iou

# ---------------------------#
# 7) INFERENCE (mixed precision)
# ---------------------------#
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    preds, gts = [], []
    for imgs, masks, _ in loader:   # <-- now unpacks 3
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            out = model(imgs)['out']
        pred = torch.argmax(out, dim=1).cpu().numpy()
        preds.extend(pred)
        gts.extend(masks.numpy())
    return preds, gts

# ---------------------------#
# 8) PLOT UTILS
# ---------------------------#
def plot_noise_sweep(noise_levels, accs, ious):
    plt.figure(figsize=(8,6))
    plt.plot(noise_levels, accs, marker='o', label='Pixel Accuracy')
    plt.plot(noise_levels, ious, marker='s', label='Leaf IoU')
    plt.title("Leaf segmentation vs. Gaussian noise (std)")
    plt.xlabel("Gaussian noise σ")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("leaf_noise_vs_metrics.png")
    plt.show()

# ---------------------------#
# 9) MAIN
# ---------------------------#
def main():
    noise_stds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    acc_list, iou_list = [], []

    for std in noise_stds:
        print(f"\n==> Evaluating with noise std = {std}")
        ds = LeafSegDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR,
                            transform=build_transform(noise_std=std))
        preds, gts = run_inference(ds, batch_size=4)
        acc, iou = compute_metrics(preds, gts)
        acc_list.append(acc)
        iou_list.append(iou)
        print(f"σ={std:.2f} · PixelAcc={acc:.4f}, MeanIoU={iou:.4f}")

    # A) Plot
    plot_noise_sweep(noise_stds, acc_list, iou_list)

    # B) Save to CSV
    pd.DataFrame({
        "noise_std": noise_stds,
        "pixel_accuracy": acc_list,
        "mean_iou": iou_list
    }).to_csv(CSV_NAME, index=False)
    print(f"\nCSV saved to: {CSV_NAME}")

if __name__ == "__main__":
    main()


### ROTATE
# """
# Rotate-sweep evaluation of a fine-tuned DeepLabV3-ResNet101 leaf model.
# Rotates each test image with expand=True, resizes to 520×520, marks the new
# black corners as ignore (=255), then measures PixelAcc & IoU.
#
# Author : you
# Date   : 12 May 2025
# """
# # ---------------------------#
# # 0) IMPORTS
# # ---------------------------#
# import os, sys
# import torch, torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# from torchvision.transforms import functional as TF
# from torchvision.models.segmentation import deeplabv3_resnet101
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# from sklearn.metrics import confusion_matrix
#
# # ---------------------------#
# # 1) CONFIG + PATHS
# # ---------------------------#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# torch.backends.cudnn.benchmark = True
#
# LEAF_ROOT       = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf"
# TEST_IMAGES_DIR = os.path.join(LEAF_ROOT, "test", "images")
# TEST_MASKS_DIR  = os.path.join(LEAF_ROOT, "test", "masks")
# WEIGHTS_PATH    = r".\finetune_deeplabv3_resnet101_leaf.pth"
#
# NUM_CLASSES = 2
# CSV_NAME    = "leaf_rotation_ignore_results.csv"
#
# # ---------------------------#
# # 2) MODEL
# # ---------------------------#
# model = deeplabv3_resnet101(weights=None, progress=True)
# # Patch heads to 2-channel
# in_ch = model.classifier[-1].in_channels
# model.classifier[-1] = nn.Conv2d(in_ch, NUM_CLASSES, 1, bias=False)
# if getattr(model, "aux_classifier", None) is not None:
#     aux_in = model.aux_classifier[-1].in_channels
#     model.aux_classifier[-1] = nn.Conv2d(aux_in, NUM_CLASSES, 1, bias=False)
#
# try:
#     load = model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device),
#                                  strict=False)
#     print("Weights loaded. Ignored keys:", load.unexpected_keys)
# except Exception as e:
#     print("❌ Could not load weights:", e); sys.exit(1)
#
# model = model.to(device).eval()
#
# # ---------------------------#
# # 3) DATASET
# # ---------------------------#
# class LeafSegDataset(Dataset):
#     """Loads (image, mask, filename) tuples from the leaf test set."""
#     def __init__(self, img_dir, mask_dir, transform=None):
#         self.img_dir   = img_dir
#         self.mask_dir  = mask_dir
#         self.transform = transform
#         self.files     = sorted([
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith((".png",".jpg",".jpeg"))
#         ])
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         fn   = self.files[idx]
#         img  = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
#         mask = Image.open(
#             os.path.join(self.mask_dir, os.path.splitext(fn)[0] + ".png")
#         ).convert("L")
#
#         if self.transform:
#             img, mask = self.transform(img, mask)
#
#         return img, mask, fn
#
# # ---------------------------#
# # 4) ROTATE → EXPAND TRANSFORM
# # ---------------------------#
# class RotatePairExpand:
#     def __init__(self, angle=0):
#         self.angle = angle
#     def __call__(self, img, mask):
#         img_rot  = img.rotate(self.angle,  resample=Image.BILINEAR, expand=True)
#         mask_rot = mask.rotate(self.angle, resample=Image.NEAREST,  expand=True)
#         return img_rot, mask_rot
#
# class SegmentationTransform:
#     """Rotate+expand, resize, mark new pixels as ignore(255), tensorise+normalise."""
#     def __init__(self, angle=0, resize=(520,520)):
#         self.angle  = angle
#         self.resize = resize
#         self.rot    = RotatePairExpand(angle)
#         self.img_tf = T.Compose([
#             T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#         ])
#         self.mask_tf = T.Resize(resize, interpolation=T.InterpolationMode.NEAREST)
#
#     def __call__(self, img, mask):
#         # 1) rotate both
#         img_r, mask_r = self.rot(img, mask)
#         # 2) rotate a valid‐area map to track original pixels
#         valid_map = Image.new("L", img.size, 1)
#         valid_r   = valid_map.rotate(self.angle, resample=Image.NEAREST, expand=True)
#         # 3) resize everything
#         img_t    = self.img_tf(img_r)
#         mask_rs  = self.mask_tf(mask_r)
#         valid_rs = valid_r.resize(self.resize, Image.NEAREST)
#
#         # 4) build mask & mark new corners as ignore=255
#         mask_np      = (np.array(mask_rs) > 0).astype(np.uint8)
#         mask_np[ valid_rs == 0 ] = 255
#         mask_t       = torch.from_numpy(mask_np.astype(np.int64))
#         return img_t, mask_t
#
# def build_transform(angle):
#     return SegmentationTransform(angle=angle, resize=(520,520))
#
# # ---------------------------#
# # 5) METRICS (GLOBAL CM)
# # ---------------------------#
# def compute_metrics(preds, gts):
#     cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
#     for p, g in zip(preds, gts):
#         p_flat = p.flatten()
#         g_flat = g.flatten()
#         valid  = (g_flat != 255)
#         cm += confusion_matrix(g_flat[valid], p_flat[valid], labels=list(range(NUM_CLASSES)))
#     pixel_acc = cm.diagonal().sum() / (cm.sum() + 1e-10)
#     mean_iou   = cm[1,1] / (cm[1,1] + cm[0,1] + cm[1,0] + 1e-10)
#     return pixel_acc, mean_iou
#
# # ---------------------------#
# # 6) INFERENCE
# # ---------------------------#
# @torch.no_grad()
# def run_inference(ds, batch_size=4):
#     loader = DataLoader(ds, batch_size=batch_size,
#                         shuffle=False, num_workers=4, pin_memory=True)
#     preds, gts = [], []
#     for imgs, masks, _ in loader:
#         imgs = imgs.to(device, non_blocking=True)
#         with torch.cuda.amp.autocast():
#             out = model(imgs)['out']
#         pred = torch.argmax(out, dim=1).cpu().numpy()
#         preds.extend(pred)
#         gts.extend(masks.numpy())
#     return preds, gts
#
# # ---------------------------#
# # 7) PLOT
# # ---------------------------#
# def plot_rotation_sweep(angles, accs, ious):
#     plt.figure(figsize=(8,6))
#     plt.plot(angles, accs, marker='o', label='Pixel Accuracy')
#     plt.plot(angles, ious, marker='s', label='Mean IoU')
#     plt.title("Leaf segmentation vs. rotation angle\n(expand=True, corners=ignore)")
#     plt.xlabel("Rotation angle (°)")
#     plt.ylabel("Score")
#     plt.ylim(0,1)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("leaf_rotation_vs_metrics.png")
#     plt.show()
#
# # ---------------------------#
# # 8) MAIN
# # ---------------------------#
# def main():
#     angles = list(range(0, 361, 30))
#     accs, ious = [], []
#
#     for a in angles:
#         print(f"\n==> Angle {a}°")
#         ds    = LeafSegDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR,
#                                transform=build_transform(a))
#         preds, gts = run_inference(ds, batch_size=4)
#         acc, iou   = compute_metrics(preds, gts)
#         accs.append(acc)
#         ious.append(iou)
#         print(f"PixelAcc={acc:.4f}  MeanIoU={iou:.4f}")
#
#     plot_rotation_sweep(angles, accs, ious)
#
#     pd.DataFrame({
#         "rotation_angle": angles,
#         "pixel_accuracy": accs,
#         "mean_iou":      ious
#     }).to_csv(CSV_NAME, index=False)
#     print(f"\nCSV saved to {CSV_NAME}")
#
# if __name__ == "__main__":
#     main()

### SCALE
# """
# DeepLabV3-ResNet101 leaf segmentation
# — performance vs. image scale factor —
#
# Author: you
# Date  : 12 May 2025
# """
# # ---------------------------#
# # 0) IMPORTS
# # ---------------------------#
# import os, sys
# import torch, torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# from torchvision.models.segmentation import deeplabv3_resnet101
# from PIL import Image
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
# # ---------------------------#
# # 1) CONFIG & PATHS
# # ---------------------------#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# torch.backends.cudnn.benchmark = True
#
# LEAF_ROOT       = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf"
# TEST_IMAGES_DIR = os.path.join(LEAF_ROOT, "test", "images")
# TEST_MASKS_DIR  = os.path.join(LEAF_ROOT, "test", "masks")
# WEIGHTS_PATH    = r".\finetune_deeplabv3_resnet101_leaf.pth"
#
# NUM_CLASSES = 2          # background / leaf
# CSV_NAME    = "leaf_scale_results.csv"
#
# # ---------------------------#
# # 2) MODEL (2-class head)
# # ---------------------------#
# model = deeplabv3_resnet101(weights=None, progress=True)
# in_ch = model.classifier[-1].in_channels
# model.classifier[-1] = nn.Conv2d(in_ch, NUM_CLASSES, 1, bias=False)
# if getattr(model, "aux_classifier", None) is not None:
#     aux_in = model.aux_classifier[-1].in_channels
#     model.aux_classifier[-1] = nn.Conv2d(aux_in, NUM_CLASSES, 1, bias=False)
#
# try:
#     load = model.load_state_dict(
#         torch.load(WEIGHTS_PATH, map_location=device), strict=False
#     )
#     print("Weights loaded. Ignored keys:", load.unexpected_keys)
# except Exception as e:
#     print("❌ Could not load weights:", e)
#     sys.exit(1)
#
# model = model.to(device).eval()
#
# # ---------------------------#
# # 3) DATASET
# # ---------------------------#
# class LeafSegDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, transform=None):
#         self.img_dir, self.mask_dir = img_dir, mask_dir
#         self.transform = transform
#         self.files = sorted([
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith((".jpg", ".jpeg", ".png"))
#         ])
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         fn = self.files[idx]
#         img  = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
#         mask = Image.open(
#             os.path.join(self.mask_dir, os.path.splitext(fn)[0] + ".png")
#         ).convert("L")
#         if self.transform:
#             img, mask = self.transform(img, mask)
#         return img, mask, fn
#
# # ---------------------------#
# # 4) SCALE TRANSFORMS
# # ---------------------------#
# class ScaleTransform:
#     """Resize a PIL image by `scale_factor` (variable output size)."""
#     def __init__(self, scale_factor=1.0, interp=Image.BICUBIC):
#         self.sf = scale_factor; self.interp = interp
#     def __call__(self, img):
#         w, h = img.size
#         return img.resize((int(w*self.sf), int(h*self.sf)), self.interp)
#
# class SegmentationScaleTransform:
#     """Scale image & mask, tensorise, normalize. No fixed output size."""
#     def __init__(self, scale_factor=1.0):
#         self.img_tf = T.Compose([
#             ScaleTransform(scale_factor, Image.BICUBIC),
#             T.ToTensor(),
#             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#         ])
#         self.mask_tf = ScaleTransform(scale_factor, Image.NEAREST)
#     def __call__(self, img, mask):
#         img  = self.img_tf(img)
#         mask = self.mask_tf(mask)
#         mask = torch.from_numpy((np.array(mask) > 0).astype(np.int64))  # 0/1
#         return img, mask
#
# # ---------------------------#
# # 5) METRICS (GLOBAL CM)
# # ---------------------------#
# def compute_metrics(preds, gts):
#     cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
#     for p, g in zip(preds, gts):
#         cm += confusion_matrix(
#             g.flatten(), p.flatten(), labels=list(range(NUM_CLASSES))
#         )
#     pixel_acc = cm.diagonal().sum() / (cm.sum() + 1e-10)
#     mean_iou   = cm[1,1] / (cm[1,1] + cm[1,0] + cm[0,1] + 1e-10)
#     return pixel_acc, mean_iou
#
# # ---------------------------#
# # 6) INFERENCE (variable sizes → batch = 1)
# # ---------------------------#
# @torch.no_grad()
# def run_inference(ds):
#     loader = DataLoader(
#         ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
#     )
#     preds, gts = [], []
#     for imgs, masks, _ in loader:
#         imgs = imgs.to(device)
#         with torch.cuda.amp.autocast():
#             out = model(imgs)['out']
#         pred = torch.argmax(out, 1).cpu().numpy()[0]
#         preds.append(pred)
#         gts.append(masks.numpy()[0])
#     return preds, gts
#
# # ---------------------------#
# # 7) PLOT
# # ---------------------------#
# def plot_scale_sweep(scales, accs, ious):
#     plt.figure(figsize=(8,6))
#     plt.plot(scales, accs, marker='o', label='Pixel Accuracy')
#     plt.plot(scales, ious, marker='s', label='Mean IoU')
#     plt.title("Leaf segmentation vs. scale factor")
#     plt.xlabel("Scale factor"); plt.ylabel("Score"); plt.ylim(0,1)
#     plt.grid(True); plt.legend(); plt.tight_layout()
#     plt.savefig("leaf_scale_vs_metrics.png"); plt.show()
#
# # ---------------------------#
# # 8) MAIN
# # ---------------------------#
# def main():
#     scale_levels = [0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0]
#     accs, ious = [], []
#     for s in scale_levels:
#         print(f"\n==> scale_factor = {s}")
#         ds = LeafSegDataset(
#             TEST_IMAGES_DIR, TEST_MASKS_DIR,
#             transform=SegmentationScaleTransform(s)
#         )
#         preds, gts = run_inference(ds)
#         acc, iou = compute_metrics(preds, gts)
#         accs.append(acc); ious.append(iou)
#         print(f"PixelAcc={acc:.4f}  MeanIoU={iou:.4f}")
#
#     plot_scale_sweep(scale_levels, accs, ious)
#     pd.DataFrame({
#         "scale_factor": scale_levels,
#         "pixel_accuracy": accs,
#         "mean_iou": ious
#     }).to_csv(CSV_NAME, index=False)
#     print(f"\nCSV saved to {CSV_NAME}")
#
# if __name__ == "__main__":
#     main()

