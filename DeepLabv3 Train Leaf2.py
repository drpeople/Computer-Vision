# ##TRAIN
#
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import numpy as np
# from PIL import Image
# from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
# import torch.nn as nn
# import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler
# import matplotlib.pyplot as plt
#
# # ---------------------------
# # 1) CONFIG + PATHS
# # ---------------------------
# train_images_dir = r"C:\\Users\\goker\\PycharmProjects\\DiplomProject\\leaf\\data\\images"
# train_masks_dir  = r"C:\\Users\\goker\\PycharmProjects\\DiplomProject\\leaf\\data\\masks"
# val_images_dir   = r"C:\\Users\\goker\\PycharmProjects\\DiplomProject\\leaf\\test\\images"
# val_masks_dir    = r"C:\\Users\\goker\\PycharmProjects\\DiplomProject\\leaf\\test\\masks"
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
#         self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg",".jpeg",".png"))])
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img_name = self.images[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#         mask_path = os.path.join(self.masks_dir, os.path.splitext(img_name)[0] + ".png")
#
#         img = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")
#
#         if self.transform:
#             img, mask = self.transform(img, mask)
#         return img, mask
#
# class LeafSegmentationTransform:
#     """Apply identical geometric / photometric transforms to image & mask"""
#     def __init__(self, resize=(520, 520), pad=(4, 4, 4, 4), sigma=0, is_train=False):
#         img_t = [T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR)]
#         if is_train:
#             img_t.append(T.RandomHorizontalFlip())
#         if sigma > 0:
#             img_t.append(T.GaussianBlur((5, 5), (sigma, sigma)))
#         img_t += [
#             T.Pad(pad),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#         self.img_t = T.Compose(img_t)
#         self.mask_t = T.Compose([
#             T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
#             T.Pad(pad, fill=0),
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
#
# def get_dataloaders(batch_size: int = 8, num_workers: int = 8):
#     train_ds = LeafDataset(train_images_dir, train_masks_dir, transform=train_transform)
#     val_ds   = LeafDataset(val_images_dir,   val_masks_dir,   transform=val_transform)
#
#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=True,
#         prefetch_factor=2,
#         persistent_workers=True,
#     )
#     val_loader   = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         prefetch_factor=2,
#         persistent_workers=True,
#     )
#     return train_loader, val_loader
#
# # ---------------------------
# # 4) MODEL, LOSS, OPTIMIZER
# # ---------------------------
#
# def get_model(num_classes: int = 2):
#     """Return a DeepLabV3‑ResNet50 model with a fresh classifier head"""
#     weights = DeepLabV3_ResNet50_Weights.DEFAULT
#     model = deeplabv3_resnet50(weights=weights).to(device)
#
#     # Replace classifier head so that it outputs `num_classes` channels instead of 21
#     in_ch = model.classifier[-1].in_channels
#     model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=False).to(device)
#     return model
#
# criterion = nn.CrossEntropyLoss()
#
# # ---------------------------
# # 5) TRAIN / EVAL (mixed precision)
# # ---------------------------
#
# def train_epoch(model, loader, optimizer, scaler):
#     model.train()
#     total_loss = 0.0
#
#     for imgs, masks in loader:
#         imgs   = imgs.to(device, non_blocking=True)
#         masks  = masks.to(device, non_blocking=True)
#
#         optimizer.zero_grad(set_to_none=True)
#         with autocast():
#             outputs = model(imgs)["out"]
#             loss    = criterion(outputs, masks)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#
#         total_loss += loss.item() * imgs.size(0)
#
#     return total_loss / len(loader.dataset)
#
#
# def eval_epoch(model, loader):
#     model.eval()
#     total_loss = 0.0
#
#     with torch.no_grad():
#         for imgs, masks in loader:
#             imgs  = imgs.to(device, non_blocking=True)
#             masks = masks.to(device, non_blocking=True)
#             outputs = model(imgs)["out"]
#             total_loss += criterion(outputs, masks).item() * imgs.size(0)
#
#     return total_loss / len(loader.dataset)
#
# # ---------------------------
# # 6) MAIN loop with checkpointing
# # ---------------------------
#
# def main():
#     train_loader, val_loader = get_dataloaders()
#     model = get_model()
#
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scaler    = GradScaler()
#
#     best_loss = float("inf")
#     best_path = "finetune_deeplabv3_resnet50_leaf.pth"
#
#     # Freeze backbone for a few warm‑up epochs
#     for p in model.backbone.parameters():
#         p.requires_grad = False
#     unfreeze_epoch = 6
#
#     num_epochs = 30
#     try:
#         for epoch in range(1, num_epochs + 1):
#             # Unfreeze backbone at the chosen epoch
#             if epoch == unfreeze_epoch:
#                 for p in model.backbone.parameters():
#                     p.requires_grad = True
#                 print(f"Backbone unfrozen at epoch {epoch}")
#
#             train_loss = train_epoch(model, train_loader, optimizer, scaler)
#             val_loss   = eval_epoch(model, val_loader)
#
#             print(f"Epoch {epoch}/{num_epochs} — Train: {train_loss:.4f} | Val: {val_loss:.4f}")
#
#             # Save checkpoint every epoch
#             ckpt_name = f"checkpoint_resnet50_epoch{epoch}.pth"
#             torch.save(model.state_dict(), ckpt_name)
#
#             # Track the best model
#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 torch.save(model.state_dict(), best_path)
#                 print(f"** Best model updated (Val {val_loss:.4f}) at epoch {epoch}")
#     except KeyboardInterrupt:
#         print("Training interrupted — saving current weights…")
#         torch.save(model.state_dict(), "interrupted_deeplabv3_resnet50_leaf.pth")
#     finally:
#         print(f"Training complete — best validation loss: {best_loss:.4f}")
#
# # ---------------------------
# # 7) QUICK INFERENCE VISUALISATION
# # ---------------------------
#
# def infer_and_plot(model, loader, num_samples: int = 5):
#     """Plot a few sample predictions"""
#     mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
#     std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
#
#     shown = 0
#     for imgs, masks in loader:
#         if shown >= num_samples:
#             break
#         imgs = imgs.to(device)
#         with torch.no_grad():
#             logits = model(imgs)["out"]
#         preds = torch.argmax(logits, 1).cpu().numpy()
#         gts   = masks.numpy()
#
#         # Undo normalisation for display
#         img_vis = (imgs[0] * std + mean).permute(1, 2, 0).cpu().numpy().clip(0, 1)
#
#         fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#         axs[0].imshow(img_vis)
#         axs[0].set_title("Image"); axs[0].axis("off")
#         axs[1].imshow(gts[0], cmap="gray")
#         axs[1].set_title("GT"); axs[1].axis("off")
#         axs[2].imshow(preds[0], cmap="gray")
#         axs[2].set_title("Prediction"); axs[2].axis("off")
#         plt.tight_layout(); plt.show()
#
#         shown += 1
#
# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()


## NOISE
#
# # ---------------------------#
# # 0) IMPORTS
# # ---------------------------#
# import os, sys
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# from torchvision.models.segmentation import (
#     deeplabv3_resnet50,   # <── was resnet101
# )
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# from sklearn.metrics import confusion_matrix
#
# # ---------------------------#
# # 1) CONFIG + SETUP
# # ---------------------------#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# torch.backends.cudnn.benchmark = True
#
# LEAF_ROOT       = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf"
# TEST_IMAGES_DIR = os.path.join(LEAF_ROOT, "test", "images")
# TEST_MASKS_DIR  = os.path.join(LEAF_ROOT, "test", "masks")
#
# # >>> NEW model weights <<<
# WEIGHTS_PATH    = r".\finetune_deeplabv3_resnet50_leaf.pth"
# CSV_NAME        = "deeplabv3_resnet50_noise_results.csv"
#
# NUM_CLASSES = 2  # background, leaf
#
# # ---------------------------#
# # 2) MODEL: LOAD & PATCH HEAD
# # ---------------------------#
# model = deeplabv3_resnet50(weights=None, progress=True)        # <── was resnet101
#
# # Replace classifier (and aux head) to output NUM_CLASSES channels
# in_ch = model.classifier[-1].in_channels
# model.classifier[-1] = nn.Conv2d(in_ch, NUM_CLASSES, kernel_size=1, bias=False)
#
# if getattr(model, "aux_classifier", None) is not None:
#     aux_in = model.aux_classifier[-1].in_channels
#     model.aux_classifier[-1] = nn.Conv2d(aux_in, NUM_CLASSES, kernel_size=1, bias=False)
#
# try:
#     load = model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device), strict=False)
#     if load.unexpected_keys:
#         print("ℹ️  Ignored keys:", load.unexpected_keys)
#     print("✅ Weights loaded from:", WEIGHTS_PATH)
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
#     """Loads (image, mask, filename) tuples from the leaf test split."""
#     def __init__(self, img_dir, mask_dir, transform=None):
#         self.img_dir   = img_dir
#         self.mask_dir  = mask_dir
#         self.transform = transform
#         self.files     = sorted(
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith((".png", ".jpg", ".jpeg"))
#         )
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
#         return img, mask, fn   # (tensor, tensor, filename)
#
# # ---------------------------#
# # 4) CUSTOM GAUSSIAN-NOISE TF
# # ---------------------------#
# class AddGaussianNoise(torch.nn.Module):
#     def __init__(self, mean=0.0, std=0.0):
#         super().__init__()
#         self.mean, self.std = mean, std
#
#     def forward(self, x):
#         if self.std <= 0:
#             return x
#         return x + torch.randn_like(x) * self.std + self.mean
#
# # ---------------------------#
# # 5) TRANSFORM PIPELINE
# # ---------------------------#
# class SegmentationTransform:
#     """Resize → tensor → noise → normalize   (mask: resize→(>0)→long)."""
#     def __init__(self, resize=(520, 520), noise_std=0.0):
#         self.img_tf = T.Compose([
#             T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             AddGaussianNoise(std=noise_std),
#             T.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225]),
#         ])
#         self.mask_tf = T.Compose([
#             T.Resize(resize, interpolation=T.InterpolationMode.NEAREST)
#         ])
#
#     def __call__(self, img, mask):
#         img  = self.img_tf(img)
#         mask = self.mask_tf(mask)
#         mask = torch.from_numpy((np.array(mask) > 0).astype(np.int64))
#         return img, mask
#
# def build_transform(resize=(520, 520), noise_std=0.0):
#     return SegmentationTransform(resize, noise_std)
#
# # ---------------------------#
# # 6) METRICS (PixelAcc & IoU)
# # ---------------------------#
# def compute_metrics(preds, gts):
#     cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
#     for p, g in zip(preds, gts):
#         cm += confusion_matrix(g.flatten(), p.flatten(),
#                                labels=list(range(NUM_CLASSES)))
#     pixel_acc = cm.diagonal().sum() / (cm.sum() + 1e-10)
#     iou       = cm[1,1] / (cm[1,1] + cm[1,0] + cm[0,1] + 1e-10)
#     return pixel_acc, iou
#
# # ---------------------------#
# # 7) INFERENCE (mixed precision)
# # ---------------------------#
# @torch.no_grad()
# def run_inference(dataset, batch_size=4):
#     loader = DataLoader(dataset, batch_size=batch_size,
#                         shuffle=False, num_workers=4,
#                         pin_memory=True)
#     preds, gts = [], []
#     for imgs, masks, _ in loader:   # ⬅️ note the filename is ignored here
#         imgs = imgs.to(device, non_blocking=True)
#         with torch.cuda.amp.autocast():
#             out = model(imgs)['out']
#         pred = torch.argmax(out, dim=1).cpu().numpy()
#         preds.extend(pred)
#         gts.extend(masks.numpy())
#     return preds, gts
#
# # ---------------------------#
# # 8) PLOT UTILS
# # ---------------------------#
# def plot_noise_sweep(noise_levels, accs, ious):
#     plt.figure(figsize=(8,6))
#     plt.plot(noise_levels, accs, marker='o', label='Pixel Accuracy')
#     plt.plot(noise_levels, ious, marker='s', label='Leaf IoU')
#     plt.title("Leaf segmentation vs. Gaussian noise (std)")
#     plt.xlabel("Gaussian noise σ")
#     plt.ylabel("Score")
#     plt.ylim(0,1)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("leaf_noise_vs_metrics.png")
#     plt.show()
#
# # ---------------------------#
# # 9) MAIN
# # ---------------------------#
# def main():
#     noise_stds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
#     acc_list, iou_list = [], []
#
#     for std in noise_stds:
#         print(f"\n==> Evaluating with noise σ = {std}")
#         ds = LeafSegDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR,
#                             transform=build_transform(noise_std=std))
#         preds, gts = run_inference(ds, batch_size=4)
#         acc, iou   = compute_metrics(preds, gts)
#         acc_list.append(acc)
#         iou_list.append(iou)
#         print(f"σ={std:.2f} · PixelAcc={acc:.4f}, MeanIoU={iou:.4f}")
#
#     # A) Plot
#     plot_noise_sweep(noise_stds, acc_list, iou_list)
#
#     # B) Save to CSV
#     pd.DataFrame({
#         "noise_std":       noise_stds,
#         "pixel_accuracy":  acc_list,
#         "mean_iou":        iou_list,
#     }).to_csv(CSV_NAME, index=False)
#     print(f"\nCSV saved to: {CSV_NAME}")
#
# if __name__ == "__main__":
#     main()


## ROTATE
#
# # ---------------------------#
# # 0) IMPORTS
# # ---------------------------#
# import os, sys
# import torch, torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# from torchvision.transforms import functional as TF
# # ➜ switched backbone ↓
# from torchvision.models.segmentation import deeplabv3_resnet50
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
#
# # ➜ NEW weights path ↓
# WEIGHTS_PATH    = r".\finetune_deeplabv3_resnet50_leaf.pth"
#
# NUM_CLASSES = 2
# CSV_NAME    = "leaf_rotation_ignore_resnet50_results.csv"
#
# # ---------------------------#
# # 2) MODEL
# # ---------------------------#
# model = deeplabv3_resnet50(weights=None, progress=True)        # ← was resnet101
#
# # patch heads to 2-channel
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
#     if load.unexpected_keys:
#         print("ℹ️  Ignored keys:", load.unexpected_keys)
#     print("✅ Weights loaded from:", WEIGHTS_PATH)
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
#         self.files     = sorted(
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith((".png", ".jpg", ".jpeg"))
#         )
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
#     def __init__(self, angle=0, resize=(520, 520)):
#         self.angle  = angle
#         self.resize = resize
#         self.rot    = RotatePairExpand(angle)
#         self.img_tf = T.Compose([
#             T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406],
#                         [0.229, 0.224, 0.225]),
#         ])
#         self.mask_tf = T.Resize(resize, interpolation=T.InterpolationMode.NEAREST)
#
#     def __call__(self, img, mask):
#         # 1) rotate both
#         img_r, mask_r = self.rot(img, mask)
#         # 2) track original pixels
#         valid_map = Image.new("L", img.size, 1)
#         valid_r   = valid_map.rotate(self.angle, resample=Image.NEAREST, expand=True)
#         # 3) resize
#         img_t    = self.img_tf(img_r)
#         mask_rs  = self.mask_tf(mask_r)
#         valid_rs = valid_r.resize(self.resize, Image.NEAREST)
#         # 4) build mask & set ignore label
#         mask_np = (np.array(mask_rs) > 0).astype(np.uint8)
#         mask_np[valid_rs == 0] = 255   # ignore label
#         return img_t, torch.from_numpy(mask_np.astype(np.int64))
#
# def build_transform(angle):
#     return SegmentationTransform(angle=angle, resize=(520, 520))
#
# # ---------------------------#
# # 5) METRICS
# # ---------------------------#
# def compute_metrics(preds, gts):
#     cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
#     for p, g in zip(preds, gts):
#         valid = (g.flatten() != 255)
#         cm += confusion_matrix(g.flatten()[valid],
#                                p.flatten()[valid],
#                                labels=list(range(NUM_CLASSES)))
#     pixel_acc = cm.diagonal().sum() / (cm.sum() + 1e-10)
#     mean_iou  = cm[1,1] / (cm[1,1] + cm[0,1] + cm[1,0] + 1e-10)
#     return pixel_acc, mean_iou
#
# # ---------------------------#
# # 6) INFERENCE
# # ---------------------------#
# @torch.no_grad()
# def run_inference(ds, batch_size=4):
#     loader = DataLoader(ds, batch_size=batch_size,
#                         shuffle=False, num_workers=4,
#                         pin_memory=True)
#     preds, gts = [], []
#     for imgs, masks, _ in loader:
#         imgs = imgs.to(device, non_blocking=True)
#         with torch.cuda.amp.autocast():
#             out = model(imgs)['out']
#         preds.extend(torch.argmax(out, 1).cpu().numpy())
#         gts.extend(masks.numpy())
#     return preds, gts
#
# # ---------------------------#
# # 7) PLOT
# # ---------------------------#
# def plot_rotation_sweep(angles, accs, ious):
#     plt.figure(figsize=(8, 6))
#     plt.plot(angles, accs, marker='o', label='Pixel Accuracy')
#     plt.plot(angles, ious, marker='s', label='Mean IoU')
#     plt.title("Leaf segmentation vs. rotation angle\n(expand=True, corners=ignore)")
#     plt.xlabel("Rotation angle (°)")
#     plt.ylabel("Score")
#     plt.ylim(0, 1)
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
#         ds = LeafSegDataset(TEST_IMAGES_DIR, TEST_MASKS_DIR,
#                             transform=build_transform(a))
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
#         "mean_iou":       ious
#     }).to_csv(CSV_NAME, index=False)
#     print(f"\nCSV saved to {CSV_NAME}")
#
# if __name__ == "__main__":
#     main()


## SCALE
#
# # ---------------------------#
# # 0) IMPORTS
# # ---------------------------#
# import os, sys
# import torch, torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# # ➜ backbone switched here ↓
# from torchvision.models.segmentation import deeplabv3_resnet50
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
#
# # ➜ new weights file ↓
# WEIGHTS_PATH    = r".\finetune_deeplabv3_resnet50_leaf.pth"
#
# NUM_CLASSES = 2          # background / leaf
# CSV_NAME    = "deeplabv3_resnet50_scale_results.csv"
#
# # ---------------------------#
# # 2) MODEL (2-class head)
# # ---------------------------#
# model = deeplabv3_resnet50(weights=None, progress=True)        # ← was resnet101
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
#     if load.unexpected_keys:
#         print("ℹ️  Ignored keys:", load.unexpected_keys)
#     print("✅ Weights loaded from:", WEIGHTS_PATH)
# except Exception as e:
#     print("❌ Could not load weights:", e); sys.exit(1)
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
#         self.files = sorted(
#             f for f in os.listdir(img_dir)
#             if f.lower().endswith((".jpg", ".jpeg", ".png"))
#         )
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
#             T.Normalize([0.485, 0.456, 0.406],
#                         [0.229, 0.224, 0.225])
#         ])
#         self.mask_tf = ScaleTransform(scale_factor, Image.NEAREST)
#     def __call__(self, img, mask):
#         img  = self.img_tf(img)
#         mask = self.mask_tf(mask)
#         mask = torch.from_numpy((np.array(mask) > 0).astype(np.int64))
#         return img, mask
#
# # ---------------------------#
# # 5) METRICS
# # ---------------------------#
# def compute_metrics(preds, gts):
#     cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
#     for p, g in zip(preds, gts):
#         cm += confusion_matrix(g.flatten(), p.flatten(),
#                                labels=list(range(NUM_CLASSES)))
#     pixel_acc = cm.diagonal().sum() / (cm.sum() + 1e-10)
#     mean_iou  = cm[1,1] / (cm[1,1] + cm[1,0] + cm[0,1] + 1e-10)
#     return pixel_acc, mean_iou
#
# # ---------------------------#
# # 6) INFERENCE (batch=1, variable size)
# # ---------------------------#
# @torch.no_grad()
# def run_inference(ds):
#     loader = DataLoader(ds, batch_size=1,
#                         shuffle=False, num_workers=0,
#                         pin_memory=False)
#     preds, gts = [], []
#     for imgs, masks, _ in loader:
#         imgs = imgs.to(device)
#         with torch.cuda.amp.autocast():
#             out = model(imgs)['out']
#         preds.append(torch.argmax(out, 1).cpu().numpy()[0])
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
#     scale_levels = [0.1, 0.25, 0.5, 0.75, 1.0,1.25, 1.5, 2.0]
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
#         "scale_factor":   scale_levels,
#         "pixel_accuracy": accs,
#         "mean_iou":       ious
#     }).to_csv(CSV_NAME, index=False)
#     print(f"\nCSV saved to {CSV_NAME}")
#
# if __name__ == "__main__":
#     main()

# ---------------------------
# 0) IMPORTS
# ---------------------------
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as TF
# ➜ backbone switched here ↓
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True

model_path = "./finetune_deeplabv3_resnet50_leaf.pth"          # ← new file

# ---------------------------
# 2) MODEL LOADING
# ---------------------------
model = deeplabv3_resnet50(weights=None, progress=True)        # ← was resnet101
in_ch = model.classifier[-1].in_channels
model.classifier[-1] = nn.Conv2d(in_ch, 2, kernel_size=1, bias=False)
if getattr(model, "aux_classifier", None) is not None:
    aux_in = model.aux_classifier[-1].in_channels
    model.aux_classifier[-1] = nn.Conv2d(aux_in, 2, kernel_size=1, bias=False)

try:
    state_dict = torch.load(model_path, map_location=device)
    load_info  = model.load_state_dict(state_dict, strict=False)
    if load_info.unexpected_keys:
        print("ℹ️  Ignored keys:", load_info.unexpected_keys)
    print(f"✅ Weights loaded from: {model_path}")
except Exception as e:
    print(f"❌ Error loading state dict: {e}")
    sys.exit(1)

model = model.to(device).eval().float()

# ---------------------------
# 3) TEST DATASET & TRANSFORMS
# ---------------------------
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

class LeafSegInferenceDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(520, 520), sigma=0):
        self.images_dir, self.masks_dir = images_dir, masks_dir
        self.files  = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self.sigma  = sigma
        self.img_tf = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
        self.mask_tf = T.Resize(image_size,
                                interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn   = self.files[idx]
        img  = Image.open(os.path.join(self.images_dir, fn)).convert("RGB")
        if self.sigma > 0:
            img = TF.gaussian_blur(img, kernel_size=5, sigma=self.sigma)

        mask = Image.open(
            os.path.join(self.masks_dir,
                         os.path.splitext(fn)[0] + ".png")
        ).convert("L")

        img_t  = self.img_tf(img)
        mask_t = torch.from_numpy(
            (np.array(self.mask_tf(mask)) > 0).astype(np.int64)
        )
        return {"image": img_t, "mask": mask_t}

# ---------------------------
# 4) INFERENCE
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    preds, gts = [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        with autocast():
            logits = model(imgs)["out"]
        preds.extend(torch.argmax(logits, 1).cpu().numpy())
        gts.extend(batch["mask"].cpu().numpy())
    return preds, gts

# ---------------------------
# 5) METRICS
# ---------------------------
def compute_metrics(preds, gts):
    cm = np.zeros((2, 2), dtype=np.int64)
    for p, g in zip(preds, gts):
        cm += confusion_matrix(g.flatten(), p.flatten(), labels=[0, 1])
    pixel_acc = cm.diagonal().sum() / (cm.sum() + 1e-10)
    mean_iou  = cm[1,1] / (cm[1,1] + cm[0,1] + cm[1,0] + 1e-10)
    return pixel_acc, mean_iou

# ---------------------------
# 6) PLOT
# ---------------------------
def plot_blur_sweep(sigmas, accs, mious):
    plt.figure(figsize=(8,6))
    plt.plot(sigmas, accs,  marker='o', label='Pixel Accuracy')
    plt.plot(sigmas, mious, marker='s', label='Mean IoU')
    plt.title("Segmentation vs Gaussian Blur")
    plt.xlabel("Blur Sigma"); plt.ylabel("Score"); plt.ylim(0,1)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig("blur_vs_metrics.png"); plt.show()

# ---------------------------
# 7) MAIN
# ---------------------------
def main():
    sigmas = [0, 1, 2, 3, 4]
    acc_list, miou_list = [], []
    for s in sigmas:
        print(f"\n--> Evaluating sigma = {s}")
        ds = LeafSegInferenceDataset(test_images_dir, test_masks_dir, sigma=s)
        preds, gts = run_inference(ds)
        acc, miou  = compute_metrics(preds, gts)
        acc_list.append(acc); miou_list.append(miou)
        print(f"Sigma={s}: PixelAcc={acc:.4f}, MeanIoU={miou:.4f}")

    plot_blur_sweep(sigmas, acc_list, miou_list)

    pd.DataFrame({
        "sigma": sigmas,
        "pixel_accuracy": acc_list,
        "mean_iou": miou_list,
    }).to_csv("deeplabv3_resnet50_blur_results.csv", index=False)   # ← new CSV
    print("Saved CSV: deeplabv3_resnet50_blur_results.csv")

if __name__ == "__main__":
    main()
