### TRAIN

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
# train_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\images"
# train_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\masks"
# val_images_dir   = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# val_masks_dir    = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True  # Auto-tune for constant-size inputs
# print(f"Using device: {device}")
#
# # ---------------------------
# # 2) DATASET DEFINITION
# # ---------------------------
# class LeafDataset(Dataset):
#     """
#     Dataset for leaf segmentation.
#     """
#     def __init__(self, images_dir, masks_dir, transform=None):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.transform = transform
#         valid_exts = ('.jpg', '.jpeg', '.png')
#         self.image_names = [
#             f for f in sorted(os.listdir(images_dir))
#             if f.lower().endswith(valid_exts)
#         ]
#
#     def __len__(self):
#         return len(self.image_names)
#
#     def __getitem__(self, idx):
#         img_name = self.image_names[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#         base = os.path.splitext(img_name)[0]
#         mask_path = os.path.join(self.masks_dir, base + ".png")
#
#         image = Image.open(img_path).convert("RGB")
#         mask  = Image.open(mask_path).convert("L")
#         if self.transform:
#             image, mask = self.transform(image, mask)
#         return image, mask
#
# # ---------------------------
# # 3) TRANSFORMATIONS
# # ---------------------------
# class LeafSegmentationTransform:
#     def __init__(self, resize=(520, 520), pad=(4,4,4,4), sigma=0, is_train=False):
#         self.resize = resize
#         self.pad = pad
#         self.sigma = sigma
#         self.is_train = is_train
#
#         image_t = [T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR)]
#         if is_train:
#             image_t.append(T.RandomHorizontalFlip())
#         if sigma > 0:
#             image_t.append(T.GaussianBlur(kernel_size=(5,5), sigma=(sigma, sigma)))
#         image_t += [T.Pad(pad), T.ToTensor(),
#                     T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
#         self.image_transform = T.Compose(image_t)
#
#         mask_t = [T.Resize(resize, interpolation=T.InterpolationMode.NEAREST), T.Pad(pad, fill=0)]
#         self.mask_transform = T.Compose(mask_t)
#
#     def __call__(self, image, mask):
#         image = self.image_transform(image)
#         mask  = self.mask_transform(mask)
#         mask_np = (np.array(mask) > 0).astype(np.int64)
#         return image, torch.from_numpy(mask_np)
#
# # Instantiate transforms
# train_transform = LeafSegmentationTransform(is_train=True)
# val_transform   = LeafSegmentationTransform(is_train=False)
#
# # ---------------------------
# # 4) DATALOADERS
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
# # 5) MODEL, LOSS, OPTIMIZER
# # ---------------------------
# def get_model(num_classes=2):
#     weights = DeepLabV3_ResNet50_Weights.DEFAULT
#     model = deeplabv3_resnet50(weights=weights, progress=True)
#     # Replace head
#     in_ch = model.classifier[-1].in_channels
#     model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
#     return model.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# # ---------------------------
# # 6) TRAIN & EVAL w/ AMP & Freezing
# # ---------------------------
# def train_epoch(model, loader, criterion, optimizer, scaler, device):
#     model.train()
#     total_loss = 0.0
#     for imgs, masks in loader:
#         imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
#         optimizer.zero_grad()
#         with autocast():
#             out = model(imgs)['out']
#             loss = criterion(out, masks)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         total_loss += loss.item() * imgs.size(0)
#     return total_loss / len(loader.dataset)
#
# def evaluate(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for imgs, masks in loader:
#             imgs, masks = imgs.to(device), masks.to(device)
#             out = model(imgs)['out']
#             total_loss += criterion(out, masks).item() * imgs.size(0)
#     return total_loss / len(loader.dataset)
#
# # ---------------------------
# # 7) TRAINING LOOP
# # ---------------------------
# def main():
#     train_loader, val_loader = get_dataloaders()
#     model = get_model()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scaler = GradScaler()
#
#     # Freeze backbone initially
#     freeze_epochs = 5
#     for param in model.backbone.parameters():
#         param.requires_grad = False
#
#     best_val = float('inf')
#     save_path = "finetuned_deeplabv3_resnet50_leaf.pth"
#
#     for epoch in range(1, 31):
#         if epoch == freeze_epochs + 1:
#             for p in model.backbone.parameters(): p.requires_grad = True
#             print("Unfroze backbone parameters.")
#
#         train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
#         val_loss   = evaluate(model, val_loader, criterion, device)
#         print(f"Epoch {epoch}/30 - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
#
#         if val_loss < best_val:
#             best_val = val_loss
#             torch.save(model.state_dict(), save_path)
#             print(f"Saved best model at epoch {epoch}.")
#
#     print("Training complete. Best Val Loss:", best_val)
#
# # ---------------------------
# # 8) INFERENCE & PLOTTING
# # ---------------------------
# def infer_and_plot(model, loader, num_samples=5):
#     mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1).to(device)
#     std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1).to(device)
#
#     count = 0
#     for imgs, masks in loader:
#         if count >= num_samples: break
#         imgs = imgs.to(device)
#         with torch.no_grad(): out = model(imgs)['out']
#         preds = torch.argmax(out, dim=1).cpu().numpy()
#         gts   = masks.numpy()
#
#         img_u = imgs.squeeze(0) * std + mean
#         img_np = img_u.permute(1,2,0).cpu().numpy().clip(0,1)
#
#         fig, ax = plt.subplots(1,3, figsize=(15,5))
#         ax[0].imshow(img_np); ax[0].set_title("Image"); ax[0].axis('off')
#         ax[1].imshow(gts[0], cmap='gray'); ax[1].set_title("GT"); ax[1].axis('off')
#         ax[2].imshow(preds[0], cmap='gray'); ax[2].set_title("Pred"); ax[2].axis('off')
#         plt.show()
#         count += 1
#
# if __name__ == '__main__':
#     torch.multiprocessing.freeze_support()
#     main()


###blur
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
# from torchvision.models.segmentation import deeplabv3_resnet50
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
# # Path to your saved DeepLabV3-ResNet50 model
# model_path = "./finetuned_deeplabv3_resnet50_leaf.pth"
#
# # ---------------------------
# # 2) MODEL LOADING
# # ---------------------------
# # Instantiate the same architecture you fine-tuned:
# model = deeplabv3_resnet50(weights=None, progress=True)
#
# # Replace classifier head to 2 classes (background vs leaf)
# in_ch = model.classifier[-1].in_channels
# model.classifier[-1] = nn.Conv2d(in_ch, 2, kernel_size=1)
# # (If you also used an auxiliary classifier during training, do the same replacement there:
# #  aux_in = model.aux_classifier[-1].in_channels
# #  model.aux_classifier[-1] = nn.Conv2d(aux_in, 2, kernel_size=1)
# # )
#
# # Load your weights
# try:
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict, strict=False)
#     print("Loaded fine-tuned weights.")
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
#         self.files      = sorted(os.listdir(images_dir))
#         self.sigma      = sigma
#
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
#         fn = self.files[idx]
#         img = Image.open(os.path.join(self.images_dir, fn)).convert("RGB")
#         if self.sigma > 0:
#             img = TF.gaussian_blur(img, kernel_size=5, sigma=self.sigma)
#
#         mask_name = os.path.splitext(fn)[0] + ".png"
#         mask = Image.open(os.path.join(self.masks_dir, mask_name)).convert("L")
#
#         img_t = self.img_tf(img)
#         mask_t = torch.from_numpy(
#             (np.array(self.mask_tf(mask)) > 0).astype(np.int64)
#         )
#
#         return {"image": img_t, "mask": mask_t}
#
# # ---------------------------
# # 4) INFERENCE + METRICS
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
#         preds_batch = torch.argmax(out, dim=1).cpu().numpy()
#         gt_batch    = batch["mask"].cpu().numpy()
#         preds.extend(preds_batch)
#         gts.extend(gt_batch)
#     return preds, gts
#
# def compute_metrics(preds, gts):
#     pix_accs, ious = [], []
#     for p, g in zip(preds, gts):
#         cm = confusion_matrix(g.flatten(), p.flatten(), labels=[0,1])
#         pix_acc = cm.diagonal().sum() / cm.sum()
#         iou     = cm[1,1] / (cm[1,1] + cm[0,1] + cm[1,0] + 1e-10)
#         pix_accs.append(pix_acc)
#         ious.append(iou)
#     return np.mean(pix_accs), np.mean(ious)
#
# # ---------------------------
# # 5) SWEEP & PLOT
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
# # 6) MAIN
# # ---------------------------
# def main():
#     sigmas = [0,1,2,3,4]
#     acc_list, miou_list = [], []
#     for s in sigmas:
#         print(f"\n--> Evaluating sigma={s}")
#         ds = LeafSegInferenceDataset(test_images_dir, test_masks_dir, sigma=s)
#         preds, gts = run_inference(ds)
#         acc, miou = compute_metrics(preds, gts)
#         acc_list.append(acc); miou_list.append(miou)
#         print(f"Sigma={s}: PixelAcc={acc:.4f}, MeanIoU={miou:.4f}")
#
#     plot_blur_sweep(sigmas, acc_list, miou_list)
#
#     df = pd.DataFrame({
#         "sigma": sigmas,
#         "pixel_accuracy": acc_list,
#         "mean_iou": miou_list
#     })
#     df.to_csv("deeplabv3_resnet50_blur_results.csv", index=False)
#     print("Saved CSV: deeplabv3_resnet50_blur_results.csv")
#
# if __name__ == "__main__":
#     main()


#### NOISE
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
from torchvision.transforms import functional as TF
import segmentation_models_pytorch as smp
import torch.nn as nn

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Path to your saved model state dict
model_path = "finetuned_deeplabv3plus_leaf.pth"

# Initialize the model (encoder_weights=None to load our state dict)
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    encoder_weights=None,
    in_channels=3,
    classes=2
)

# Load the state dict
try:
    state_dict = torch.load(model_path, map_location=device)
except Exception as e:
    print(f"Error loading state dict: {e}")
    sys.exit(1)

model.load_state_dict(state_dict, strict=False)

# --- Patch modules to fix problematic attributes ---
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        # Force padding_mode to be "zeros"
        m.padding_mode = "zeros"
    if isinstance(m, torch.nn.Upsample):
        m.mode = "bilinear"

model = model.float().to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH NOISE
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(352, 352), noise_std=0.0):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.image_size  = image_size
        self.noise_std   = noise_std

        self.image_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)

        # Optionally add Gaussian noise
        if self.noise_std > 0:
            tensor_img = TF.to_tensor(image)
            noise = torch.randn(tensor_img.size()) * self.noise_std
            noisy_img = torch.clamp(tensor_img + noise, 0, 1)
            image = TF.to_pil_image(noisy_img)

        pixel_values = self.image_transform(image)

        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.image_size, Image.NEAREST)

        # Binarize mask and cast to int64 for sklearn
        label = self.mask_transform(mask).squeeze(0)
        label = (label > 0).long()

        return {"pixel_values": pixel_values, "labels": label}

# ---------------------------
# 3) INFERENCE FUNCTION
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    pred_masks, gt_masks = [], []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)

        with autocast():
            outputs = model(pixel_values)  # returns tensor [B, classes, H, W]
            preds = torch.argmax(outputs, dim=1)

        for p in preds:
            pred_masks.append((p == 1).cpu().numpy().astype(np.int64))
        for g in batch["labels"]:
            gt_masks.append(g.cpu().numpy().astype(np.int64))

    return pred_masks, gt_masks

# ---------------------------
# 4) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(preds, gts):
    pix_accs, ious = [], []
    for p, g in zip(preds, gts):
        cm = confusion_matrix(g.flatten(), p.flatten(), labels=[0, 1])
        # pixel accuracy
        pix_acc = cm.diagonal().sum() / cm.sum()
        # IoU for leaf class
        iou = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0] + 1e-10)
        pix_accs.append(pix_acc)
        ious.append(iou)
    return np.mean(pix_accs), np.mean(ious)

# ---------------------------
# 5) PLOTTING RESULTS
# ---------------------------
def plot_noise_sweep(noise_levels, pixel_accs, mious, save_path="deeplabv3plus_noise_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(noise_levels, mious, marker='s', label='Mean IoU')
    plt.title("DeepLabV3+ Segmentation vs. Gaussian Noise Std")
    plt.xlabel("Noise Std")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()

# ---------------------------
# 6) MAIN SCRIPT
# ---------------------------
def main():
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    pixel_acc_list, miou_list = [], []

    for std in noise_levels:
        print(f"\n=== Evaluating with noise std={std} ===")
        ds = LeafSegFineTuneDataset(
            test_images_dir,
            test_masks_dir,
            image_size=(352, 352),
            noise_std=std
        )
        preds, gts = run_inference(ds)
        pix_acc, miou = compute_metrics(preds, gts)
        pixel_acc_list.append(pix_acc)
        miou_list.append(miou)
        print(f"Results (noise_std={std}): PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")

    plot_noise_sweep(noise_levels, pixel_acc_list, miou_list)

    results_df = pd.DataFrame({
        "noise_std": noise_levels,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    results_df.to_csv("deeplabv3plus_noise_results.csv", index=False)
    print("CSV saved: deeplabv3plus_noise_results.csv")

if __name__ == "__main__":
    main()

#### ROTATE
#
# """
# DeepLabV3+ rotation-sweep evaluation
# ===================================
# Measures how robust the fine-tuned ResNet-101 DeepLabV3+ model is to
# in-plane image rotations.  Every test image and its mask are rotated by
# θ ∈ {0°, 30°, …, 360°}, resized/padded to **528 × 528** (a multiple of 16),
# and fed through the network.  We report pixel accuracy and foreground
# IoU for each θ, save a CSV, and draw a plot.
#
# Design notes (mirrors the blur-sweep script)
# -------------------------------------------
# * **Dataset reuse** – brand-new `LeafSegRotationDataset` copies the
#   I/O/normalisation pipeline from `LeafSegInferenceDataset` but adds a
#   PIL rotation step.
# * **Shape sanity** – everything is resized to 528 × 528 before tensor
#   conversion; segmentation_models_pytorch bypasses shape errors.
# * **Model lives in the main process** to avoid Windows worker spam.
# * **`num_workers=0`** by default (Windows-friendly).  Raise it on Linux
#   for more speed.
# """
#
# import os
# import sys
# import math
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
# import segmentation_models_pytorch as smp
# import torch.nn as nn
#
# # -----------------------------------------------------------------------------
# # 1) DATASET – rotation version
# # -----------------------------------------------------------------------------
#
# test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
#
# class LeafSegRotationDataset(Dataset):
#     """Leaf test set with a fixed in-plane rotation applied to both image and mask."""
#
#     def __init__(self, images_dir: str, masks_dir: str,
#                  image_size: tuple[int, int] = (528, 528), angle: int = 0):
#         self.images_dir = images_dir
#         self.masks_dir  = masks_dir
#         self.files      = sorted(os.listdir(images_dir))
#         self.angle      = angle  # degrees
#         self.size       = image_size
#
#         self.img_tf = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         self.mask_tf = T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)
#
#     def __len__(self):
#         return len(self.files)
#
#     def _rotate(self, img: Image.Image, resample, is_mask=False):
#         """Rotate with expand=True, then center-crop/pad back to square size."""
#         rotated = img.rotate(self.angle, resample=resample, expand=True)
#         # Center-crop longer side, then resize to target
#         w, h = rotated.size
#         short = min(w, h)
#         left = (w - short) // 2
#         upper = (h - short) // 2
#         cropped = rotated.crop((left, upper, left + short, upper + short))
#         return cropped
#
#     def __getitem__(self, idx):
#         fname = self.files[idx]
#         img = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")
#         mask_name = os.path.splitext(fname)[0] + ".png"
#         mask = Image.open(os.path.join(self.masks_dir, mask_name)).convert("L")
#
#         if self.angle % 360 != 0:
#             img = self._rotate(img, resample=Image.BILINEAR)
#             mask = self._rotate(mask, resample=Image.NEAREST)
#
#         img_t  = self.img_tf(img)
#         mask_t = torch.from_numpy((np.array(self.mask_tf(mask)) > 0).astype(np.int64))
#         return {"image": img_t, "mask": mask_t}
#
#
# # -----------------------------------------------------------------------------
# # 2) MODEL LOADING (identical to blur script)
# # -----------------------------------------------------------------------------
#
# MODEL_PATH = "finetuned_deeplabv3plus_leaf.pth"
#
# def load_model(device: torch.device):
#     model = smp.DeepLabV3Plus(
#         encoder_name="resnet101",
#         encoder_weights=None,
#         in_channels=3,
#         classes=2,
#     )
#     state_dict = torch.load(MODEL_PATH, map_location=device)
#     missing, unexpected = model.load_state_dict(state_dict, strict=False)
#     print(f"✔  Checkpoint loaded (missing={len(missing)}, unexpected={len(unexpected)})")
#
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             m.padding_mode = "zeros"
#         if isinstance(m, nn.Upsample):
#             m.mode = "bilinear"
#
#     return model.to(device).eval()
#
#
# # -----------------------------------------------------------------------------
# # 3) INFERENCE + METRICS
# # -----------------------------------------------------------------------------
#
# @torch.no_grad()
# def run_inference(model: nn.Module, dataset: Dataset, device: torch.device,
#                   batch_size: int = 4, num_workers: int = 0):
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=num_workers, pin_memory=device.type == "cuda")
#     preds, gts = [], []
#     for batch in loader:
#         imgs = batch["image"].to(device)
#         with autocast():
#             logits = model(imgs)
#             pred = torch.argmax(logits, dim=1)
#         preds.extend(pred.cpu().numpy())
#         gts.extend(batch["mask"].cpu().numpy())
#     return preds, gts
#
#
# def compute_metrics(preds, gts):
#     cm = confusion_matrix(np.concatenate(gts).ravel(),
#                           np.concatenate(preds).ravel(),
#                           labels=[0, 1])
#     pix_acc = cm.diagonal().sum() / cm.sum()
#     iou_fg  = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0] + 1e-10)
#     return pix_acc, iou_fg
#
#
# # -----------------------------------------------------------------------------
# # 4) PLOTTING
# # -----------------------------------------------------------------------------
#
# def plot_rotation_sweep(angles, accs, ious,
#                         save_path: str = "deeplabv3plus_rotation_results.png"):
#     plt.figure(figsize=(8, 6))
#     plt.plot(angles, accs, marker="s", label="Pixel Accuracy")
#     plt.plot(angles, ious, marker="o", label="IoU (leaf)")
#     plt.title("DeepLabV3+ • Rotation robustness")
#     plt.xlabel("Rotation angle (°)")
#     plt.ylabel("Metric value")
#     plt.ylim(0, 1)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     print(f"📈  Plot saved → {save_path}")
#     if os.getenv("DISPLAY", ""):
#         plt.show()
#     plt.close()
#
#
# # -----------------------------------------------------------------------------
# # 5) MAIN
# # -----------------------------------------------------------------------------
#
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     model = load_model(device)
#
#     angles = list(range(0, 361, 30))  # 0, 30, …, 360
#     pix_accs, ious = [], []
#
#     for a in angles:
#         print(f"\n=== Evaluating θ={a}° ===")
#         ds = LeafSegRotationDataset(test_images_dir, test_masks_dir,
#                                     image_size=(528, 528), angle=a)
#         preds, gts = run_inference(model, ds, device,
#                                    batch_size=4, num_workers=0)
#         pix_acc, iou = compute_metrics(preds, gts)
#         pix_accs.append(pix_acc)
#         ious.append(iou)
#         print(f"θ={a}°: PixelAcc={pix_acc:.4f}, IoU={iou:.4f}")
#
#     result_csv = "deeplabv3plus_rotation_results.csv"
#     pd.DataFrame({
#         "rotation_angle": angles,
#         "pixel_accuracy": pix_accs,
#         "iou_leaf": ious,
#     }).to_csv(result_csv, index=False)
#     print(f"📄  CSV saved → {result_csv}")
#
#     plot_rotation_sweep(angles, pix_accs, ious)
#
#
# if __name__ == "__main__":
#     main()

### SCALE

"""
DeepLabV3+ scale‑sweep evaluation
================================
Tests how the fine‑tuned **ResNet‑101 DeepLabV3+** model handles global
isotropic scaling of its inputs.  Each test image / mask is first
resized to the baseline **352 × 352**, then uniformly scaled by a factor
*s* ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}.  After scaling,
both image and mask are **padded** (not resized!) to the next multiple
of 16 so that segmentation_models_pytorch accepts the shape.  The model
runs once per scale factor; we record pixel accuracy and foreground IoU
and plot the results.

Key points (mirrors the blur/rotation scripts)
---------------------------------------------
* **Dataset class** – `LeafSegScaleDataset` applies scaling and dynamic
  padding while preserving content resolution.
* **Model initialisation** – lives in `main()`, loaded once, Windows‑
  friendly.
* **Padding helper** – guarantees height & width divisible by 16 while
  keeping the scaled pixels intact (zero‑padding on right/bottom).
* **Workers** – default `num_workers=0` for Windows; feel free to bump
  on Linux.
"""

# import os
# import sys
# import math
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image, ImageOps
# from torch.cuda.amp import autocast
# from sklearn.metrics import confusion_matrix
# from torchvision.transforms import functional as TF
# import segmentation_models_pytorch as smp
# import torch.nn as nn
# import torch.nn.functional as F
#
# # -----------------------------------------------------------------------------
# # 1) PATHS
# # -----------------------------------------------------------------------------
#
# test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
# MODEL_PATH      = "finetuned_deeplabv3plus_leaf.pth"
#
# # -----------------------------------------------------------------------------
# # 2) DATASET – SCALE VARIANT
# # -----------------------------------------------------------------------------
#
# def _pad_to_multiple(img: Image.Image, multiple: int = 16, fill: int = 0):
#     """Pad right/bottom so that both dims are divisible by *multiple*."""
#     w, h = img.size
#     pad_w = (-w) % multiple
#     pad_h = (-h) % multiple
#     if pad_w == 0 and pad_h == 0:
#         return img
#     return ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=fill)
#
#
# class LeafSegScaleDataset(Dataset):
#     """Leaf test set with isotropic scaling + padding to mult‑of‑16."""
#
#     def __init__(self, images_dir: str, masks_dir: str,
#                  base_size: tuple[int, int] = (352, 352), scale: float = 1.0):
#         self.images_dir = images_dir
#         self.masks_dir  = masks_dir
#         self.files      = sorted(os.listdir(images_dir))
#         self.base_size  = base_size
#         self.scale      = scale
#
#         self.norm_tf = T.Normalize(mean=[0.485, 0.456, 0.406],
#                                    std=[0.229, 0.224, 0.225])
#
#     def __len__(self):
#         return len(self.files)
#
#     def _load_pair(self, fname):
#         img = Image.open(os.path.join(self.images_dir, fname)).convert("RGB")
#         mask_name = os.path.splitext(fname)[0] + ".png"
#         mask = Image.open(os.path.join(self.masks_dir, mask_name)).convert("L")
#         return img, mask
#
#     def __getitem__(self, idx):
#         fname = self.files[idx]
#         img, mask = self._load_pair(fname)
#
#         # 1) Resize to baseline size
#         img  = img.resize(self.base_size, Image.BILINEAR)
#         mask = mask.resize(self.base_size, Image.NEAREST)
#
#         # 2) Uniform scaling
#         if self.scale != 1.0:
#             new_w = int(self.base_size[0] * self.scale)
#             new_h = int(self.base_size[1] * self.scale)
#             img  = img.resize((new_w, new_h), Image.BILINEAR)
#             mask = mask.resize((new_w, new_h), Image.NEAREST)
#
#         # 3) Pad to next multiple of 16
#         img  = _pad_to_multiple(img, multiple=16, fill=0)
#         mask = _pad_to_multiple(mask, multiple=16, fill=0)
#
#         # 4) To tensor & normalise
#         img_t  = TF.to_tensor(img)
#         img_t  = self.norm_tf(img_t)
#         mask_t = torch.from_numpy((np.array(mask) > 0).astype(np.int64))
#
#         return {"image": img_t, "mask": mask_t}
#
#
# # -----------------------------------------------------------------------------
# # 3) MODEL LOADER
# # -----------------------------------------------------------------------------
#
# def load_model(device: torch.device):
#     model = smp.DeepLabV3Plus(
#         encoder_name="resnet101",
#         encoder_weights=None,
#         in_channels=3,
#         classes=2,
#     )
#     state_dict = torch.load(MODEL_PATH, map_location=device)
#     missing, unexpected = model.load_state_dict(state_dict, strict=False)
#     print(f"✔  Checkpoint loaded (missing={len(missing)}, unexpected={len(unexpected)})")
#
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             m.padding_mode = "zeros"
#         if isinstance(m, nn.Upsample):
#             m.mode = "bilinear"
#
#     return model.to(device).eval()
#
#
# # -----------------------------------------------------------------------------
# # 4) INFERENCE & METRICS
# # -----------------------------------------------------------------------------
#
# @torch.no_grad()
# def run_inference(model: nn.Module, dataset: Dataset, device: torch.device,
#                   batch_size: int = 4, num_workers: int = 0):
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=num_workers, pin_memory=device.type == "cuda")
#     preds, gts = [], []
#     for batch in loader:
#         imgs = batch["image"].to(device)
#         with autocast():
#             logits = model(imgs)
#             pred = torch.argmax(logits, dim=1)
#         preds.extend(pred.cpu().numpy())
#         gts.extend(batch["mask"].cpu().numpy())
#     return preds, gts
#
#
# def compute_metrics(preds, gts):
#     cm = confusion_matrix(np.concatenate(gts).ravel(),
#                           np.concatenate(preds).ravel(),
#                           labels=[0, 1])
#     pix_acc = cm.diagonal().sum() / cm.sum()
#     iou_fg  = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0] + 1e-10)
#     return pix_acc, iou_fg
#
#
# # -----------------------------------------------------------------------------
# # 5) PLOTTING
# # -----------------------------------------------------------------------------
#
# def plot_scale_sweep(scales, accs, ious,
#                      save_path: str = "deeplabv3plus_scale_results.png"):
#     plt.figure(figsize=(8, 6))
#     plt.plot(scales, accs, marker="o", label="Pixel Accuracy")
#     plt.plot(scales, ious, marker="s", label="IoU (leaf)")
#     plt.title("DeepLabV3+ • Scale robustness")
#     plt.xlabel("Scale factor (×)")
#     plt.ylabel("Metric value")
#     plt.ylim(0, 1)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150)
#     print(f"📈  Plot saved → {save_path}")
#     if os.getenv("DISPLAY", ""):
#         plt.show()
#     plt.close()
#
#
# # -----------------------------------------------------------------------------
# # 6) MAIN
# # -----------------------------------------------------------------------------
#
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     model = load_model(device)
#
#     scales = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
#     pix_accs, ious = [], []
#
#     for s in scales:
#         print(f"\n=== Evaluating scale ×{s} ===")
#         ds = LeafSegScaleDataset(test_images_dir, test_masks_dir,
#                                  base_size=(352, 352), scale=s)
#         preds, gts = run_inference(model, ds, device,
#                                    batch_size=4, num_workers=0)
#         pix_acc, iou = compute_metrics(preds, gts)
#         pix_accs.append(pix_acc)
#         ious.append(iou)
#         print(f"×{s}: PixelAcc={pix_acc:.4f}, IoU={iou:.4f}")
#
#     result_csv = "deeplabv3plus_scale_results.csv"
#     pd.DataFrame({
#         "scale_factor": scales,
#         "pixel_accuracy": pix_accs,
#         "iou_leaf": ious,
#     }).to_csv(result_csv, index=False)
#     print(f"📄  CSV saved → {result_csv}")
#
#     plot_scale_sweep(scales, pix_accs, ious)
#
#
# if __name__ == "__main__":
#     main()
