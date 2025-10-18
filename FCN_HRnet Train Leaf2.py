# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import numpy as np
# from PIL import Image
# from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
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
#         img_path = os.path.join(self.images_dir, img_name)
#         mask_path = os.path.join(self.masks_dir, os.path.splitext(img_name)[0] + ".png")
#         image = Image.open(img_path).convert("RGB")
#         mask  = Image.open(mask_path).convert("L")
#         if self.transform:
#             image, mask = self.transform(image, mask)
#         return image, mask
#
# class LeafSegmentationTransform:
#     def __init__(self, resize=(520,520), pad=(4,4,4,4), sigma=0, is_train=False):
#         t_img = [T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR)]
#         if is_train:
#             t_img.append(T.RandomHorizontalFlip())
#         if sigma > 0:
#             t_img.append(T.GaussianBlur((5,5), (sigma,sigma)))
#         t_img += [T.Pad(pad), T.ToTensor(),
#                   T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
#         self.img_t = T.Compose(t_img)
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
# def get_dataloaders(batch_size=8, num_workers=4):
#     train_ds = LeafDataset(train_images_dir, train_masks_dir, transform=train_transform)
#     val_ds   = LeafDataset(val_images_dir,   val_masks_dir,   transform=val_transform)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
#                               num_workers=num_workers, pin_memory=True)
#     val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
#                               num_workers=num_workers, pin_memory=True)
#     return train_loader, val_loader
#
# # ---------------------------
# # 4) MODEL, LOSS, OPTIMIZER
# # ---------------------------
# def get_model(num_classes=2):
#     weights = FCN_ResNet50_Weights.DEFAULT
#     model = fcn_resnet50(weights=weights)
#     in_ch = model.classifier[-1].in_channels
#     # Replace final conv (no bias, float32) to avoid AMP dtype issues
#     model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=False)
#     return model.to(device)
#
# criterion = nn.CrossEntropyLoss()
#
# # ---------------------------
# # 5) TRAIN/EVAL w/AMP
# # ---------------------------
# def train_epoch(model, loader, optimizer, scaler):
#     model.train()
#     total_loss = 0.0
#     for images, masks in loader:
#         images = images.to(device, non_blocking=True)
#         masks  = masks.to(device, non_blocking=True)
#         optimizer.zero_grad()
#         with autocast():
#             outputs = model(images)['out']
#             loss = criterion(outputs, masks)
#         # Scaled backward with AMP
#         scaler.scale(loss).backward()
#         # Synchronize to catch device-side errors before stepping optimizer
#         try:
#             torch.cuda.synchronize()
#         except RuntimeError as sync_err:
#             print(f"CUDA synchronization error after backward: {sync_err}")
#             raise
#         # Apply optimizer step
#         try:
#             scaler.step(optimizer)
#         except RuntimeError as opt_err:
#             print(f"Runtime error during optimizer step: {opt_err}")
#             # Attempt to clear cache and break to avoid stale state
#             torch.cuda.empty_cache()
#             raise
#         scaler.update()
#         total_loss += loss.item() * images.size(0)
#     # Final sync to ensure no pending errors
#     torch.cuda.synchronize()
#     return total_loss / len(loader.dataset)
#
# def eval_epoch(model, loader):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for images, masks in loader:
#             images = images.to(device)
#             masks  = masks.to(device)
#             outputs = model(images)['out']
#             total_loss += criterion(outputs, masks).item() * images.size(0)
#     return total_loss / len(loader.dataset)
#
# # ---------------------------
# # 6) MAIN with checkpointing
# # ---------------------------
# def main():
#     print(f"Using device: {device}")
#     train_loader, val_loader = get_dataloaders()
#     model = get_model()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     scaler = GradScaler()
#     best_loss = float('inf')
#     best_path = "finetune_fcn50_leaf.pth"
#
#     # Freeze backbone first 5 epochs
#     for param in model.backbone.parameters():
#         param.requires_grad = False
#     unfreeze_epoch = 6
#
#     try:
#         for epoch in range(1, 31):
#             if epoch == unfreeze_epoch:
#                 for param in model.backbone.parameters():
#                     param.requires_grad = True
#                 print(f"Backbone unfrozen at epoch {epoch}")
#
#             train_loss = train_epoch(model, train_loader, optimizer, scaler)
#             val_loss   = eval_epoch(model, val_loader)
#             print(f"Epoch {epoch}/30 — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#
#             # Save checkpoint each epoch
#             torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pth")
#             # Save best
#             if val_loss < best_loss:
#                 best_loss = val_loss
#                 torch.save(model.state_dict(), best_path)
#                 print(f"Best model updated (Val Loss: {val_loss:.4f}) at epoch {epoch}")
#     except KeyboardInterrupt:
#         print("Training interrupted — saving current model...")
#         torch.save(model.state_dict(), "interrupted_model.pth")
#     finally:
#         print(f"Done — Best Val Loss: {best_loss:.4f}")
#
# # ---------------------------
# # 7) INFERENCE & PLOTTING
# # ---------------------------
# def infer_and_plot(model, loader, num_samples=5):
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
#     std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
#     count = 0
#     for images, masks in loader:
#         if count >= num_samples:
#             break
#         images = images.to(device)
#         with torch.no_grad():
#             preds = model(images)['out']
#         preds_np = torch.argmax(preds, dim=1).cpu().numpy()
#         masks_np = masks.numpy()
#
#         img_vis = (images.squeeze(0) * std + mean).permute(1,2,0).cpu().numpy().clip(0,1)
#         fig, axs = plt.subplots(1,3,figsize=(15,5))
#         axs[0].imshow(img_vis); axs[0].set_title("Image"); axs[0].axis('off')
#         axs[1].imshow(masks_np[0], cmap='gray'); axs[1].set_title("Ground Truth"); axs[1].axis('off')
#         axs[2].imshow(preds_np[0], cmap='gray'); axs[2].set_title("Prediction"); axs[2].axis('off')
#         plt.tight_layout(); plt.show()
#         count += 1
#
# if __name__ == "__main__":
#     torch.multiprocessing.freeze_support()
#     main()

###blur
# import os
# import sys
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import torchvision.models as models
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from torch.cuda.amp import autocast
# from sklearn.metrics import confusion_matrix
# import pandas as pd
#
# # ---------------------------
# # 1) CONFIG + SETUP
# # ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# torch.backends.cudnn.benchmark = True  # Performance optimization
#
# # Path to your saved FCN-ResNet50 model
# model_path = "./finetune_fcn50_leaf.pth"
#
# # Instantiate FCN-ResNet50 for 2-class (leaf vs background)
# model = models.segmentation.fcn_resnet50(
#     weights=None,      # don't load the default 21-class weights
#     num_classes=2      # our fine-tuned model has 2 output channels
# )
#
# # Load the state dict
# try:
#     state_dict = torch.load(model_path, map_location=device)
# except Exception as e:
#     print(f"Error loading state dict: {e}")
#     sys.exit(1)
#
# model.load_state_dict(state_dict, strict=False)
# model = model.float().to(device)
# model.eval()
#
# # Paths to your leaf test dataset
# test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
# # ---------------------------
# # 2) LEAF SEGMENTATION DATASET
# # ---------------------------
# class LeafSegFineTuneDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, image_size=(352, 352), sigma=0):
#         self.images_dir  = images_dir
#         self.masks_dir   = masks_dir
#         self.image_files = sorted(os.listdir(images_dir))
#         self.image_size  = image_size
#         self.sigma       = sigma
#
#         self.image_transform = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
#         ])
#         self.mask_transform = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
#             T.ToTensor()
#         ])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#         image = Image.open(img_path).convert("RGB")
#         image = image.resize(self.image_size, Image.BILINEAR)
#         if self.sigma > 0:
#             image = T.functional.gaussian_blur(image, kernel_size=5, sigma=self.sigma)
#
#         mask_name = img_name.replace('.jpg', '.png')
#         mask_path = os.path.join(self.masks_dir, mask_name)
#         mask = Image.open(mask_path).convert("L")
#         mask = mask.resize(self.image_size, Image.NEAREST)
#
#         pixel_values = self.image_transform(image)
#         label = self.mask_transform(mask).squeeze(0)
#         label = (label > 0).long()
#
#         return {"pixel_values": pixel_values, "labels": label}
#
# # ---------------------------
# # 3) INFERENCE FUNCTION
# # ---------------------------
# @torch.no_grad()
# def run_inference(dataset, batch_size=4):
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=4, pin_memory=True)
#     pred_masks, gt_masks = [], []
#
#     for batch in loader:
#         imgs = batch["pixel_values"].to(device)
#         with autocast():
#             out = model(imgs)['out']  # FCN returns dict with 'out'
#             preds = out.argmax(dim=1)
#
#         for p in preds:
#             pred_masks.append((p == 1).cpu().numpy())
#         for g in batch["labels"]:
#             gt_masks.append(g.numpy())
#
#     return pred_masks, gt_masks
#
# # ---------------------------
# # 4) METRICS
# # ---------------------------
# def compute_metrics(pred_masks, gt_masks):
#     pixel_accs, ious = [], []
#     for pred, gt in zip(pred_masks, gt_masks):
#         pred_flat = pred.flatten()
#         gt_flat   = gt.flatten()
#         cm = confusion_matrix(gt_flat, pred_flat, labels=[0,1])
#         pixel_acc = np.diag(cm).sum() / cm.sum()
#         iou = cm[1,1] / (cm[1,1] + cm[0,1] + cm[1,0] + 1e-10)
#         pixel_accs.append(pixel_acc)
#         ious.append(iou)
#     return np.mean(pixel_accs), np.mean(ious)
#
# # ---------------------------
# # 5) PLOTTING
# # ---------------------------
# def plot_blur_sweep(blur_levels, pixel_accs, mious, save_path="fcn_blur_results.png"):
#     plt.figure(figsize=(8,6))
#     plt.plot(blur_levels, pixel_accs, marker='o', label='Pixel Accuracy')
#     plt.plot(blur_levels, mious,    marker='s', label='Mean IoU')
#     plt.title("FCN-ResNet50 Leaf Segmentation vs. Blur Sigma")
#     plt.xlabel("Gaussian Blur Sigma")
#     plt.ylabel("Metric Value")
#     plt.ylim(0,1)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()
#     print(f"Plot saved to: {save_path}")
#
# # ---------------------------
# # 6) MAIN
# # ---------------------------
# def main():
#     blur_levels = [0,1,2,3,4]
#     pix_acc_list, miou_list = [], []
#
#     for sigma in blur_levels:
#         print(f"\n=== Sigma={sigma} ===")
#         ds = LeafSegFineTuneDataset(test_images_dir, test_masks_dir,
#                                     image_size=(352,352), sigma=sigma)
#         preds, gts = run_inference(ds, batch_size=4)
#         pix_acc, miou = compute_metrics(preds, gts)
#         pix_acc_list.append(pix_acc)
#         miou_list.append(miou)
#         print(f"PixelAcc={pix_acc:.4f}, MeanIoU={miou:.4f}")
#
#     plot_blur_sweep(blur_levels, pix_acc_list, miou_list)
#
#     # Save CSV
#     df = pd.DataFrame({
#         "sigma": blur_levels,
#         "pixel_accuracy": pix_acc_list,
#         "mean_iou": miou_list
#     })
#     csv_path = "fcn_blur_results.csv"
#     df.to_csv(csv_path, index=False)
#     print(f"CSV saved to: {csv_path}")
#
# if __name__ == "__main__":
#     main()


###noise

# import os
# import sys
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import torchvision.models as models
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from torch.cuda.amp import autocast
# from sklearn.metrics import confusion_matrix
# import pandas as pd
#
# # ---------------------------
# # 1) CONFIG + SETUP
# # ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# torch.backends.cudnn.benchmark = True
#
# # Path to your saved FCN-ResNet50 model
# model_path = "./finetune_fcn50_leaf.pth"
#
# # Instantiate FCN-ResNet50 for 2-class (leaf vs. background)
# model = models.segmentation.fcn_resnet50(
#     weights=None,
#     num_classes=2
# )
# try:
#     state_dict = torch.load(model_path, map_location=device)
# except Exception as e:
#     print(f"Error loading state dict: {e}")
#     sys.exit(1)
# model.load_state_dict(state_dict, strict=False)
# model = model.float().to(device)
# model.eval()
#
# # Paths to your leaf test dataset
# test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
# # ---------------------------
# # 2) DATASET (with additive noise)
# # ---------------------------
# class LeafSegFineTuneDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, image_size=(352,352), noise_std=0.0):
#         self.images_dir  = images_dir
#         self.masks_dir   = masks_dir
#         self.image_files = sorted(os.listdir(images_dir))
#         self.image_size  = image_size
#         self.noise_std   = noise_std
#
#         self.image_transform = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
#         ])
#         self.mask_transform = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
#             T.ToTensor()
#         ])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#         image = Image.open(img_path).convert("RGB")
#
#         # Convert to tensor + normalize
#         pixel_values = self.image_transform(image)
#         # Apply additive Gaussian noise
#         if self.noise_std > 0:
#             pixel_values = pixel_values + torch.randn_like(pixel_values) * self.noise_std
#
#         # Load and process mask
#         mask_name = img_name.replace('.jpg', '.png')
#         mask_path = os.path.join(self.masks_dir, mask_name)
#         mask = Image.open(mask_path).convert("L")
#         label = self.mask_transform(mask).squeeze(0)
#         label = (label > 0).long()
#
#         return {"pixel_values": pixel_values, "labels": label}
#
# # ---------------------------
# # 3) INFERENCE
# # ---------------------------
# @torch.no_grad()
# def run_inference(dataset, batch_size=4):
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=4, pin_memory=True)
#     preds, gts = [], []
#
#     for batch in loader:
#         imgs = batch["pixel_values"].to(device, non_blocking=True)
#         with autocast():
#             out = model(imgs)['out']
#             pred_batch = out.argmax(dim=1)
#         preds.extend((pred_batch == 1).cpu().numpy())
#         gts.extend(batch["labels"].numpy())
#
#     return preds, gts
#
# # ---------------------------
# # 4) METRICS
# # ---------------------------
# def compute_metrics(pred_masks, gt_masks):
#     pixel_accs, ious = [], []
#     for pred, gt in zip(pred_masks, gt_masks):
#         pred_f = pred.flatten()
#         gt_f   = gt.flatten()
#         cm = confusion_matrix(gt_f, pred_f, labels=[0,1])
#         pixel_accs.append(np.diag(cm).sum() / cm.sum())
#         ious.append(cm[1,1] / (cm[1,1] + cm[0,1] + cm[1,0] + 1e-10))
#     return np.mean(pixel_accs), np.mean(ious)
#
# # ---------------------------
# # 5) PLOTTING
# # ---------------------------
# def plot_noise_sweep(noise_levels, pixel_accs, mious, save_path="noise_sweep.png"):
#     plt.figure(figsize=(8,6))
#     plt.plot(noise_levels, pixel_accs, marker='o', label='Pixel Accuracy')
#     plt.plot(noise_levels, mious,    marker='s', label='Mean IoU')
#     plt.title("Leaf Segmentation vs. Additive Gaussian Noise")
#     plt.xlabel("Noise Std")
#     plt.ylabel("Metric Value")
#     plt.ylim(0,1)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()
#     print(f"Plot saved to: {save_path}")
#
# # ---------------------------
# # 6) MAIN
# # ---------------------------
# def main():
#     noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
#     pixel_acc_list, miou_list = [], []
#
#     for std in noise_levels:
#         print(f"\n=== Noise std={std} ===")
#         ds = LeafSegFineTuneDataset(
#             test_images_dir, test_masks_dir,
#             image_size=(352,352), noise_std=std
#         )
#         preds, gts = run_inference(ds, batch_size=4)
#         pa, mi = compute_metrics(preds, gts)
#         pixel_acc_list.append(pa)
#         miou_list.append(mi)
#         print(f"PixelAcc={pa:.4f}, MeanIoU={mi:.4f}")
#
#     plot_noise_sweep(noise_levels, pixel_acc_list, miou_list)
#
#     # Save CSV
#     df = pd.DataFrame({
#         "noise_std": noise_levels,
#         "pixel_accuracy": pixel_acc_list,
#         "mean_iou": miou_list
#     })
#     csv_path = "fcn_noise_results.csv"
#     df.to_csv(csv_path, index=False)
#     print(f"CSV saved to: {csv_path}")
#
# if __name__ == "__main__":
#     main()


### ROTATE
# import os
# import sys
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import torchvision.models as models
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from torch.cuda.amp import autocast
# from sklearn.metrics import confusion_matrix
# import pandas as pd
#
# # ---------------------------
# # 1) CONFIG + SETUP
# # ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# torch.backends.cudnn.benchmark = True  # optimize for fixed-size inputs
#
# # Path to your saved FCN-ResNet50 model
# model_path = "./finetune_fcn50_leaf.pth"
#
# # Instantiate FCN-ResNet50 for 2-class (leaf vs. background)
# model = models.segmentation.fcn_resnet50(
#     weights=None,
#     num_classes=2
# )
# try:
#     state_dict = torch.load(model_path, map_location=device)
# except Exception as e:
#     print(f"Error loading state dict: {e}")
#     sys.exit(1)
# model.load_state_dict(state_dict, strict=False)
# model = model.float().to(device)
# model.eval()
#
# # Paths to your leaf test dataset
# test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
# # ---------------------------
# # 2) DATASET WITH ROTATION
# # ---------------------------
# class RotatedLeafDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, image_size=(352,352), angle=0):
#         self.images_dir  = images_dir
#         self.masks_dir   = masks_dir
#         self.image_files = sorted(os.listdir(images_dir))
#         self.image_size  = image_size
#         self.angle       = angle
#
#         # common transforms
#         self.image_transform = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
#             T.ToTensor(),
#             T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
#         ])
#         self.mask_transform = T.Compose([
#             T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
#             T.ToTensor()
#         ])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#         image = Image.open(img_path).convert("RGB")
#         # rotate input image
#         if self.angle != 0:
#             image = image.rotate(self.angle, resample=Image.BILINEAR, expand=True)
#
#         mask_name = img_name.replace('.jpg', '.png')
#         mask_path = os.path.join(self.masks_dir, mask_name)
#         mask = Image.open(mask_path).convert("L")
#         # rotate mask
#         if self.angle != 0:
#             mask = mask.rotate(self.angle, resample=Image.NEAREST, expand=True)
#
#         # apply transforms
#         pixel_values = self.image_transform(image)
#         label = self.mask_transform(mask).squeeze(0)
#         label = (label > 0).long()
#
#         return {"pixel_values": pixel_values, "labels": label}
#
# # ---------------------------
# # 3) INFERENCE
# # ---------------------------
# @torch.no_grad()
# def run_inference(dataset, batch_size=4):
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=4, pin_memory=True)
#     preds, gts = [], []
#
#     for batch in loader:
#         imgs = batch["pixel_values"].to(device, non_blocking=True)
#         with autocast():
#             out = model(imgs)['out']
#             p_batch = out.argmax(dim=1)
#         preds.extend((p_batch == 1).cpu().numpy())
#         gts.extend(batch["labels"].numpy())
#
#     return preds, gts
#
# # ---------------------------
# # 4) METRICS
# # ---------------------------
# def compute_metrics(pred_masks, gt_masks):
#     pa_list, iou_list = [], []
#     for pred, gt in zip(pred_masks, gt_masks):
#         pred_f = pred.flatten()
#         gt_f   = gt.flatten()
#         cm = confusion_matrix(gt_f, pred_f, labels=[0,1])
#         pa_list.append(np.diag(cm).sum() / cm.sum())
#         iou_list.append(cm[1,1] / (cm[1,1] + cm[0,1] + cm[1,0] + 1e-10))
#     return np.mean(pa_list), np.mean(iou_list)
#
# # ---------------------------
# # 5) PLOTTING
# # ---------------------------
# def plot_rotation_sweep(angles, pixel_accs, mious, save_path="rotation_sweep.png"):
#     plt.figure(figsize=(8,6))
#     plt.plot(angles, pixel_accs, marker='o', label='Pixel Accuracy')
#     plt.plot(angles, mious,    marker='s', label='Mean IoU')
#     plt.title("Leaf Segmentation vs. Rotation Angle")
#     plt.xlabel("Rotation Angle (degrees)")
#     plt.ylabel("Metric Value")
#     plt.ylim(0,1)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()
#     print(f"Plot saved to: {save_path}")
#
# # ---------------------------
# # 6) MAIN
# # ---------------------------
# def main():
#     angles = list(range(0, 331, 30))
#     pa_list, miou_list = [], []
#
#     for angle in angles:
#         print(f"\n=== Angle = {angle}° ===")
#         ds = RotatedLeafDataset(
#             test_images_dir, test_masks_dir,
#             image_size=(352,352), angle=angle
#         )
#         preds, gts = run_inference(ds, batch_size=4)
#         pa, mi = compute_metrics(preds, gts)
#         pa_list.append(pa)
#         miou_list.append(mi)
#         print(f"PixelAcc={pa:.4f}, MeanIoU={mi:.4f}")
#
#     plot_rotation_sweep(angles, pa_list, miou_list)
#
#     # Save CSV
#     df = pd.DataFrame({
#         "rotation_angle": angles,
#         "pixel_accuracy": pa_list,
#         "mean_iou": miou_list
#     })
#     csv_path = "fcn_rotation_results.csv"
#     df.to_csv(csv_path, index=False)
#     print(f"CSV saved to: {csv_path}")
#
# if __name__ == "__main__":
#     main()


###SCALE

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
import pandas as pd

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True

# Path to your saved FCN-ResNet50 model
model_path = "./finetune_fcn50_leaf.pth"

# Instantiate FCN-ResNet50 for 2-class (leaf vs. background)
model = models.segmentation.fcn_resnet50(
    weights=None,
    num_classes=2
)
try:
    state_dict = torch.load(model_path, map_location=device)
except Exception as e:
    print(f"Error loading state dict: {e}")
    sys.exit(1)
model.load_state_dict(state_dict, strict=False)
model = model.float().to(device)
model.eval()

# Paths to your leaf test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"

# ---------------------------
# 2) DATASET (with scaling)
# ---------------------------
class LeafSegScaleDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(352,352), scale_factor=1.0):
        self.images_dir   = images_dir
        self.masks_dir    = masks_dir
        self.image_files  = sorted(os.listdir(images_dir))
        self.image_size   = image_size
        self.scale_factor = scale_factor

        self.image_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
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
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")

        # Apply scaling on PIL images
        if self.scale_factor != 1.0:
            w, h = image.size
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            image = image.resize((new_w, new_h), Image.BILINEAR)
            mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Transform to tensor and normalize
        pixel_values = self.image_transform(image)
        # Prepare label mask
        label = self.mask_transform(mask).squeeze(0)
        label = (label > 0).long()

        return {"pixel_values": pixel_values, "labels": label}

# ---------------------------
# 3) INFERENCE
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    preds, gts = [], []
    for batch in loader:
        imgs = batch["pixel_values"].to(device, non_blocking=True)
        with autocast():
            out = model(imgs)['out']
            p_batch = out.argmax(dim=1)
        preds.extend((p_batch == 1).cpu().numpy())
        gts.extend(batch["labels"].numpy())
    return preds, gts

# ---------------------------
# 4) METRICS
# ---------------------------
def compute_metrics(pred_masks, gt_masks):
    pixel_accs, ious = [], []
    for pred, gt in zip(pred_masks, gt_masks):
        p = pred.flatten()
        g = gt.flatten()
        cm = confusion_matrix(g, p, labels=[0,1])
        pixel_accs.append(np.diag(cm).sum() / cm.sum())
        ious.append(cm[1,1] / (cm[1,1] + cm[0,1] + cm[1,0] + 1e-10))
    return np.mean(pixel_accs), np.mean(ious)

# ---------------------------
# 5) PLOTTING
# ---------------------------
def plot_scale_sweep(scale_levels, pixel_accs, mious, save_path="scale_sweep.png"):
    plt.figure(figsize=(8,6))
    plt.plot(scale_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(scale_levels, mious,    marker='s', label='Mean IoU')
    plt.title("Leaf Segmentation vs. Scale Factor")
    plt.xlabel("Scale Factor")
    plt.ylabel("Metric Value")
    plt.ylim(0,1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to: {save_path}")

# ---------------------------
# 6) MAIN
# ---------------------------
def main():
    scale_levels = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    pixel_acc_list, miou_list = [], []
    for sf in scale_levels:
        print(f"\n=== Scale factor={sf} ===")
        ds = LeafSegScaleDataset(
            test_images_dir, test_masks_dir,
            image_size=(352,352), scale_factor=sf
        )
        preds, gts = run_inference(ds, batch_size=4)
        pa, mi = compute_metrics(preds, gts)
        pixel_acc_list.append(pa)
        miou_list.append(mi)
        print(f"PixelAcc={pa:.4f}, MeanIoU={mi:.4f}")
    plot_scale_sweep(scale_levels, pixel_acc_list, miou_list)

    # Save CSV
    df = pd.DataFrame({
        "scale_factor": scale_levels,
        "pixel_accuracy": pixel_acc_list,
        "mean_iou": miou_list
    })
    csv_path = "fcn_scale_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
