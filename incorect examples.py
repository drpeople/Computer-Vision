### Incorect predicted examples Segmentation
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as T
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
# from torch.cuda.amp import autocast
# from sklearn.metrics import confusion_matrix
# from torchvision.transforms import functional as TF
#
# # ---------------------------
# # 1) CONFIG + SETUP
# # ---------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True
#
# model_path = "./mask2former_finetuned_leaf"  # update as needed
# processor = AutoImageProcessor.from_pretrained(model_path)
# model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path).to(device)
# model.eval()
#
# test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
# test_masks_dir  = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
# # ---------------------------
# # 2) CUSTOM COLLATE_FN FOR PIL IMAGES
# # ---------------------------
# def collate_with_pil(batch):
#     pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
#     labels       = torch.stack([item["labels"]      for item in batch], dim=0)
#     orig_images  = [item["orig_image"]  for item in batch]
#     orig_masks   = [item["orig_mask"]   for item in batch]
#     return {
#         "pixel_values": pixel_values,
#         "labels":       labels,
#         "orig_image":   orig_images,
#         "orig_mask":    orig_masks
#     }
#
# # ---------------------------
# # 3) DATASET
# # ---------------------------
# class LeafSegFineTuneDataset(Dataset):
#     def __init__(self, images_dir, masks_dir, processor, image_size=(352,352), noise_std=0.0):
#         self.images_dir  = images_dir
#         self.masks_dir   = masks_dir
#         self.image_files = sorted(os.listdir(images_dir))
#         self.processor   = processor
#         self.image_size  = image_size
#         self.noise_std   = noise_std
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#
#         # 1) load full-res originals
#         orig_image = Image.open(img_path).convert("RGB")
#         mask_name  = img_name.replace('.jpg', '.png')
#         orig_mask  = Image.open(os.path.join(self.masks_dir, mask_name)).convert("L")
#
#         # 2) resize for model input
#         image = orig_image.resize(self.image_size, Image.BILINEAR)
#         if self.noise_std > 0:
#             t = TF.to_tensor(image)
#             t = torch.clamp(t + torch.randn_like(t) * self.noise_std, 0, 1)
#             image = TF.to_pil_image(t)
#
#         # 3) prepare label
#         mask = orig_mask.resize(self.image_size, Image.NEAREST)
#         label = T.ToTensor()(mask).squeeze(0)
#         label = (label > 0).float()
#
#         # 4) processor -> pixel_values
#         inputs = self.processor(images=image, return_tensors="pt")
#         pixel_values = inputs["pixel_values"].squeeze(0)
#
#         return {
#             "pixel_values": pixel_values,  # for model
#             "labels":       label,         # resized GT mask
#             "orig_image":   orig_image,    # full-res for plotting
#             "orig_mask":    orig_mask      # full-res for plotting
#         }
#
# # ---------------------------
# # 4) INFERENCE
# # ---------------------------
# @torch.no_grad()
# def run_inference(dataset, batch_size=4):
#     loader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=False,
#         num_workers=4, pin_memory=True, collate_fn=collate_with_pil
#     )
#
#     orig_images, orig_masks, pred_masks, gt_masks = [], [], [], []
#
#     for batch in loader:
#         orig_images.extend(batch["orig_image"])
#         orig_masks.extend(batch["orig_mask"])
#         gt_masks.extend([m.numpy() for m in batch["labels"]])
#
#         pv = batch["pixel_values"].to(device)
#         with autocast():
#             outputs = model(pixel_values=pv, return_dict=True)
#             sizes = [dataset.image_size[::-1]] * pv.size(0)
#             segs  = processor.post_process_semantic_segmentation(outputs, target_sizes=sizes)
#
#         for seg in segs:
#             pm = (seg == 1).cpu().numpy().astype(np.uint8)
#             pred_masks.append(pm)
#
#     return orig_images, orig_masks, pred_masks, gt_masks
#
# # ---------------------------
# # 5) METRICS
# # ---------------------------
# def compute_metrics(pred_masks, gt_masks):
#     pixel_accs, ious = [], []
#     for p, g in zip(pred_masks, gt_masks):
#         p_flat = p.flatten()
#         g_flat = g.flatten()
#         cm = confusion_matrix(g_flat, p_flat, labels=[0,1])
#         pixel_accs.append(np.diag(cm).sum() / cm.sum())
#         ious.append(cm[1,1] / (cm[1,1] + cm[0,1] + cm[1,0] + 1e-10))
#     return np.mean(pixel_accs), np.mean(ious)
#
# # ---------------------------
# # 6) PLOTTING CORRECT vs WRONG (with guards)
# # ---------------------------
# def plot_correct_vs_wrong(orig_images, orig_masks, pred_masks, gt_masks,
#                           num_examples=5, save_prefix="leaf_seg"):
#     correct_idx = [i for i,(p,g) in enumerate(zip(pred_masks,gt_masks)) if np.array_equal(p,g)]
#     wrong_idx   = [i for i in range(len(pred_masks)) if i not in correct_idx]
#
#     def _plot(indices, title, fname):
#         n = min(len(indices), num_examples)
#         fig, axs = plt.subplots(n, 3, figsize=(12, 4*n))
#         fig.suptitle(title)
#         for row,i in enumerate(indices[:n]):
#             axs[row,0].imshow(orig_images[i]); axs[row,0].axis("off"); axs[row,0].set_title("Image")
#             axs[row,1].imshow(orig_masks[i], cmap="gray"); axs[row,1].axis("off"); axs[row,1].set_title("GT Mask")
#             axs[row,2].imshow(pred_masks[i], cmap="gray"); axs[row,2].axis("off"); axs[row,2].set_title("Pred Mask")
#         plt.tight_layout(rect=[0,0,1,0.95])
#         plt.savefig(f"{fname}.png")
#         plt.show()
#
#     if correct_idx:
#         _plot(correct_idx, "Correctly Predicted Examples", f"{save_prefix}_correct")
#     else:
#         print("No correctly predicted examples to plot.")
#
#     if wrong_idx:
#         _plot(wrong_idx, "Incorrectly Predicted Examples", f"{save_prefix}_wrong")
#     else:
#         print("No incorrectly predicted examples to plot.")
#
# # ---------------------------
# # 7) MAIN
# # ---------------------------
# def main():
#     ds = LeafSegFineTuneDataset(
#         test_images_dir, test_masks_dir,
#         processor, image_size=(352,352), noise_std=0.0
#     )
#
#     orig_imgs, orig_msks, preds, gts = run_inference(ds, batch_size=4)
#
#     pix_acc, miou = compute_metrics(preds, gts)
#     print(f"Pixel Accuracy: {pix_acc:.4f}, Mean IoU: {miou:.4f}")
#
#     plot_correct_vs_wrong(orig_imgs, orig_msks, preds, gts,
#                           num_examples=5, save_prefix="leaf_seg")
#
# if __name__ == "__main__":
#     main()


### incorect examples Recognition

import os
import torch
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import requests

# ---------------------- SETUP ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Load pretrained ViT-B-16 model
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights).to(device)
model.eval()

# Load ImageNet class names
dataset_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = requests.get(dataset_url).text.splitlines()

# Normalization parameters
norm_mean = weights.meta.get("mean", (0.5, 0.5, 0.5))
norm_std = weights.meta.get("std", (0.5, 0.5, 0.5))

# Base folder for validation images (subfolders named by class index)
base_folder = r"C:\Users\goker\PycharmProjects\DiplomProject\ILSVRC2012_img_val_subset"

# ---------------------- DATASET ----------------------
class ImageDataset(Dataset):
    def __init__(self, base_folder, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for subfolder in os.listdir(base_folder):
            subfolder_path = os.path.join(base_folder, subfolder)
            if os.path.isdir(subfolder_path):
                label_idx = int(subfolder)
                for fname in os.listdir(subfolder_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(subfolder_path, fname))
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        return img_tensor, label, path

# Inference transform (no noise)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_mean, std=norm_std),
])

dataset = ImageDataset(base_folder, transform=preprocess)

# ---------------------- FIND FIRST MISPREDICTION AND PLOT ----------------------
for i in range(len(dataset)):
    img_tensor, true_label, img_path = dataset[i]
    img_input = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_input)
        pred_idx = torch.argmax(outputs, dim=1).item()

    if pred_idx != true_label:
        # Load original image
        orig = Image.open(img_path).convert("RGB")

        # Plot with labels underneath each image
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(bottom=0.2)
        for ax, label, text in zip(
            axes,
            [true_label, pred_idx],
            ["True", "Predicted"]
        ):
            ax.imshow(orig)
            ax.axis('off')
            # Place label text below image
            ax.set_title(f"{text}: {imagenet_classes[label]}", y=-0.1)
            ax.title.set_fontsize(12)

        plt.tight_layout()
        plt.show()
        print(f"Stopped at image: {img_path}")
        print(f"True label: {imagenet_classes[true_label]}")
        print(f"Predicted label: {imagenet_classes[pred_idx]}")
        break


### incorect examples Localization
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
# from ultralytics import YOLO
#
# # ---------------------- SETUP ----------------------
# val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
# annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"
#
# # Load COCO annotations and YOLOv8 model
# coco = COCO(annotation_path)
# model = YOLO('yolov8l.pt')
#
# def get_pred_label(pred):
#     # Map YOLO class index to label name
#     return model.names[int(pred)]
#
# # ---------------------- FIND FIRST MISPREDICTION ----------------------
# for img_id in coco.getImgIds():
#     img_info = coco.loadImgs(img_id)[0]
#     fname = img_info['file_name']
#     img_path = os.path.join(val_images_path, fname)
#
#     # Load and convert image\
#     img_bgr = cv2.imread(img_path)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#
#     # Ground truth: take first annotation
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     anns = coco.loadAnns(ann_ids)
#     if not anns:
#         continue
#     gt_ann = anns[0]
#     x, y, w, h = gt_ann['bbox']
#     gt_box = [int(x), int(y), int(x + w), int(y + h)]
#     gt_label = coco.loadCats([gt_ann['category_id']])[0]['name']
#
#     # Run inference
#     results = model.predict(source=img_rgb, conf=0.25, verbose=False)
#     boxes = results[0].boxes.data.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 6))
#
#     # Determine best prediction
#     if boxes.size > 0:
#         best = boxes[np.argmax(boxes[:, 4])]
#         pred_box = [int(best[0]), int(best[1]), int(best[2]), int(best[3])]
#         pred_label = get_pred_label(best[5])
#     else:
#         pred_box = None
#         pred_label = 'No Detection'
#
#     # Check mismatch
#     if pred_label != gt_label:
#         # Draw ground truth box (green)
#         img_gt = img_rgb.copy()
#         cv2.rectangle(img_gt, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)
#         cv2.putText(img_gt, gt_label, (gt_box[0], gt_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
#
#         # Draw predicted box (blue)
#         img_pred = img_rgb.copy()
#         if pred_box is not None:
#             cv2.rectangle(img_pred, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (255, 0, 0), 2)
#             cv2.putText(img_pred, pred_label, (pred_box[0], pred_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
#
#         # Side-by-side plot
#         fig, axes = plt.subplots(1, 2, figsize=(14, 7))
#         fig.subplots_adjust(bottom=0.15)
#
#         axes[0].imshow(img_gt)
#         axes[0].axis('off')
#         axes[0].set_title('Ground Truth', fontsize=14)
#
#         axes[1].imshow(img_pred)
#         axes[1].axis('off')
#         axes[1].set_title('Prediction', fontsize=14)
#
#         plt.show()
#         print(f"Stopped at image: {img_path}")
#         print(f"True label: {gt_label}")
#         print(f"Predicted label: {pred_label}")
#         break
