# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# from torchvision import transforms
# from transformers import MaskFormerConfig, MaskFormerModel, AutoImageProcessor
# from torch.cuda.amp import autocast, GradScaler
# import torch.nn.functional as F
#
# # Fixed image size – note that the base MaskFormer expects images to be resized.
# IMAGE_SIZE = (512, 512)
# GT_THRESHOLD = 0.05
#
#
# # --- Helper function to move data to device ---
# def move_to_device(x, device):
#     if isinstance(x, torch.Tensor):
#         return x.to(device)
#     elif isinstance(x, list):
#         return [move_to_device(item, device) for item in x]
#     else:
#         return x
#
#
# # --- Custom Dataset for Fine-Tuning ---
# class LeafSegFineTuneDataset(Dataset):
#     """
#     Loads an image and its corresponding segmentation map.
#     The segmentation map is assumed to be a grayscale image where nonzero values indicate the leaf.
#     """
#
#     def __init__(self, images_dir, masks_dir, image_size=IMAGE_SIZE):
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.image_files = sorted(os.listdir(images_dir))
#         self.image_size = image_size
#         self.image_transform = transforms.Compose([
#             transforms.Resize(image_size, interpolation=Image.BILINEAR),
#         ])
#         self.mask_transform = transforms.Compose([
#             transforms.Resize(image_size, interpolation=Image.NEAREST),
#         ])
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.images_dir, img_name)
#         image = Image.open(img_path).convert("RGB")
#         image = self.image_transform(image)
#
#         mask_name = os.path.splitext(img_name)[0] + ".png"
#         mask_path = os.path.join(self.masks_dir, mask_name)
#         mask = Image.open(mask_path).convert("L")
#         mask = self.mask_transform(mask)
#
#         # Binarize the mask using GT_THRESHOLD.
#         mask_np = np.array(mask)
#         mask_np = (mask_np > (GT_THRESHOLD * 255)).astype(np.int64)
#
#         return {
#             "image": image,
#             "segmentation_map": mask_np
#         }
#
#
# # --- Custom collate function ---
# def collate_fn(batch):
#     images = [item["image"] for item in batch]
#     segmentation_maps = [item["segmentation_map"] for item in batch]
#     return {"images": images, "segmentation_maps": segmentation_maps}
#
#
# # --- Define a simple segmentation model with a segmentation head ---
# class MaskFormerSegmentationModel(nn.Module):
#     def __init__(self, pretrained_model_name, num_labels):
#         super().__init__()
#         # Load configuration and backbone.
#         config = MaskFormerConfig.from_pretrained(pretrained_model_name)
#         self.backbone = MaskFormerModel.from_pretrained(pretrained_model_name, config=config)
#         # Change the segmentation head to accept input dimension of 16.
#         self.segmentation_head = nn.Linear(16, num_labels)
#
#     def forward(self, pixel_values, **kwargs):
#         outputs = self.backbone(pixel_values, return_dict=True, **kwargs)
#         # Get the encoder's last hidden state.
#         x = outputs.encoder_last_hidden_state
#         # If x is 3D (batch, seq_len, feature_dim), pool over the sequence dimension.
#         if x.dim() == 3:
#             pooled = x.mean(dim=1)  # shape: (batch, feature_dim)
#         elif x.dim() == 4:
#             # Pool over spatial dimensions if 4D.
#             pooled = x.mean(dim=[1, 2])
#         else:
#             raise ValueError("Unexpected tensor shape: {}".format(x.shape))
#         logits = self.segmentation_head(pooled)  # shape: (batch, num_labels)
#         return logits
#
#
# # --- Training and Validation Functions ---
# def train_one_epoch(model, dataloader, optimizer, device, scaler, processor, num_labels):
#     model.train()
#     total_loss = 0.0
#     # Use BCEWithLogitsLoss; expects input and target to have the same shape.
#     criterion = nn.BCEWithLogitsLoss()
#
#     for batch in dataloader:
#         inputs = processor(images=batch["images"], return_tensors="pt")
#         # For this simplified example, derive a global label per image using majority class in the segmentation map.
#         labels = []
#         for seg_map in batch["segmentation_maps"]:
#             unique, counts = np.unique(seg_map, return_counts=True)
#             label = unique[np.argmax(counts)]
#             labels.append(label)
#         labels = torch.tensor(labels, dtype=torch.long)
#         # Convert labels to one-hot encoding (shape: [batch, num_labels]).
#         labels_onehot = F.one_hot(labels, num_classes=num_labels).float()
#
#         inputs = {k: move_to_device(v, device) for k, v in inputs.items()}
#         labels_onehot = labels_onehot.to(device)
#
#         optimizer.zero_grad()
#         with autocast():
#             logits = model(**inputs)
#             loss = criterion(logits, labels_onehot)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         total_loss += loss.item()
#     return total_loss / len(dataloader)
#
#
# def validate_one_epoch(model, dataloader, device, processor, num_labels):
#     model.eval()
#     total_loss = 0.0
#     criterion = nn.BCEWithLogitsLoss()
#
#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = processor(images=batch["images"], return_tensors="pt")
#             labels = []
#             for seg_map in batch["segmentation_maps"]:
#                 unique, counts = np.unique(seg_map, return_counts=True)
#                 label = unique[np.argmax(counts)]
#                 labels.append(label)
#             labels = torch.tensor(labels, dtype=torch.long)
#             labels_onehot = F.one_hot(labels, num_classes=num_labels).float()
#
#             inputs = {k: move_to_device(v, device) for k, v in inputs.items()}
#             labels_onehot = labels_onehot.to(device)
#
#             logits = model(**inputs)
#             loss = criterion(logits, labels_onehot)
#             total_loss += loss.item()
#     return total_loss / len(dataloader)
#
#
# # --- Main Fine-Tuning Script ---
# def main():
#     # Update these paths to your training and validation data directories.
#     train_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\images"
#     train_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\masks"
#     val_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
#     val_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#     batch_size = 4
#     num_epochs = 6
#     learning_rate = 1e-4
#
#     train_dataset = LeafSegFineTuneDataset(train_images_dir, train_masks_dir, image_size=IMAGE_SIZE)
#     val_dataset = LeafSegFineTuneDataset(val_images_dir, val_masks_dir, image_size=IMAGE_SIZE)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                               num_workers=4, pin_memory=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                             num_workers=4, pin_memory=True, collate_fn=collate_fn)
#
#     pretrained_model_name = "facebook/maskformer-swin-tiny-ade"
#     num_labels = 2  # e.g., background and leaf.
#     model = MaskFormerSegmentationModel(pretrained_model_name, num_labels)
#     processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
#     model.to(device)
#
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scaler = GradScaler()
#
#     print("Starting fine-tuning...")
#     for epoch in range(num_epochs):
#         train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, processor, num_labels)
#         val_loss = validate_one_epoch(model, val_loader, device, processor, num_labels)
#         print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
#
#     # Save model state and processor.
#     save_dir = "./maskformer_finetuned_leaf"
#     os.makedirs(save_dir, exist_ok=True)
#     torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
#     processor.save_pretrained(save_dir)
#     print("Fine-tuning complete. Model and processor saved to:", save_dir)
#
#
# if __name__ == "__main__":
#     main()


import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data._utils.collate import default_collate
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# Settings
IMAGE_SIZE = (512, 512)
GT_THRESHOLD = 0.05  # for binarizing ground-truth masks


def custom_collate_fn(batch):
    """
    Custom collate function to keep original PIL images (for plotting)
    while default-collating the rest.
    """
    original_pils = [item.pop("original_pil") for item in batch]
    collated = default_collate(batch)
    collated["original_pil"] = original_pils
    return collated


# --- Dataset Definition ---
class LeafSegDataset(Dataset):
    """
    Loads the original PIL image (for plotting) and processes it for the model.
    Converts the ground truth mask to a binary mask (0 for background, 1 for leaf).
    """

    def __init__(self, images_dir, masks_dir, processor, image_size=IMAGE_SIZE):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor = processor
        self.image_size = image_size
        # Transformation for the mask: resize and convert to tensor.
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.NEAREST),
            transforms.ToTensor()  # converts mask to float tensor in [0,1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Load original image for plotting.
        original_pil = Image.open(img_path).convert("RGB")
        # Resize image for model input.
        pil_image = original_pil.resize(self.image_size, Image.BILINEAR)

        # Load mask; assume corresponding mask has same base name with .png extension.
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask_img = Image.open(mask_path).convert("L")

        # Debug: print unique pixel values in the raw mask.
        mask_vals = set(mask_img.getdata())
        print(f"[DEBUG] Mask file: {mask_name}, unique pixel values: {mask_vals}")

        # Transform mask and apply threshold for binarization.
        mask = self.mask_transform(mask_img)  # shape: (1, H, W)
        mask = (mask > GT_THRESHOLD).float().squeeze(0)  # shape: (H, W)
        print(f"[DEBUG] Mask file: {mask_name}, unique values after threshold: {torch.unique(mask).tolist()}")

        # Process the image using the processor.
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # shape: (3, H, W)

        return {
            "original_pil": original_pil,  # for plotting
            "pixel_values": pixel_values,  # model input
            "labels": mask  # binary ground truth mask
        }


def compute_iou(pred_mask, gt_mask):
    """Compute IoU for binary masks of shape (H, W)."""
    gt_mask = gt_mask.to(pred_mask.device)
    intersection = ((pred_mask == 1) & (gt_mask == 1)).float().sum()
    union = ((pred_mask == 1) | (gt_mask == 1)).float().sum()
    if union == 0:
        return 1.0
    return (intersection / union).item()


# -------------------------------
# Main Evaluation Script
# -------------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the processor.
    save_dir = "./maskformer_finetuned_leaf"  # update to your saved model directory
    processor = AutoImageProcessor.from_pretrained(save_dir)

    # **Force the model to have 2 labels (background, leaf)**
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        save_dir,
        num_labels=2,  # we want 2-class segmentation
        ignore_mismatched_sizes=True  # allows re-sizing final layer if needed
    )

    # **Set label mapping** for the model config.
    model.config.id2label = {0: "background", 1: "leaf"}
    model.config.label2id = {"background": 0, "leaf": 1}

    # Debug prints to verify config.
    print("model.config.num_labels =", model.config.num_labels)
    print("model.config.id2label =", model.config.id2label)
    print("model.config.label2id =", model.config.label2id)

    model.to(device)
    model.eval()

    # Create test dataset and DataLoader using the custom collate function.
    test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
    test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
    test_dataset = LeafSegDataset(test_images_dir, test_masks_dir, processor, image_size=IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    total_iou = 0.0
    num_samples = 0
    plot_samples = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)  # shape: (B, 3, H, W)
            labels = batch["labels"]  # shape: (B, H, W) on CPU

            # Forward pass.
            outputs = model(pixel_values=pixel_values, return_dict=True)

            # Debug: print available output keys and, if available, logits stats.
            print("[DEBUG] Model outputs keys:", outputs.keys())
            if hasattr(outputs, "sem_seg_logits"):
                sem_logits = outputs.sem_seg_logits  # typically shape: (B, num_labels, H, W)
                print("[DEBUG] sem_seg_logits shape:", sem_logits.shape)
                print("[DEBUG] Logits min, max:",
                      sem_logits.min().item(), sem_logits.max().item())
                # Direct prediction via argmax (bypassing processor post-processing)
                direct_pred = sem_logits.argmax(dim=1)
                print("[DEBUG] Unique values from direct argmax prediction:",
                      torch.unique(direct_pred).tolist())

            # Use the processor to post-process the segmentation map.
            target_sizes = [IMAGE_SIZE[::-1]] * pixel_values.size(0)
            pred_maps_list = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)

            for i, pred_map in enumerate(pred_maps_list):
                # Debug: see which IDs were predicted.
                unique_ids = torch.unique(pred_map).tolist()
                print(f"[DEBUG] Unique predicted class IDs from processor: {unique_ids}")

                # For binary segmentation, assume class ID 1 is "leaf".
                pred_mask = (pred_map == 1).long()
                gt = labels[i].long()
                iou = compute_iou(pred_mask, gt)
                total_iou += iou
                num_samples += 1

                plot_samples.append({
                    "original_pil": batch["original_pil"][i],
                    "gt_mask": gt,
                    "pred_mask": pred_mask,
                    "pred_map": pred_map
                })

    mean_iou = total_iou / num_samples if num_samples > 0 else 0.0
    print(f"Mean IoU over test data: {mean_iou:.4f}")

    # Plot a few samples.
    num_to_plot = min(4, len(plot_samples))
    for idx in range(num_to_plot):
        sample = plot_samples[idx]
        image = sample["original_pil"]
        gt_mask = sample["gt_mask"]
        pred_mask = sample["pred_mask"]
        pred_map = sample["pred_map"]

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(gt_mask.cpu(), cmap="gray")
        plt.title("GT Mask")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(pred_mask.cpu(), cmap="gray")
        plt.title("Pred Mask (Leaf class)")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(pred_map.cpu(), cmap="viridis")
        plt.title("Predicted Semantic Map")
        plt.axis("off")

        plt.show()
