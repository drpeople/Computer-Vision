import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from torch.cuda.amp import autocast
from sklearn.metrics import confusion_matrix
from torchvision.transforms import functional as TF

# ---------------------------
# 1) CONFIG + SETUP
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True  # Performance optimization

# Load CLIPSeg model
model_path = "./clipseg_finetuned_leaf"  # Ensure this path exists
processor = CLIPSegProcessor.from_pretrained(model_path)
model = CLIPSegForImageSegmentation.from_pretrained(model_path).to(device)
model.eval()

# Paths to test dataset
test_images_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\images"
test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"


# ---------------------------
# 2) LEAF SEGMENTATION DATASET WITH NOISE
# ---------------------------
class LeafSegFineTuneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor, image_size=(352, 352), prompt="a photo of a leaf",
                 noise_std=0.0):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.processor = processor
        self.image_size = image_size
        self.prompt = prompt
        self.noise_std = noise_std  # Standard deviation for Gaussian noise

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # Open and resize the image
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)

        # If noise_std > 0, add Gaussian noise to the image
        if self.noise_std > 0:
            # Convert PIL image to tensor in [0,1]
            tensor_img = TF.to_tensor(image)
            # Generate Gaussian noise; note: noise_std is in the same scale as the tensor image
            noise = torch.randn(tensor_img.size()) * self.noise_std
            noisy_img = tensor_img + noise
            noisy_img = torch.clamp(noisy_img, 0, 1)
            image = TF.to_pil_image(noisy_img)

        # Load and resize mask
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(self.image_size, Image.NEAREST)

        # Process image with the CLIPSeg processor
        inputs = self.processor(text=[self.prompt], images=[image], padding="max_length", return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Convert mask to binary tensor (leaf vs background)
        label = T.ToTensor()(mask).squeeze(0)
        label = (label > 0).float()

        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "labels": label}


# ---------------------------
# 3) INFERENCE FUNCTION
# ---------------------------
@torch.no_grad()
def run_inference(dataset, batch_size=4):
    """Runs inference with mixed precision for efficiency."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    pred_masks, gt_masks = [], []

    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast():
            outputs = model(pixel_values=batch["pixel_values"], input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"], return_dict=True)
            logits = outputs.logits
            preds = torch.sigmoid(logits) > 0.5
            preds = preds.cpu().numpy()

        gt_np = batch["labels"].cpu().numpy()

        for p, g in zip(preds, gt_np):
            pred_masks.append(p)
            gt_masks.append(g)

    return pred_masks, gt_masks


# ---------------------------
# 4) METRICS (Pixel Accuracy & Mean IoU)
# ---------------------------
def compute_metrics(pred_masks, gt_masks):
    pixel_accs = []
    ious = []
    # Compute per-image metrics and average
    for pred, gt in zip(pred_masks, gt_masks):
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
        pixel_acc = np.diag(cm).sum() / np.sum(cm)
        iou = cm[1, 1] / (cm[1, 1] + cm[0, 1] + cm[1, 0] + 1e-10)
        pixel_accs.append(pixel_acc)
        ious.append(iou)
    return np.mean(pixel_accs), np.mean(ious)


# ---------------------------
# 5) PLOTTING RESULTS
# ---------------------------
def plot_noise_sweep(noise_levels, pixel_accs, mious, save_path="clipseg_noise_results.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, pixel_accs, marker='o', label='Pixel Accuracy')
    plt.plot(noise_levels, mious, marker='s', label='Mean IoU')
    plt.title("CLIPSeg Segmentation vs. Gaussian Noise Std")
    plt.xlabel("Noise Std")
    plt.ylabel("Metric Value")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()

    # Save the plot before showing it
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    plt.show()


# ---------------------------
# 6) MAIN SCRIPT
# ---------------------------
def main():
    # Noise levels to test; adjust these values as needed
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    pixel_acc_list, miou_list = [], []

    for std in noise_levels:
        print(f"\n=== Evaluating with noise std={std} ===")
        # Pass the current noise standard deviation to the dataset
        dataset = LeafSegFineTuneDataset(test_images_dir, test_masks_dir, processor, noise_std=std)

        preds, gts = run_inference(dataset)
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
    results_df.to_csv("clipseg_noise_results.csv", index=False)
    print("CSV saved: clipseg_noise_results.csv")


if __name__ == "__main__":
    main()
