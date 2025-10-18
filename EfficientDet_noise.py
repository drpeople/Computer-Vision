import os
import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import torch
from effdet import create_model  # EfficientDet model creator
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# -------------------------------
# File paths and Initialization
# -------------------------------
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations
coco = COCO(annotation_path)

# Load EfficientDet model (EfficientDet-D0 in prediction mode)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model('tf_efficientdet_d0', pretrained=True, bench_task='predict')
model.to(device)
model.eval()

# -------------------------------
# Helper Functions
# -------------------------------
def get_class_images(coco, images_path, num_samples=5):
    """Select a few images per class from the dataset."""
    selected_images = {}
    for category_id in coco.getCatIds():
        image_ids = coco.getImgIds(catIds=category_id)
        image_ids.sort()
        sampled_ids = image_ids[:num_samples]
        selected_images[category_id] = [coco.loadImgs(img_id)[0]['file_name'] for img_id in sampled_ids]
    return selected_images

# Get selected images
selected_images = get_class_images(coco, val_images_path)

def add_noise(image_path, noise_std):
    """
    Load an image and add Gaussian noise in the normalized [0,1] domain.
    The image is first normalized to [0,1], noise is added, then the image is clipped
    and scaled back to [0,255]. This follows the same approach as in the second script.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image {image_path} could not be loaded.")
    # Convert image to float32 and normalize to [0,1]
    image_norm = image.astype(np.float32) / 255.0
    # Generate Gaussian noise (mean=0, std=noise_std) in the normalized domain
    noise = np.random.normal(loc=0.0, scale=noise_std, size=image_norm.shape).astype(np.float32)
    noisy_norm = np.clip(image_norm + noise, 0.0, 1.0)
    # Scale back to [0,255] and convert to uint8
    noisy_image = (noisy_norm * 255).astype(np.uint8)
    return noisy_image

def preprocess_image_for_effdet(image, model_size):
    """
    Preprocess the image for EfficientDet:
      - Convert from BGR to RGB.
      - Resize to (model_size, model_size).
      - Scale pixel values to [0,1] and normalize using ImageNet statistics.
      - Return the tensor and scale factors (scale_x, scale_y) to map the predicted
        boxes back to the original image coordinates.
    """
    orig_h, orig_w = image.shape[:2]
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to the model’s input size
    resized = cv2.resize(image_rgb, (model_size, model_size))
    # Normalize pixel values to [0,1]
    resized = resized.astype(np.float32) / 255.0
    # Convert to tensor and re-order dimensions: HWC -> CHW, add batch dim
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
    # Normalize using ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    # Calculate scale factors to map boxes from model coordinates to original image
    scale_x = orig_w / model_size
    scale_y = orig_h / model_size
    return tensor, (scale_x, scale_y)

def run_inference(model, selected_images, images_path, noise_std=0, conf_threshold=0.25, model_size=512, device=device):
    """
    Run inference on images with added Gaussian noise.
    The image is preprocessed to the model input size, and predicted boxes are scaled
    back to the original image coordinates.
    """
    predictions = {}
    for category_id, image_files in selected_images.items():
        for image_file in image_files:
            image_path_full = os.path.join(images_path, image_file)
            # Apply noise if specified; otherwise, read the image normally
            if noise_std > 0:
                image = add_noise(image_path_full, noise_std)
            else:
                image = cv2.imread(image_path_full)
            if image is None:
                raise ValueError(f"Image {image_path_full} could not be loaded.")

            # Preprocess image for EfficientDet
            input_tensor, (scale_x, scale_y) = preprocess_image_for_effdet(image, model_size)
            input_tensor = input_tensor.to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(input_tensor)[0]
            outputs = outputs.cpu().numpy()

            if outputs.shape[0] > 0:
                # EfficientDet returns boxes in the model input coordinate system:
                # [x1, y1, x2, y2, score, class]
                boxes = outputs[:, :4].copy()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                scores = outputs[:, 4]
                labels = outputs[:, 5]
                valid = scores >= conf_threshold
                filtered_boxes = np.concatenate([boxes[valid],
                                                 scores[valid][:, None],
                                                 labels[valid][:, None]], axis=1)
            else:
                filtered_boxes = np.empty((0, 6))
            predictions[image_file] = filtered_boxes
    return predictions

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) for two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0

def calculate_metrics(coco, predictions, iou_threshold=0.5):
    """
    Calculate detection metrics based on predictions and ground-truth.
    Returns Precision, Recall, F1 Score, Mean IoU and mAP.
    """
    y_true = []
    y_pred = []
    iou_scores = []
    ap_per_class = {}

    for image_file, preds in predictions.items():
        # Retrieve image ID by file name
        image_id = [img['id'] for img in coco.dataset['images'] if img['file_name'] == image_file][0]
        # Load ground truth annotations for this image
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        true_boxes = [[x, y, x + w, y + h] for x, y, w, h in [ann['bbox'] for ann in anns]]
        pred_boxes = preds[:, :4] if preds.size > 0 else []
        pred_scores = preds[:, 4] if preds.size > 0 else []

        matched_gt = set()
        for pb, score in zip(pred_boxes, pred_scores):
            max_iou = 0
            best_match = -1
            for i, tb in enumerate(true_boxes):
                iou = compute_iou(tb, pb)
                if iou > max_iou:
                    max_iou = iou
                    best_match = i
            if max_iou >= iou_threshold and best_match not in matched_gt:
                matched_gt.add(best_match)
                y_true.append(1)
                y_pred.append(1)
                iou_scores.append(max_iou)
            else:
                y_true.append(0)
                y_pred.append(1)
        # Add false negatives for ground-truth boxes not matched
        for i, tb in enumerate(true_boxes):
            if i not in matched_gt:
                y_true.append(1)
                y_pred.append(0)

    # Compute per-class average precision (AP) for each category (optional)
    for category_id in coco.getCatIds():
        class_name = coco.loadCats([category_id])[0]['name']
        category_trues = [
            t for img_file, t in zip(predictions.keys(), y_true)
            if class_name in [coco.loadCats([ann['category_id']])[0]['name']
                              for ann in coco.loadAnns(coco.getAnnIds(imgIds=[
                                  [img['id'] for img in coco.dataset['images']
                                   if img['file_name'] == img_file][0]
                              ]))]
        ]
        category_preds = [
            p for img_file, p in zip(predictions.keys(), y_pred)
            if class_name in [coco.loadCats([ann['category_id']])[0]['name']
                              for ann in coco.loadAnns(coco.getAnnIds(imgIds=[
                                  [img['id'] for img in coco.dataset['images']
                                   if img['file_name'] == img_file][0]
                              ]))]
        ]
        ap_per_class[category_id] = precision_score(category_trues, category_preds) if category_preds else 0

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mean_ap = np.mean(list(ap_per_class.values()))
    mean_iou = np.mean(iou_scores) if iou_scores else 0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Mean IoU': mean_iou,
        'mAP': mean_ap,
    }

# -------------------------------
# Main Inference and Evaluation
# -------------------------------
# Define the noise levels to test (normalized noise values, e.g., 0.0 to 0.2)

noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
results = []

for noise_std in noise_levels:
    print(f"\nInference with noise_std {noise_std}")
    predictions = run_inference(model, selected_images, val_images_path, noise_std=noise_std, conf_threshold=0.25, model_size=512, device=device)
    metrics = calculate_metrics(coco, predictions)
    metrics['Noise Std'] = noise_std
    results.append(metrics)
    for metric, value in metrics.items():
        if metric != 'Noise Std':
            print(f"{metric}: {value:.4f}")

# Save results to a CSV file
results_df = pd.DataFrame(results)
csv_path = "EfficientDet_noise_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# -------------------------------
# Plotting results
# -------------------------------
metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU', 'mAP']
for metric in metrics_to_plot:
    plt.figure(figsize=(8, 6))
    plt.plot(results_df['Noise Std'], results_df[metric], marker='o', label=metric)
    plt.xlabel('Noise Std', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'{metric} vs. Noise Std', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
