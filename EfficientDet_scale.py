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


def scale_image(image_path, scale_factor):
    """Load an image and scale it by the given factor."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image {image_path} could not be loaded.")
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_image


def preprocess_image_for_effdet(image, model_size):
    """
    Preprocess the image for EfficientDet:
      - Convert from BGR to RGB.
      - Resize to (model_size, model_size).
      - Scale pixel values to [0,1] and normalize using ImageNet statistics.
      - Return the tensor and scale factors (scale_x, scale_y) to map
        from model input coordinates back to the scaled image coordinates.
    """
    # Get dimensions of the scaled image
    h, w = image.shape[:2]
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to the model’s input size
    resized_image = cv2.resize(image_rgb, (model_size, model_size))
    # Scale pixel values to [0, 1]
    resized_image = resized_image.astype(np.float32) / 255.0
    # Convert to tensor and re-order dimensions: HWC -> CHW, add batch dim
    tensor = torch.from_numpy(resized_image).permute(2, 0, 1).unsqueeze(0)
    # Normalize using ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    # Compute scale factors to map boxes from model input coordinates back to scaled image coordinates
    scale_x = w / model_size
    scale_y = h / model_size
    return tensor, (scale_x, scale_y)


def run_inference(model, selected_images, images_path, scale_factor=1.0, conf_threshold=0.25, model_size=512, device=device):
    """
    Run inference on images scaled by the given factor.
    After detection, bounding boxes are mapped from model output coordinates to the original image.
    """
    predictions = {}
    for category_id, image_files in selected_images.items():
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)
            # Scale image if needed
            if scale_factor != 1.0:
                scaled_image = scale_image(image_path, scale_factor)
            else:
                scaled_image = cv2.imread(image_path)
            if scaled_image is None:
                raise ValueError(f"Image {image_path} could not be loaded.")

            # Preprocess the scaled image for EfficientDet
            input_tensor, (scale_x, scale_y) = preprocess_image_for_effdet(scaled_image, model_size)
            input_tensor = input_tensor.to(device)

            # Run model inference
            with torch.no_grad():
                outputs = model(input_tensor)[0]
            outputs = outputs.cpu().numpy()

            if outputs.shape[0] > 0:
                # The model outputs boxes in the model input coordinate system:
                # [x1, y1, x2, y2, score, class]
                boxes = outputs[:, :4].copy()
                # Map boxes to scaled image coordinates
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                # Map boxes to original image coordinates
                boxes[:, :4] = boxes[:, :4] / scale_factor
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
        # Get the image id using file name
        image_id = [img['id'] for img in coco.dataset['images'] if img['file_name'] == image_file][0]
        # Ground truth annotations and boxes
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
                    [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]
                ]))]
        ]
        category_preds = [
            p for img_file, p in zip(predictions.keys(), y_pred)
            if class_name in [coco.loadCats([ann['category_id']])[0]['name']
                              for ann in coco.loadAnns(coco.getAnnIds(imgIds=[
                    [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]
                ]))]
        ]
        ap_per_class[category_id] = precision_score(category_trues, category_preds) if category_preds else 0

    precision = precision_score(y_true, y_pred, zero_division=0)
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
# Get a subset of images per class
selected_images = get_class_images(coco, val_images_path)

# Define the scale factors (e.g., 50%, 75%, 100%, 125%, 150%, 200%, 300%)
scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]
results = []

for scale in scale_factors:
    print(f"\nInference with scale factor {scale}")
    predictions = run_inference(model, selected_images, val_images_path, scale_factor=scale, conf_threshold=0.25, model_size=512, device=device)
    metrics = calculate_metrics(coco, predictions)
    metrics['Scale Factor'] = scale
    results.append(metrics)
    # Print selected metrics for review
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    print(f"Mean IoU: {metrics['Mean IoU']:.4f}")
    print(f"mAP: {metrics['mAP']:.4f}")

# Save results to CSV
results_df = pd.DataFrame(results)
csv_path = "EfficientDet_scale_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# -------------------------------
# Plotting: Recall, F1 Score, Mean IoU and mAP
# -------------------------------
metrics_to_plot = ['Recall', 'F1 Score', 'Mean IoU', 'mAP']
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, metric in enumerate(metrics_to_plot):
    axs[i].plot(results_df['Scale Factor'], results_df[metric], marker='o', label=metric)
    axs[i].set_xlabel('Scale Factor', fontsize=12)
    axs[i].set_ylabel(metric, fontsize=12)
    axs[i].set_title(f'{metric} vs. Scale Factor', fontsize=14)
    axs[i].grid(True)
    axs[i].legend(fontsize=12)

plt.tight_layout()
plt.show()
