import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from torchvision.models.detection import retinanet_resnet50_fpn
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# -------------------------------
# File paths and Initialization
# -------------------------------
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations and RetinaNet model
coco = COCO(annotation_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = retinanet_resnet50_fpn(pretrained=True)
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


# Define transformation for RetinaNet
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])


def run_inference(model, selected_images, images_path, scale_factor=1.0, conf_threshold=0.45):
    """
    Run inference on images scaled by the given factor.
    After detection, the bounding boxes are rescaled back to the original image dimensions.
    """
    predictions = {}
    for category_id, image_files in selected_images.items():
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)

            # Scale image if needed
            if scale_factor != 1.0:
                image = scale_image(image_path, scale_factor)
            else:
                image = cv2.imread(image_path)

            # Convert image to RGB and apply transform
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image_rgb).to(device).unsqueeze(0)

            # Run model inference with RetinaNet
            with torch.no_grad():
                outputs = model(image_tensor)

            # Extract predictions from outputs
            pred_boxes = outputs[0]['boxes'].cpu().numpy()
            pred_scores = outputs[0]['scores'].cpu().numpy()
            pred_labels = outputs[0]['labels'].cpu().numpy()

            # Apply confidence threshold
            valid_indices = pred_scores > conf_threshold
            pred_boxes = pred_boxes[valid_indices]
            pred_scores = pred_scores[valid_indices]
            pred_labels = pred_labels[valid_indices]

            # Rescale bounding boxes back to original image dimensions
            pred_boxes /= scale_factor

            # Combine boxes, scores, and labels into a single array
            if pred_boxes.size > 0:
                boxes = np.column_stack((pred_boxes, pred_scores, pred_labels))
            else:
                boxes = np.empty((0, 6))

            predictions[image_file] = boxes
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

        # Load ground truth annotations and boxes
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

    # Compute per-class AP (optional)
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

# Define the scale factors (e.g., 50%, 75%, 100%, 125%, 150%, etc.)
scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]
results = []

for scale in scale_factors:
    print(f"\nInference with scale factor {scale}")
    predictions = run_inference(model, selected_images, val_images_path, scale_factor=scale)
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
csv_path = "RetinaNet_scale_results.csv"
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
