import os
import random
import json
from pathlib import Path
from pycocotools.coco import COCO
from ultralytics import YOLO
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# File paths
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations
coco = COCO(annotation_path)

# Load YOLOv8 model
model = YOLO('yolov8l.pt')  # Replace 'yolov8n.pt' with your specific model file

# Function to get two consistent images per class
# num of samples is how many images i use from each lass
def get_class_images(coco, images_path, num_samples=5):
    selected_images = {}

    for category_id in coco.getCatIds():
        image_ids = coco.getImgIds(catIds=category_id)
        image_ids.sort()  # Ensure consistency across runs
        sampled_ids = image_ids[:num_samples]
        selected_images[category_id] = [coco.loadImgs(img_id)[0]['file_name'] for img_id in sampled_ids]

    return selected_images

# Get selected images
selected_images = get_class_images(coco, val_images_path)

# Run inference and collect predictions
def run_inference(model, selected_images, images_path, conf_threshold=0.25):
    predictions = {}
    for category_id, image_files in selected_images.items():
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)
            # Run model inference with adjusted confidence threshold
            results = model(image_path, conf=conf_threshold)
            predictions[image_file] = results[0].boxes.data.cpu().numpy() if results[0].boxes else np.empty((0, 6))
    return predictions

# Collect predictions
predictions = run_inference(model, selected_images, val_images_path)

# Helper function to compute IoU
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    return intersection / union if union > 0 else 0

# Calculate performance metrics
def calculate_metrics(coco, predictions, images_path, iou_threshold=0.5):
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
                y_pred.append(1)
                y_true.append(0)

        for i, tb in enumerate(true_boxes):
            if i not in matched_gt:
                y_true.append(1)
                y_pred.append(0)

    for category_id in coco.getCatIds():
        class_name = coco.loadCats([category_id])[0]['name']
        category_trues = [
            t for img_file, t in zip(predictions.keys(), y_true)
            if class_name in [coco.loadCats([ann['category_id']])[0]['name'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=[
                [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]
            ]))]
        ]
        category_preds = [
            p for img_file, p in zip(predictions.keys(), y_pred)
            if class_name in [coco.loadCats([ann['category_id']])[0]['name'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=[
                [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]
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


# Calculate metrics
metrics = calculate_metrics(coco, predictions, val_images_path)

# Print metrics
print("Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")


