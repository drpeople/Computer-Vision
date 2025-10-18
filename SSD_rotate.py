import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.models.detection import ssd300_vgg16

# -------------------------------
# File paths and Initialization
# -------------------------------
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations
coco = COCO(annotation_path)

# Load SSD300_VGG16 model (pretrained on COCO) and set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ssd300_vgg16(pretrained=True).to(device)
model.eval()

# Define transformation: convert images (in RGB) to tensor (values in [0,1])
transform = T.Compose([
    T.ToTensor(),
])

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

def rotate_image_and_labels(image_path, annotations, angle):
    """
    Rotate the image and adjust the ground-truth bounding boxes.
    Returns the rotated image and updated annotations.
    """
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix for the given angle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # Compute new dimensions of the rotated image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Rotate the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

    # Adjust bounding boxes
    rotated_annotations = []
    for ann in annotations:
        x, y, bw, bh = ann['bbox']
        # Define box corners
        box = np.array([[x, y], [x + bw, y], [x, y + bh], [x + bw, y + bh]])
        # Append ones for homogeneous coordinates
        ones = np.ones((box.shape[0], 1))
        points = np.hstack([box, ones])
        # Rotate all points
        rotated_box = np.dot(rotation_matrix, points.T).T

        x_min, y_min = rotated_box[:, 0].min(), rotated_box[:, 1].min()
        x_max, y_max = rotated_box[:, 0].max(), rotated_box[:, 1].max()

        rotated_annotations.append({
            'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
            'category_id': ann['category_id']
        })

    return rotated_image, rotated_annotations

def run_inference(model, selected_images, images_path, angle=0, conf_threshold=0.2):
    """
    Run inference on rotated images.
    Returns a dictionary mapping each image file to predictions and ground truth.
    """
    predictions = {}
    with torch.no_grad():
        for category_id, image_files in selected_images.items():
            for image_file in image_files:
                image_path = os.path.join(images_path, image_file)

                # Get ground-truth annotations
                image_id = [img['id'] for img in coco.dataset['images'] if img['file_name'] == image_file][0]
                ann_ids = coco.getAnnIds(imgIds=image_id)
                anns = coco.loadAnns(ann_ids)

                # Rotate image and adjust annotations
                rotated_image, rotated_annotations = rotate_image_and_labels(image_path, anns, angle)

                # Convert rotated image from BGR to RGB and transform to tensor
                image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
                image_tensor = transform(image_rgb).to(device)
                image_tensor = image_tensor.unsqueeze(0)

                # Run model inference
                outputs = model(image_tensor)[0]
                if outputs['boxes'].numel() > 0:
                    boxes = outputs['boxes'].cpu().numpy()
                    scores = outputs['scores'].cpu().numpy()
                    labels = outputs['labels'].cpu().numpy()
                    # Filter by confidence threshold
                    keep = scores >= conf_threshold
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    # Format as: [x1, y1, x2, y2, confidence, class_id]
                    if boxes.size > 0:
                        dets = np.hstack([boxes, scores.reshape(-1, 1), labels.reshape(-1, 1)])
                    else:
                        dets = np.empty((0, 6))
                else:
                    dets = np.empty((0, 6))
                predictions[image_file] = {
                    'predictions': dets,
                    'ground_truth': rotated_annotations
                }
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
    Calculate detection metrics based on rotated predictions and ground-truth.
    Returns Precision, Recall, F1 Score, Mean IoU and mAP.
    """
    y_true = []
    y_pred = []
    iou_scores = []
    ap_per_class = {}

    for image_file, data in predictions.items():
        preds = data['predictions']
        anns = data['ground_truth']

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

    # Compute per-class AP (using precision as a rough approximation)
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
# Get a subset of images per class
selected_images = get_class_images(coco, val_images_path)

# Define the rotation angles (in degrees)
rotation_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
results = []

for angle in rotation_angles:
    print(f"\nInference with rotation angle {angle}")
    predictions = run_inference(model, selected_images, val_images_path, angle)
    metrics = calculate_metrics(coco, predictions)
    metrics['Rotation Angle'] = angle
    results.append(metrics)
    print(f"Rotation Angle: {angle}")
    for metric, value in metrics.items():
        if metric != 'Rotation Angle':
            print(f"{metric}: {value:.4f}")

# Save results to a CSV file
results_df = pd.DataFrame(results)
csv_path = "SSD300_VGG16_rotation_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# -------------------------------
# Plotting results
# -------------------------------
metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU', 'mAP']
for metric in metrics_to_plot:
    plt.figure(figsize=(8, 6))
    plt.plot(results_df['Rotation Angle'], results_df[metric], marker='o', label=metric)
    plt.xlabel('Rotation Angle (degrees)', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'{metric} vs. Rotation Angle', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
