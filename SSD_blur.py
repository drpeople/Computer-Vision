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

# File paths
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations
coco = COCO(annotation_path)

# Load the SSD300_VGG16 model (pretrained on COCO) and set it to evaluation mode.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ssd300_vgg16(pretrained=True).to(device)
model.eval()

# Define transformation: converts images (in RGB) to tensor with values in [0,1]
transform = T.Compose([
    T.ToTensor(),
])


# Function to get consistent images per class
def get_class_images(coco, images_path, num_samples=5):
    selected_images = {}
    for category_id in coco.getCatIds():
        image_ids = coco.getImgIds(catIds=category_id)
        image_ids.sort()
        sampled_ids = image_ids[:num_samples]
        selected_images[category_id] = [coco.loadImgs(img_id)[0]['file_name'] for img_id in sampled_ids]
    return selected_images


# Get selected images
selected_images = get_class_images(coco, val_images_path)


# Function to apply Gaussian blur to an image
def blur_image(image_path, sigma):
    image = cv2.imread(image_path)
    # Use a fixed kernel size; kernel size must be positive odd integers.
    kernel_size = (5, 5)
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)
    return blurred_image


# Run inference and collect predictions using SSD300_VGG16
def run_inference(model, selected_images, images_path, sigma=0, conf_threshold=0.2):
    predictions = {}
    with torch.no_grad():
        for category_id, image_files in selected_images.items():
            for image_file in image_files:
                image_path = os.path.join(images_path, image_file)
                # If sigma > 0, apply Gaussian blur; otherwise, use the original image.
                if sigma > 0:
                    image_bgr = blur_image(image_path, sigma)
                else:
                    image_bgr = cv2.imread(image_path)
                # Convert from BGR to RGB
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # Transform the image to a tensor and add batch dimension
                image_tensor = transform(image_rgb).to(device)
                image_tensor = image_tensor.unsqueeze(0)

                # Run inference
                outputs = model(image_tensor)[0]
                boxes = outputs['boxes'].cpu().numpy() if outputs['boxes'].numel() > 0 else np.empty((0, 4))
                scores = outputs['scores'].cpu().numpy() if outputs['scores'].numel() > 0 else np.empty((0,))
                labels = outputs['labels'].cpu().numpy() if outputs['labels'].numel() > 0 else np.empty((0,))

                # Filter detections by confidence threshold and format as:
                # [x1, y1, x2, y2, confidence, class_id]
                keep = scores >= conf_threshold
                filtered_boxes = boxes[keep]
                filtered_scores = scores[keep]
                filtered_labels = labels[keep]
                if filtered_boxes.size > 0:
                    dets = np.hstack([
                        filtered_boxes,
                        filtered_scores.reshape(-1, 1),
                        filtered_labels.reshape(-1, 1)
                    ])
                else:
                    dets = np.empty((0, 6))
                predictions[image_file] = dets
    return predictions


# Helper function to compute IoU between two boxes
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


# Calculate performance metrics comparing predictions to ground truth annotations
def calculate_metrics(coco, predictions, iou_threshold=0.5):
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
        for i, tb in enumerate(true_boxes):
            if i not in matched_gt:
                y_true.append(1)
                y_pred.append(0)

    # Compute average precision per class based on COCO categories.
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


# Perform inference for the original and blurred images
sigma_levels = [0, 1, 2, 3, 4]
results = []

for sigma in sigma_levels:
    print(f"\nInference with sigma {sigma}")
    predictions = run_inference(model, selected_images, val_images_path, sigma)
    metrics = calculate_metrics(coco, predictions)
    metrics['Sigma'] = sigma
    results.append(metrics)
    for metric, value in metrics.items():
        if metric != 'Sigma':
            print(f"{metric}: {value:.4f}")

# Save results to a CSV file
results_df = pd.DataFrame(results)
csv_path = "SSD300_VGG16_blur_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# Plotting results
metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU', 'mAP']
for metric in metrics_to_plot:
    plt.figure(figsize=(8, 6))
    plt.plot(results_df['Sigma'], results_df[metric], marker='o', label=metric)
    plt.xlabel('Sigma (Blur Level)', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'{metric} vs. Sigma (Blur Level)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
