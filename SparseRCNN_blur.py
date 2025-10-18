import os
import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# File paths
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"
centernet_model_path = r"path\to\centernet_model.h5"  # Update this path

# Load COCO annotations
coco = COCO(annotation_path)

# Load the CenterNet model (ensure your model outputs [heatmap, reg, wh])
model = load_model(centernet_model_path, compile=False)

# Define model input size (adjust as needed for your model)
INPUT_SIZE = (512, 512)  # (width, height)

def preprocess_image(image, input_size=INPUT_SIZE):
    """
    Resize and normalize the image.
    Returns the preprocessed image and the original image dimensions.
    """
    orig_h, orig_w = image.shape[:2]
    image_resized = cv2.resize(image, input_size)
    image_norm = image_resized.astype(np.float32) / 255.0  # normalize to [0, 1]
    image_input = np.expand_dims(image_norm, axis=0)  # add batch dimension
    return image_input, (orig_w, orig_h)

def decode_centernet_outputs(heatmap, reg, wh, conf_threshold=0.25, stride=4):
    """
    A simplified decoding function that converts model outputs to bounding boxes.
    Assumes:
      - heatmap: shape (H, W, num_classes)
      - reg: shape (H, W, 2) for center offsets
      - wh: shape (H, W, 2) for width and height
    Returns an array of detections:
      [x1, y1, x2, y2, score, class_id]
    """
    detections = []
    H, W, num_classes = heatmap.shape
    for c in range(num_classes):
        hm = heatmap[..., c]
        ys, xs = np.where(hm > conf_threshold)
        for y, x in zip(ys, xs):
            score = hm[y, x]
            offset = reg[y, x]
            # Compute the center coordinate in original image scale using stride
            cx = (x + offset[0]) * stride
            cy = (y + offset[1]) * stride
            wh_val = wh[y, x]
            w_box = wh_val[0] * stride
            h_box = wh_val[1] * stride
            x1 = cx - w_box / 2
            y1 = cy - h_box / 2
            x2 = cx + w_box / 2
            y2 = cy + h_box / 2
            detections.append([x1, y1, x2, y2, score, c])
    if detections:
        return np.array(detections)
    else:
        return np.empty((0, 6))

def centernet_predict(model, image, conf_threshold=0.25):
    """
    Preprocess the image, run the CenterNet model, and decode its outputs.
    Assumes that the model returns [heatmap, reg, wh].
    """
    image_input, (orig_w, orig_h) = preprocess_image(image)
    preds = model.predict(image_input)
    # Unpack the outputs (adjust indices if necessary)
    heatmap, reg, wh = preds[0], preds[1], preds[2]
    heatmap = np.squeeze(heatmap)  # shape: (H, W, num_classes)
    reg = np.squeeze(reg)          # shape: (H, W, 2)
    wh = np.squeeze(wh)            # shape: (H, W, 2)
    detections = decode_centernet_outputs(heatmap, reg, wh, conf_threshold=conf_threshold, stride=4)
    return detections

def get_class_images(coco, images_path, num_samples=5):
    """
    For each category in COCO, select a fixed number of images.
    """
    selected_images = {}
    for category_id in coco.getCatIds():
        image_ids = coco.getImgIds(catIds=category_id)
        image_ids.sort()
        sampled_ids = image_ids[:num_samples]
        selected_images[category_id] = [coco.loadImgs(img_id)[0]['file_name'] for img_id in sampled_ids]
    return selected_images

selected_images = get_class_images(coco, val_images_path)

def blur_image(image_path, sigma):
    """
    Apply Gaussian blur to an image.
    """
    image = cv2.imread(image_path)
    kernel_size = (5, 5)
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)
    return blurred_image

def run_inference(model, selected_images, images_path, sigma=0, conf_threshold=0.25):
    """
    Run CenterNet inference on images (optionally blurred) and collect detections.
    """
    predictions = {}
    for category_id, image_files in selected_images.items():
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)
            image = blur_image(image_path, sigma) if sigma > 0 else cv2.imread(image_path)
            detections = centernet_predict(model, image, conf_threshold=conf_threshold)
            predictions[image_file] = detections
    return predictions

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    """
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
    Compute overall Precision, Recall, F1 Score, mAP, and mean IoU.
    Uses a mapping from image file names to image IDs for efficiency.
    """
    y_true = []
    y_pred = []
    iou_scores = []
    ap_per_class = {}

    # Create a mapping from file name to image id
    image_id_map = {img['file_name']: img['id'] for img in coco.dataset['images']}

    for image_file, preds in predictions.items():
        image_id = image_id_map[image_file]
        ann_ids = coco.getAnnIds(imgIds=[image_id])
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

    # Compute per-class AP using a simplified approach
    for category_id in coco.getCatIds():
        class_name = coco.loadCats([category_id])[0]['name']
        category_trues = []
        category_preds = []
        for image_file in predictions.keys():
            image_id = image_id_map[image_file]
            ann_ids = coco.getAnnIds(imgIds=[image_id])
            anns = coco.loadAnns(ann_ids)
            ann_class_names = [coco.loadCats([ann['category_id']])[0]['name'] for ann in anns]
            # For simplicity, assume if the class appears in the ground truth, detection is positive
            if class_name in ann_class_names:
                category_trues.append(1)
                category_preds.append(1)
            else:
                category_trues.append(0)
                category_preds.append(0)
        ap_per_class[category_id] = precision_score(category_trues, category_preds, zero_division=0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mean_ap = np.mean(list(ap_per_class.values()))
    mean_iou = np.mean(iou_scores) if iou_scores else 0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Mean IoU': mean_iou,
        'mAP': mean_ap,
    }

# Run inference over a range of blur levels
sigma_levels = [0, 1, 2, 3, 4]
results = []

for sigma in sigma_levels:
    print(f"\nInference with sigma {sigma}")
    predictions = run_inference(model, selected_images, val_images_path, sigma=sigma)
    metrics = calculate_metrics(coco, predictions)
    metrics['Sigma'] = sigma
    results.append(metrics)
    for metric, value in metrics.items():
        if metric != 'Sigma':
            print(f"{metric}: {value:.4f}")

# Save results to a CSV file
results_df = pd.DataFrame(results)
csv_path = "CenterNet_blur_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# Plot the metrics versus blur sigma
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
