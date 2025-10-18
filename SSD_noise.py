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

# Define the transformation (the model expects images as tensors normalized between [0,1])
transform = T.Compose([
    T.ToTensor(),  # converts to tensor and scales pixel values to [0,1]
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


# Function to add Gaussian noise to an image using normalized noise values
def add_noise(image_path, noise_std):
    """
    Load an image, convert to RGB, add Gaussian noise in the normalized [0,1] domain,
    and return the noisy image.
    """
    # Load image and convert from BGR to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to float32 in [0, 1]
    image_float = image.astype(np.float32) / 255.0

    # Generate Gaussian noise with standard deviation noise_std
    noise = np.random.normal(loc=0.0, scale=noise_std, size=image_float.shape)

    # Add noise and clip to [0, 1]
    noisy_image_float = np.clip(image_float + noise, 0, 1)

    # Convert back to 8-bit [0, 255]
    noisy_image = (noisy_image_float * 255).astype(np.uint8)
    return noisy_image


# Run inference and collect predictions using SSD300_VGG16
def run_inference(model, selected_images, images_path, noise_std=0.0, conf_threshold=0.25):
    predictions = {}
    with torch.no_grad():
        for category_id, image_files in selected_images.items():
            for image_file in image_files:
                image_path = os.path.join(images_path, image_file)
                # Read the image and convert from BGR to RGB
                if noise_std > 0:
                    image_np = add_noise(image_path, noise_std)
                else:
                    image_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

                # Transform and prepare image tensor
                image_tensor = transform(image_np).to(device)
                image_tensor = image_tensor.unsqueeze(0)  # add batch dimension

                # Run model inference
                outputs = model(image_tensor)[0]
                boxes = outputs['boxes'].cpu().numpy() if outputs['boxes'].numel() > 0 else np.empty((0, 4))
                scores = outputs['scores'].cpu().numpy() if outputs['scores'].numel() > 0 else np.empty((0,))
                labels = outputs['labels'].cpu().numpy() if outputs['labels'].numel() > 0 else np.empty((0,))

                # Filter detections by confidence threshold
                keep = scores >= conf_threshold
                filtered_boxes = boxes[keep]
                filtered_scores = scores[keep]
                filtered_labels = labels[keep]

                # Format detections as: [x1, y1, x2, y2, confidence, class_id]
                if filtered_boxes.size > 0:
                    dets = np.hstack([filtered_boxes,
                                      filtered_scores.reshape(-1, 1),
                                      filtered_labels.reshape(-1, 1)])
                else:
                    dets = np.empty((0, 6))
                predictions[image_file] = dets
    return predictions


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


# Perform inference for the original and noisy images using SSD300_VGG16
# Define noise levels as normalized standard deviations (e.g., 0.0, 0.05, 0.1, 0.15, 0.2)
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
results = []

for noise_std in noise_levels:
    print(f"\nInference with noise_std {noise_std}")
    predictions = run_inference(model, selected_images, val_images_path, noise_std=noise_std)
    metrics = calculate_metrics(coco, predictions)
    metrics['Noise Std'] = noise_std
    results.append(metrics)
    for metric, value in metrics.items():
        if metric != 'Noise Std':
            print(f"{metric}: {value:.4f}")

# Save results to a CSV file
results_df = pd.DataFrame(results)
csv_path = "SSD300_VGG16_noise_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# Plotting results
metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU', 'mAP']
for metric in metrics_to_plot:
    plt.figure(figsize=(8, 6))
    plt.plot(results_df['Noise Std'], results_df[metric], marker='o', label=metric)
    plt.xlabel('Noise Std (normalized)', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.title(f'{metric} vs. Noise Std', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
