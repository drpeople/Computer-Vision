# import os
# import cv2
# import numpy as np
# import pandas as pd
# import torch
# from PIL import Image
# import torchvision.transforms as T
# from pycocotools.coco import COCO
# from sklearn.metrics import precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt
# from torchvision.models.detection import (
#     retinanet_resnet50_fpn_v2,
#     RetinaNet_ResNet50_FPN_V2_Weights,
# )
#
# # -------------------------------
# # File paths and Initialization
# # -------------------------------
# val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
# annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"
#
# # Load COCO annotations and initialize RetinaNet‑ResNet50‑FPN‑v2 model
# coco = COCO(annotation_path)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Use official COCO‑trained weights (v2) and their pre‑processing pipeline
# weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
# model = retinanet_resnet50_fpn_v2(weights=weights)
# model.to(device).eval()
#
# # Model‑specific transforms (resize, tensor, normalize, …)
# transform = weights.transforms()
#
# # -------------------------------
# # Helper Functions
# # -------------------------------
#
# def get_class_images(coco, images_path, num_samples=5):
#     """Select a few images per class from the dataset."""
#     selected_images = {}
#     for category_id in coco.getCatIds():
#         image_ids = coco.getImgIds(catIds=category_id)
#         image_ids.sort()
#         sampled_ids = image_ids[:num_samples]
#         selected_images[category_id] = [
#             coco.loadImgs(img_id)[0]["file_name"] for img_id in sampled_ids
#         ]
#     return selected_images
#
#
# def add_noise(image_path, noise_std):
#     """Load an image, add Gaussian noise in [0,1] domain, return noisy uint8 image."""
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_float = image.astype(np.float32) / 255.0
#     noise = np.random.normal(loc=0.0, scale=noise_std, size=image_float.shape)
#     noisy_image_float = np.clip(image_float + noise, 0, 1)
#     noisy_image = (noisy_image_float * 255).astype(np.uint8)
#     return noisy_image
#
#
# def run_inference(model, selected_images, images_path, noise_std=0.0, conf_threshold=0.45):
#     """Run inference on (possibly noisy) images and return predictions dict."""
#     predictions = {}
#     for category_id, image_files in selected_images.items():
#         for image_file in image_files:
#             image_path = os.path.join(images_path, image_file)
#             if noise_std > 0:
#                 image = add_noise(image_path, noise_std)
#             else:
#                 image = cv2.imread(image_path)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#             # Preprocess & move to device
#             pil_img = Image.fromarray(image)
#             image_tensor = transform(pil_img).unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 outputs = model(image_tensor)
#
#             pred_boxes = outputs[0]["boxes"].cpu().numpy()
#             pred_scores = outputs[0]["scores"].cpu().numpy()
#             pred_labels = outputs[0]["labels"].cpu().numpy()
#
#             valid = pred_scores > conf_threshold
#             pred_boxes = pred_boxes[valid]
#             pred_scores = pred_scores[valid]
#             pred_labels = pred_labels[valid]
#
#             pred_data = (
#                 np.column_stack((pred_boxes, pred_scores, pred_labels))
#                 if pred_boxes.size > 0
#                 else np.empty((0, 6))
#             )
#             predictions[image_file] = pred_data
#     return predictions
#
#
# def compute_iou(box1, box2):
#     """Compute Intersection over Union (IoU) for two boxes."""
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
#     intersection = max(0, x2 - x1) * max(0, y2 - y1)
#     area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union = area_box1 + area_box2 - intersection
#     return intersection / union if union > 0 else 0
#
#
# def calculate_metrics(coco, predictions, iou_threshold=0.5):
#     """Calculate detection metrics and return a dict of results."""
#     y_true, y_pred, iou_scores = [], [], []
#     ap_per_class = {}
#
#     for image_file, preds in predictions.items():
#         image_id = [img["id"] for img in coco.dataset["images"] if img["file_name"] == image_file][0]
#         ann_ids = coco.getAnnIds(imgIds=image_id)
#         anns = coco.loadAnns(ann_ids)
#         true_boxes = [[x, y, x + w, y + h] for x, y, w, h in [ann["bbox"] for ann in anns]]
#
#         pred_boxes = preds[:, :4] if preds.size > 0 else []
#         pred_scores = preds[:, 4] if preds.size > 0 else []
#
#         matched_gt = set()
#         for pb, score in zip(pred_boxes, pred_scores):
#             max_iou, best_match = 0, -1
#             for i, tb in enumerate(true_boxes):
#                 iou = compute_iou(tb, pb)
#                 if iou > max_iou:
#                     max_iou, best_match = iou, i
#             if max_iou >= iou_threshold and best_match not in matched_gt:
#                 matched_gt.add(best_match)
#                 y_true.append(1)
#                 y_pred.append(1)
#                 iou_scores.append(max_iou)
#             else:
#                 y_true.append(0)
#                 y_pred.append(1)
#
#         for i, _ in enumerate(true_boxes):
#             if i not in matched_gt:
#                 y_true.append(1)
#                 y_pred.append(0)
#
#     for category_id in coco.getCatIds():
#         class_name = coco.loadCats([category_id])[0]["name"]
#         category_trues = [t for img_file, t in zip(predictions.keys(), y_true) if class_name in [coco.loadCats([ann["category_id"]])[0]["name"] for ann in coco.loadAnns(coco.getAnnIds(imgIds=[[img["id"] for img in coco.dataset["images"] if img["file_name"] == img_file][0]]))]]
#         category_preds = [p for img_file, p in zip(predictions.keys(), y_pred) if class_name in [coco.loadCats([ann["category_id"]])[0]["name"] for ann in coco.loadAnns(coco.getAnnIds(imgIds=[[img["id"] for img in coco.dataset["images"] if img["file_name"] == img_file][0]]))]]
#         ap_per_class[category_id] = precision_score(category_trues, category_preds) if category_preds else 0
#
#     precision = precision_score(y_true, y_pred, zero_division=0)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     mean_ap = np.mean(list(ap_per_class.values()))
#     mean_iou = np.mean(iou_scores) if iou_scores else 0
#
#     return {"Precision": precision, "Recall": recall, "F1 Score": f1, "Mean IoU": mean_iou, "mAP": mean_ap}
#
#
# # -------------------------------
# # Main Inference and Evaluation
# # -------------------------------
# selected_images = get_class_images(coco, val_images_path)
#
# noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# results = []
#
# for noise_std in noise_levels:
#     print(f"\nInference with noise_std {noise_std}")
#     predictions = run_inference(model, selected_images, val_images_path, noise_std=noise_std, conf_threshold=0.45)
#     metrics = calculate_metrics(coco, predictions)
#     metrics["Noise Std"] = noise_std
#     results.append(metrics)
#     for metric, value in metrics.items():
#         if metric != "Noise Std":
#             print(f"{metric}: {value:.4f}")
#
# # Save metrics to CSV
# results_df = pd.DataFrame(results)
# csv_path = "RetinaNet_v2_noise_results.csv"
# results_df.to_csv(csv_path, index=False)
# print(f"\nResults saved to {csv_path}")
#
# # -------------------------------
# # Plot metrics vs. noise
# # -------------------------------
# metrics_to_plot = ["Precision", "Recall", "F1 Score", "Mean IoU", "mAP"]
# for metric in metrics_to_plot:
#     plt.figure(figsize=(8, 6))
#     plt.plot(results_df["Noise Std"], results_df[metric], marker="o", label=metric)
#     plt.xlabel("Noise Std (normalized)", fontsize=12)
#     plt.ylabel(metric, fontsize=12)
#     plt.title(f"{metric} vs. Noise Std", fontsize=14)
#     plt.grid(True)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.show()


import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# -------------------------------
# File paths and Initialization
# -------------------------------
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations and initialize RetinaNet model
coco = COCO(annotation_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = retinanet_resnet50_fpn_v2(pretrained=True)
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
        selected_images[category_id] = [
            coco.loadImgs(img_id)[0]['file_name'] for img_id in sampled_ids
        ]
    return selected_images


# Get selected images
selected_images = get_class_images(coco, val_images_path)


def add_noise(image_path, noise_std):
    """
    Load an image, convert to RGB, add Gaussian noise in the normalized [0,1] domain,
    and return the noisy image.
    """
    # Load and convert image to RGB
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


# Define transformation for RetinaNet
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
])


def run_inference(model, selected_images, images_path, noise_std=0.0, conf_threshold=0.45):
    """
    Run inference on (possibly noisy) images.
    For each image, if noise_std > 0, add noise;
    otherwise, load the image normally (converted to RGB).
    """
    predictions = {}
    for category_id, image_files in selected_images.items():
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)
            if noise_std > 0:
                image = add_noise(image_path, noise_std)
            else:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Preprocess image for RetinaNet
            image_tensor = transform(image).to(device).unsqueeze(0)

            # Run inference using RetinaNet
            with torch.no_grad():
                outputs = model(image_tensor)

            # Extract predictions: boxes, scores, labels
            pred_boxes = outputs[0]['boxes'].cpu().numpy()
            pred_scores = outputs[0]['scores'].cpu().numpy()
            pred_labels = outputs[0]['labels'].cpu().numpy()

            # Apply confidence threshold
            valid = pred_scores > conf_threshold
            pred_boxes = pred_boxes[valid]
            pred_scores = pred_scores[valid]
            pred_labels = pred_labels[valid]

            # Combine predictions into an array: [x1, y1, x2, y2, score, label]
            if pred_boxes.size > 0:
                pred_data = np.column_stack((pred_boxes, pred_scores, pred_labels))
            else:
                pred_data = np.empty((0, 6))
            predictions[image_file] = pred_data
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
    Calculate detection metrics based on predictions and COCO ground-truth.
    Returns Precision, Recall, F1 Score, Mean IoU and mAP.
    """
    y_true = []
    y_pred = []
    iou_scores = []
    ap_per_class = {}

    for image_file, preds in predictions.items():
        # Retrieve image ID using file name
        image_id = [img['id'] for img in coco.dataset['images'] if img['file_name'] == image_file][0]
        # Ground truth annotations and boxes
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        true_boxes = [
            [x, y, x + w, y + h] for x, y, w, h in [ann['bbox'] for ann in anns]
        ]
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

        # Add false negatives for any ground truth boxes not matched
        for i, tb in enumerate(true_boxes):
            if i not in matched_gt:
                y_true.append(1)
                y_pred.append(0)

    # Compute per-class AP (optional)
    for category_id in coco.getCatIds():
        class_name = coco.loadCats([category_id])[0]['name']
        category_trues = [
            t for img_file, t in zip(predictions.keys(), y_true)
            if class_name in (
                [coco.loadCats([ann['category_id']])[0]['name']
                 for ann in coco.loadAnns(
                    coco.getAnnIds(imgIds=[
                        [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]
                    ])
                )]
            )
        ]
        category_preds = [
            p for img_file, p in zip(predictions.keys(), y_pred)
            if class_name in (
                [coco.loadCats([ann['category_id']])[0]['name']
                 for ann in coco.loadAnns(
                    coco.getAnnIds(imgIds=[
                        [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]
                    ])
                )]
            )
        ]
        ap_per_class[category_id] = (
            precision_score(category_trues, category_preds) if category_preds else 0
        )

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

# Define noise levels (now in normalized domain, similar to the second script)
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
results = []

for noise_std in noise_levels:
    print(f"\nInference with noise_std {noise_std}")
    predictions = run_inference(model, selected_images, val_images_path, noise_std=noise_std, conf_threshold=0.45)
    metrics = calculate_metrics(coco, predictions)
    metrics['Noise Std'] = noise_std
    results.append(metrics)
    for metric, value in metrics.items():
        if metric != 'Noise Std':
            print(f"{metric}: {value:.4f}")

# Save results to a CSV file
results_df = pd.DataFrame(results)
csv_path = "RetinaNetv2_noise_results.csv"
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
