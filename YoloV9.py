### YOLOV9 blur ##
#

# import os
# import cv2
# import numpy as np
# import pandas as pd
# from pycocotools.coco import COCO
# from ultralytics import YOLO
# from sklearn.metrics import precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt
#
# # File paths
# val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
# annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"
#
# # Load COCO annotations
# coco = COCO(annotation_path)
#
# # Load YOLO9 model
# model = YOLO('yolov9c.pt')
#
#
#
# # Function to get consistent images per class
# def get_class_images(coco, images_path, num_samples=5):
#     selected_images = {}
#     for category_id in coco.getCatIds():
#         image_ids = coco.getImgIds(catIds=category_id)
#         image_ids.sort()
#         sampled_ids = image_ids[:num_samples]
#         selected_images[category_id] = [coco.loadImgs(img_id)[0]['file_name'] for img_id in sampled_ids]
#     return selected_images
#
# # Get selected images
# selected_images = get_class_images(coco, val_images_path)
#
# # Function to apply Gaussian blur to an image
# def blur_image(image_path, sigma):
#     image = cv2.imread(image_path)
#     kernel_size = (5, 5)  # Fixed kernel size
#     blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)
#     return blurred_image
#
# # Run inference and collect predictions
# def run_inference(model, selected_images, images_path, sigma=0, conf_threshold=0.25):
#     predictions = {}
#     for category_id, image_files in selected_images.items():
#         for image_file in image_files:
#             image_path = os.path.join(images_path, image_file)
#             if sigma > 0:
#                 image = blur_image(image_path, sigma)
#             else:
#                 image = cv2.imread(image_path)
#
#             # Run model inference for yolov9
#             results = model.predict(source=image, conf=conf_threshold, save=False, verbose=False)
#
#             predictions[image_file] = results[0].boxes.data.cpu().numpy() if results[0].boxes else np.empty((0, 6))
#     return predictions
#
# # Helper function to compute IoU
# def compute_iou(box1, box2):
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
#
#     intersection = max(0, x2 - x1) * max(0, y2 - y1)
#     area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union = area_box1 + area_box2 - intersection
#
#     return intersection / union if union > 0 else 0
#
# # Calculate performance metrics
# def calculate_metrics(coco, predictions, iou_threshold=0.5):
#     y_true = []
#     y_pred = []
#     iou_scores = []
#     ap_per_class = {}
#
#     for image_file, preds in predictions.items():
#         # Retrieve image ID by file name
#         image_id = [img['id'] for img in coco.dataset['images'] if img['file_name'] == image_file][0]
#
#         # Load ground truth annotations for this image
#         ann_ids = coco.getAnnIds(imgIds=image_id)
#         anns = coco.loadAnns(ann_ids)
#         true_boxes = [[x, y, x + w, y + h] for x, y, w, h in [ann['bbox'] for ann in anns]]
#         pred_boxes = preds[:, :4] if preds.size > 0 else []
#         pred_scores = preds[:, 4] if preds.size > 0 else []
#
#         matched_gt = set()
#         for pb, score in zip(pred_boxes, pred_scores):
#             max_iou = 0
#             best_match = -1
#             for i, tb in enumerate(true_boxes):
#                 iou = compute_iou(tb, pb)
#                 if iou > max_iou:
#                     max_iou = iou
#                     best_match = i
#
#             if max_iou >= iou_threshold and best_match not in matched_gt:
#                 matched_gt.add(best_match)
#                 y_true.append(1)
#                 y_pred.append(1)
#                 iou_scores.append(max_iou)
#             else:
#                 y_pred.append(1)
#                 y_true.append(0)
#
#         for i, tb in enumerate(true_boxes):
#             if i not in matched_gt:
#                 y_true.append(1)
#                 y_pred.append(0)
#
#     for category_id in coco.getCatIds():
#         class_name = coco.loadCats([category_id])[0]['name']
#         category_trues = [
#             t for img_file, t in zip(predictions.keys(), y_true)
#             if class_name in [coco.loadCats([ann['category_id']])[0]['name'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=[
#                 [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]
#             ]))]
#         ]
#         category_preds = [
#             p for img_file, p in zip(predictions.keys(), y_pred)
#             if class_name in [coco.loadCats([ann['category_id']])[0]['name'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=[
#                 [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]
#             ]))]
#         ]
#         ap_per_class[category_id] = precision_score(category_trues, category_preds) if category_preds else 0
#
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     mean_ap = np.mean(list(ap_per_class.values()))
#     mean_iou = np.mean(iou_scores) if iou_scores else 0
#
#     return {
#         'Precision': precision,
#         'Recall': recall,
#         'F1 Score': f1,
#         'Mean IoU': mean_iou,
#         'mAP': mean_ap,
#     }
#
# # Perform inference for the original and blurred images
# sigma_levels = [0, 1, 2, 3, 4]
# results = []
#
# for sigma in sigma_levels:
#     print(f"\nInference with sigma {sigma}")
#     predictions = run_inference(model, selected_images, val_images_path, sigma)
#     metrics = calculate_metrics(coco, predictions)
#     metrics['Sigma'] = sigma
#     results.append(metrics)
#     for metric, value in metrics.items():
#         if metric != 'Sigma':
#             print(f"{metric}: {value:.4f}")
#
# # Save results to a CSV file
# results_df = pd.DataFrame(results)
# csv_path = "Yolo9_blur_results.csv"
# results_df.to_csv(csv_path, index=False)
# print(f"\nResults saved to {csv_path}")
#
# # Plotting results
# metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU', 'mAP']
# for metric in metrics_to_plot:
#     plt.figure(figsize=(8, 6))
#     plt.plot(results_df['Sigma'], results_df[metric], marker='o', label=metric)
#     plt.xlabel('Sigma (Blur Level)', fontsize=12)
#     plt.ylabel(metric, fontsize=12)
#     plt.title(f'{metric} vs. Sigma (Blur Level)', fontsize=14)
#     plt.grid(True)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.show()
#
#

### YOLO9 noise ##
#
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from pycocotools.coco import COCO
# from ultralytics import YOLO
# from sklearn.metrics import precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt
#
# # File paths
# val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
# annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"
#
# # Load COCO annotations
# coco = COCO(annotation_path)
#
# # Load YOLO9 model
# model = YOLO('yolov9c.pt')
#
# # Function to get consistent images per class
# def get_class_images(coco, images_path, num_samples=5):
#     selected_images = {}
#     for category_id in coco.getCatIds():
#         image_ids = coco.getImgIds(catIds=category_id)
#         image_ids.sort()
#         sampled_ids = image_ids[:num_samples]
#         selected_images[category_id] = [coco.loadImgs(img_id)[0]['file_name'] for img_id in sampled_ids]
#     return selected_images
#
# # Get selected images
# selected_images = get_class_images(coco, val_images_path)
#
# # Function to add Gaussian noise in the normalized [0, 1] domain
# def add_noise(image_path, noise_std):
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # Convert to float32 and scale to [0, 1]
#     image = image.astype(np.float32) / 255.0
#     # Generate Gaussian noise with mean 0 and specified std
#     noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
#     # Add noise and clip to [0, 1]
#     noisy_image = np.clip(image + noise, 0, 1)
#     # Scale back to [0, 255] and convert to uint8
#     noisy_image = (noisy_image * 255).astype(np.uint8)
#     return noisy_image
#
# # Run inference and collect predictions
# def run_inference(model, selected_images, images_path, noise_std=0.0, conf_threshold=0.25):
#     predictions = {}
#     for category_id, image_files in selected_images.items():
#         for image_file in image_files:
#             image_path = os.path.join(images_path, image_file)
#             if noise_std > 0:
#                 image = add_noise(image_path, noise_std)
#             else:
#                 image = cv2.imread(image_path)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             # Run model inference
#             results = model.predict(source=image, conf=conf_threshold, save=False, verbose=False)
#             predictions[image_file] = results[0].boxes.data.cpu().numpy() if results[0].boxes else np.empty((0, 6))
#     return predictions
#
# # Helper function to compute IoU
# def compute_iou(box1, box2):
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
# # Calculate performance metrics
# def calculate_metrics(coco, predictions, iou_threshold=0.5):
#     y_true = []
#     y_pred = []
#     iou_scores = []
#     ap_per_class = {}
#
#     for image_file, preds in predictions.items():
#         # Retrieve image ID by file name
#         image_id = [img['id'] for img in coco.dataset['images'] if img['file_name'] == image_file][0]
#         ann_ids = coco.getAnnIds(imgIds=image_id)
#         anns = coco.loadAnns(ann_ids)
#         true_boxes = [[x, y, x + w, y + h] for x, y, w, h in [ann['bbox'] for ann in anns]]
#         pred_boxes = preds[:, :4] if preds.size > 0 else []
#         pred_scores = preds[:, 4] if preds.size > 0 else []
#
#         matched_gt = set()
#         for pb, score in zip(pred_boxes, pred_scores):
#             max_iou = 0
#             best_match = -1
#             for i, tb in enumerate(true_boxes):
#                 iou = compute_iou(tb, pb)
#                 if iou > max_iou:
#                     max_iou = iou
#                     best_match = i
#
#             if max_iou >= iou_threshold and best_match not in matched_gt:
#                 matched_gt.add(best_match)
#                 y_true.append(1)
#                 y_pred.append(1)
#                 iou_scores.append(max_iou)
#             else:
#                 y_pred.append(1)
#                 y_true.append(0)
#
#         for i, tb in enumerate(true_boxes):
#             if i not in matched_gt:
#                 y_true.append(1)
#                 y_pred.append(0)
#
#     for category_id in coco.getCatIds():
#         class_name = coco.loadCats([category_id])[0]['name']
#         category_trues = [
#             t for img_file, t in zip(predictions.keys(), y_true)
#             if class_name in [coco.loadCats([ann['category_id']])[0]['name'] for ann in coco.loadAnns(
#                 coco.getAnnIds(imgIds=[[img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]])
#             )]
#         ]
#         category_preds = [
#             p for img_file, p in zip(predictions.keys(), y_pred)
#             if class_name in [coco.loadCats([ann['category_id']])[0]['name'] for ann in coco.loadAnns(
#                 coco.getAnnIds(imgIds=[[img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]])
#             )]
#         ]
#         ap_per_class[category_id] = precision_score(category_trues, category_preds) if category_preds else 0
#
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     mean_ap = np.mean(list(ap_per_class.values()))
#     mean_iou = np.mean(iou_scores) if iou_scores else 0
#
#     return {
#         'Precision': precision,
#         'Recall': recall,
#         'F1 Score': f1,
#         'Mean IoU': mean_iou,
#         'mAP': mean_ap,
#     }
#
# # Perform inference for the original and noisy images
# # Noise levels updated to normalized values: 0.0, 0.05, 0.1, 0.2, 0.3
# noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# results = []
#
# for noise_std in noise_levels:
#     print(f"\nInference with noise std {noise_std}")
#     predictions = run_inference(model, selected_images, val_images_path, noise_std)
#     metrics = calculate_metrics(coco, predictions)
#     metrics['Noise Std'] = noise_std
#     results.append(metrics)
#     for metric, value in metrics.items():
#         if metric != 'Noise Std':
#             print(f"{metric}: {value:.4f}")
#
# # Save results to a CSV file
# results_df = pd.DataFrame(results)
# csv_path = "Yolo9_noise_results.csv"
# results_df.to_csv(csv_path, index=False)
# print(f"\nResults saved to {csv_path}")
#
# # Plotting results
# metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU', 'mAP']
# for metric in metrics_to_plot:
#     plt.figure(figsize=(8, 6))
#     plt.plot(results_df['Noise Std'], results_df[metric], marker='o', label=metric)
#     plt.xlabel('Noise Std', fontsize=12)
#     plt.ylabel(metric, fontsize=12)
#     plt.title(f'{metric} vs. Noise Std', fontsize=14)
#     plt.grid(True)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.show()


### YOLO9 scale ##

import os
import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# -------------------------------
# File paths and Initialization
# -------------------------------
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations and YOLO model
coco = COCO(annotation_path)
model = YOLO('yolov9c.pt')


#-------------------------------
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


def run_inference(model, selected_images, images_path, scale_factor=1.0, conf_threshold=0.25):
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

            # Run model inference on the (possibly scaled) image
            results = model.predict(source=image, conf=conf_threshold, save=False, verbose=False)
            if results[0].boxes:
                boxes = results[0].boxes.data.cpu().numpy()
                # Adjust boxes back to original coordinates
                boxes[:, :4] = boxes[:, :4] / scale_factor
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

    # Compute per-class average precision (AP) for each category (optional, based on class-wise filtering)
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

# Define the scale factors (original, 50%, 75%, 125% and 150%)
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
csv_path = "Yolo9_scale_results.csv"
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

### YOLO9 rotate ##

# import os
# import cv2
# import numpy as np
# import pandas as pd
# from pycocotools.coco import COCO
# from ultralytics import YOLO
# from sklearn.metrics import precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt
#
# # File paths
# val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
# annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"
#
# # Load COCO annotations
# coco = COCO(annotation_path)
#
# # Load YOLO9 model
# model = YOLO('yolov9c.pt')
#
# # Function to get consistent images per class
# def get_class_images(coco, images_path, num_samples=5):
#     selected_images = {}
#     for category_id in coco.getCatIds():
#         image_ids = coco.getImgIds(catIds=category_id)
#         image_ids.sort()
#         sampled_ids = image_ids[:num_samples]
#         selected_images[category_id] = [coco.loadImgs(img_id)[0]['file_name'] for img_id in sampled_ids]
#     return selected_images
#
# # Get selected images
# selected_images = get_class_images(coco, val_images_path)
#
# # Function to rotate an image and adjust bounding boxes
# def rotate_image_and_labels(image_path, annotations, angle):
#     image = cv2.imread(image_path)
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#
#     # Rotation matrix
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     cos = np.abs(rotation_matrix[0, 0])
#     sin = np.abs(rotation_matrix[0, 1])
#
#     # Compute new bounding dimensions
#     new_w = int((h * sin) + (w * cos))
#     new_h = int((h * cos) + (w * sin))
#
#     # Adjust the rotation matrix to the new dimensions
#     rotation_matrix[0, 2] += (new_w / 2) - center[0]
#     rotation_matrix[1, 2] += (new_h / 2) - center[1]
#
#     # Rotate the image
#     rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
#
#     # Adjust bounding boxes
#     rotated_annotations = []
#     for ann in annotations:
#         x, y, w, h = ann['bbox']
#         box = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
#         ones = np.ones((box.shape[0], 1))
#         points = np.hstack([box, ones])
#         rotated_box = np.dot(rotation_matrix, points.T).T
#
#         x_min, y_min = rotated_box[:, 0].min(), rotated_box[:, 1].min()
#         x_max, y_max = rotated_box[:, 0].max(), rotated_box[:, 1].max()
#
#         rotated_annotations.append({
#             'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
#             'category_id': ann['category_id']
#         })
#
#     return rotated_image, rotated_annotations
#
# # Run inference and collect predictions
# def run_inference(model, selected_images, images_path, angle=0, conf_threshold=0.25):
#     predictions = {}
#     for category_id, image_files in selected_images.items():
#         for image_file in image_files:
#             image_path = os.path.join(images_path, image_file)
#
#             # Load ground truth annotations
#             image_id = [img['id'] for img in coco.dataset['images'] if img['file_name'] == image_file][0]
#             ann_ids = coco.getAnnIds(imgIds=image_id)
#             anns = coco.loadAnns(ann_ids)
#
#             # Rotate image and annotations
#             rotated_image, rotated_annotations = rotate_image_and_labels(image_path, anns, angle)
#
#             # Run model inference
#             results = model.predict(source=rotated_image, conf=conf_threshold, save=False, verbose=False)
#             predictions[image_file] = {
#                 'predictions': results[0].boxes.data.cpu().numpy() if results[0].boxes else np.empty((0, 6)),
#                 'ground_truth': rotated_annotations
#             }
#     return predictions
#
# # Helper function to compute IoU
# def compute_iou(box1, box2):
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])
#
#     intersection = max(0, x2 - x1) * max(0, y2 - y1)
#     area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#     union = area_box1 + area_box2 - intersection
#
#     return intersection / union if union > 0 else 0
#
# # Calculate performance metrics
# def calculate_metrics(coco, predictions, iou_threshold=0.5):
#     y_true = []
#     y_pred = []
#     iou_scores = []
#     ap_per_class = {}
#
#     for image_file, data in predictions.items():
#         preds = data['predictions']
#         anns = data['ground_truth']
#
#         true_boxes = [[x, y, x + w, y + h] for x, y, w, h in [ann['bbox'] for ann in anns]]
#         pred_boxes = preds[:, :4] if preds.size > 0 else []
#         pred_scores = preds[:, 4] if preds.size > 0 else []
#
#         matched_gt = set()
#         for pb, score in zip(pred_boxes, pred_scores):
#             max_iou = 0
#             best_match = -1
#             for i, tb in enumerate(true_boxes):
#                 iou = compute_iou(tb, pb)
#                 if iou > max_iou:
#                     max_iou = iou
#                     best_match = i
#
#             if max_iou >= iou_threshold and best_match not in matched_gt:
#                 matched_gt.add(best_match)
#                 y_true.append(1)
#                 y_pred.append(1)
#                 iou_scores.append(max_iou)
#             else:
#                 y_pred.append(1)
#                 y_true.append(0)
#
#         for i, tb in enumerate(true_boxes):
#             if i not in matched_gt:
#                 y_true.append(1)
#                 y_pred.append(0)
#
#     for category_id in coco.getCatIds():
#         class_name = coco.loadCats([category_id])[0]['name']
#         category_trues = [
#             t for img_file, t in zip(predictions.keys(), y_true)
#             if class_name in [coco.loadCats([ann['category_id']])[0]['name'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=[
#                 [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]
#             ]))]
#         ]
#         category_preds = [
#             p for img_file, p in zip(predictions.keys(), y_pred)
#             if class_name in [coco.loadCats([ann['category_id']])[0]['name'] for ann in coco.loadAnns(coco.getAnnIds(imgIds=[
#                 [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_file][0]
#             ]))]
#         ]
#         ap_per_class[category_id] = precision_score(category_trues, category_preds) if category_preds else 0
#
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     mean_ap = np.mean(list(ap_per_class.values()))
#     mean_iou = np.mean(iou_scores) if iou_scores else 0
#
#     return {
#         'Precision': precision,
#         'Recall': recall,
#         'F1 Score': f1,
#         'Mean IoU': mean_iou,
#         'mAP': mean_ap,
#     }
#
# # Perform inference for rotated images
# rotation_angles = [0, 30, 60, 90, 120, 150, 180]
# results = []
#
# for angle in rotation_angles:
#     print(f"\nInference with rotation angle {angle}")
#     predictions = run_inference(model, selected_images, val_images_path, angle)
#     metrics = calculate_metrics(coco, predictions)
#     metrics['Rotation Angle'] = angle
#     results.append(metrics)
#     for metric, value in metrics.items():
#         if metric != 'Rotation Angle':
#             print(f"{metric}: {value:.4f}")
#
# # Save results to a CSV file
# results_df = pd.DataFrame(results)
# csv_path = "Yolo9_rotation_results.csv"
# results_df.to_csv(csv_path, index=False)
# print(f"\nResults saved to {csv_path}")
#
# # Plotting results
# metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU', 'mAP']
# for metric in metrics_to_plot:
#     plt.figure(figsize=(8, 6))
#     plt.plot(results_df['Rotation Angle'], results_df[metric], marker='o', label=metric)
#     plt.xlabel('Rotation Angle (degrees)', fontsize=12)
#     plt.ylabel(metric, fontsize=12)
#     plt.title(f'{metric} vs. Rotation Angle', fontsize=14)
#     plt.grid(True)
#     plt.legend(fontsize=12)
#     plt.tight_layout()
#     plt.show()
