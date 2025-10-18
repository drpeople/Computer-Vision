# """
# Evaluate detection robustness to Gaussian blur
# using a COCO-pre-trained RetinaNet-ResNet-50-FPN (v2).
#
# Requires:
#     pip install --upgrade torch torchvision opencv-python-headless \
#         pycocotools scikit-learn matplotlib pandas pillow
# """
# import os, cv2, numpy as np, pandas as pd
# from PIL import Image
# from pycocotools.coco import COCO
# from sklearn.metrics import precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt
# import torch
# import torchvision
# from torchvision.models.detection import (
#     retinanet_resnet50_fpn_v2,
#     RetinaNet_ResNet50_FPN_V2_Weights,
# )
#
# # ------------------------------------------------------------------
# # Paths
# val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
# annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"
#
# # ------------------------------------------------------------------
# # COCO helpers
# coco = COCO(annotation_path)
#
# def get_class_images(coco, root, num_samples=5):
#     sel = {}
#     for cid in coco.getCatIds():
#         img_ids = sorted(coco.getImgIds(catIds=cid))[:num_samples]
#         sel[cid] = [coco.loadImgs(i)[0]["file_name"] for i in img_ids]
#     return sel
#
# selected_images = get_class_images(coco, val_images_path)
#
# # ------------------------------------------------------------------
# # Load RetinaNet-R50-FPN (v2) and its preprocessing pipeline
# weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT   # COCO-2017 ✕ 90 classes
# model   = retinanet_resnet50_fpn_v2(weights=weights).eval()
# device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# preprocess = weights.transforms()                     # size/mean/std. etc. :contentReference[oaicite:0]{index=0}
#
# # ------------------------------------------------------------------
# # Image utils
# def blur_image(path, sigma):
#     img = cv2.imread(path)
#     return cv2.GaussianBlur(img, (5, 5), sigmaX=sigma, sigmaY=sigma)
#
# def run_inference(model, image_dict, root, sigma=0, conf_thr=0.45):
#     """Return {file_name: np.ndarray([[x1 y1 x2 y2 conf cls], …])}."""
#     out = {}
#     for cid, files in image_dict.items():
#         for fname in files:
#             p = os.path.join(root, fname)
#             bgr = blur_image(p, sigma) if sigma else cv2.imread(p)
#             rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#             pil = Image.fromarray(rgb)
#             tensor = preprocess(pil).to(device)        # shape C×H×W (float32 0-1)
#             preds  = model([tensor])[0]                # list length 1
#
#             boxes  = preds["boxes"].detach().cpu().numpy()
#             scores = preds["scores"].detach().cpu().numpy()
#             labels = preds["labels"].detach().cpu().numpy()
#
#             keep   = scores >= conf_thr
#             arr    = np.concatenate(
#                 [boxes[keep], scores[keep, None], labels[keep, None]], axis=1
#             )
#             out[fname] = arr if arr.size else np.empty((0, 6))
#     return out
#
# # ------------------------------------------------------------------
# # IoU + metrics (unchanged)
# def compute_iou(b1, b2):
#     x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
#     x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
#     inter  = max(0, x2 - x1) * max(0, y2 - y1)
#     area1  = (b1[2] - b1[0]) * (b1[3] - b1[1])
#     area2  = (b2[2] - b2[0]) * (b2[3] - b2[1])
#     union  = area1 + area2 - inter
#     return inter / union if union else 0
#
# def calculate_metrics(coco, preds, iou_thr=0.5):
#     y_t, y_p, ious, ap_cls = [], [], [], {}
#     for f, det in preds.items():
#         img_id = next(img["id"] for img in coco.dataset["images"] if img["file_name"] == f)
#         gt_ann = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
#         gt_box = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in gt_ann)]
#
#         pb, ps = (det[:, :4], det[:, 4]) if det.size else ([], [])
#         matched = set()
#         for b, s in zip(pb, ps):
#             best, idx = 0, -1
#             for i, g in enumerate(gt_box):
#                 iou = compute_iou(g, b)
#                 if iou > best:
#                     best, idx = iou, i
#             if best >= iou_thr and idx not in matched:
#                 matched.add(idx); y_t.append(1); y_p.append(1); ious.append(best)
#             else:
#                 y_t.append(0); y_p.append(1)
#         for i in range(len(gt_box)):
#             if i not in matched:
#                 y_t.append(1); y_p.append(0)
#
#     for cid in coco.getCatIds():
#         cls_t = [t for f, t in zip(preds.keys(), y_t)
#                  if any(a["category_id"] == cid for a in
#                         coco.loadAnns(coco.getAnnIds(imgIds=[
#                             next(img["id"] for img in coco.dataset["images"] if img["file_name"] == f)
#                         ])))]
#         cls_p = [p for f, p in zip(preds.keys(), y_p)
#                  if any(a["category_id"] == cid for a in
#                         coco.loadAnns(coco.getAnnIds(imgIds=[
#                             next(img["id"] for img in coco.dataset["images"] if img["file_name"] == f)
#                         ])))]
#         ap_cls[cid] = precision_score(cls_t, cls_p) if cls_p else 0
#
#     return dict(
#         Precision = precision_score(y_t, y_p),
#         Recall    = recall_score(y_t, y_p),
#         F1_Score  = f1_score(y_t, y_p),
#         Mean_IoU  = np.mean(ious) if ious else 0,
#         mAP       = np.mean(list(ap_cls.values()))
#     )
#
# # ------------------------------------------------------------------
# # Main loop over blur levels
# sigmas, records = [0, 1, 2, 3, 4], []
# for s in sigmas:
#     print(f"\nSigma {s}: inference running …")
#     pr = run_inference(model, selected_images, val_images_path, sigma=s)
#     mt = calculate_metrics(coco, pr); mt["Sigma"] = s
#     records.append(mt)
#     for k, v in mt.items():
#         if k != "Sigma": print(f"{k}: {v:.4f}")
#
# # ------------------------------------------------------------------
# # Save + plot
# df = pd.DataFrame(records)
# csv_path = "RetinaNet_blurv2_results.csv"
# df.to_csv(csv_path, index=False)
# print(f"\nResults saved to {csv_path}")
#
# for m in ["Precision", "Recall", "F1_Score", "Mean_IoU", "mAP"]:
#     plt.figure(figsize=(8, 6))
#     plt.plot(df["Sigma"], df[m], marker="o")
#     plt.title(f"{m} vs. Gaussian Blur σ")
#     plt.xlabel("Sigma (blur strength)"); plt.ylabel(m); plt.grid(True)
#     plt.tight_layout(); plt.show()


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

# File paths
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations
coco = COCO(annotation_path)

# Load RetinaNet model (pre-trained on COCO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = retinanet_resnet50_fpn_v2(pretrained=True)
model.to(device)
model.eval()

# Image preprocessing function
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor()
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
    kernel_size = (5, 5)  # Fixed kernel size
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)
    return blurred_image


# Run inference using RetinaNet
def run_inference(model, selected_images, images_path, sigma=0, conf_threshold=0.45):
    predictions = {}

    for category_id, image_files in selected_images.items():
        for image_file in image_files:
            image_path = os.path.join(images_path, image_file)

            if sigma > 0:
                image = blur_image(image_path, sigma)
            else:
                image = cv2.imread(image_path)

            # Convert to RGB format for RetinaNet
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image_rgb).to(device).unsqueeze(0)

            # Run model inference
            with torch.no_grad():
                outputs = model(image_tensor)

            # Process predictions
            pred_boxes = outputs[0]['boxes'].cpu().numpy()
            pred_scores = outputs[0]['scores'].cpu().numpy()
            pred_labels = outputs[0]['labels'].cpu().numpy()

            # Apply confidence threshold
            valid_indices = pred_scores > conf_threshold
            pred_boxes = pred_boxes[valid_indices]
            pred_scores = pred_scores[valid_indices]
            pred_labels = pred_labels[valid_indices]

            # Save predictions
            predictions[image_file] = np.column_stack((pred_boxes, pred_scores, pred_labels))

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
csv_path = "RetinaNetv2_blur_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# Plotting results
metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU']
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
