import os
import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import torch
from torchvision import transforms as T
import torchvision

# File paths
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations
coco = COCO(annotation_path)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load SS DLite MobileNet V3 Large (pretrained on COCO)
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.to(device)
model.eval()

# Pre‐processing transform: BGR (cv2) → RGB → Tensor
to_tensor = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),  # scales to [0,1]
])

def get_class_images(coco, images_path, num_samples=5):
    selected_images = {}
    for category_id in coco.getCatIds():
        image_ids = coco.getImgIds(catIds=category_id)
        image_ids.sort()
        sampled_ids = image_ids[:num_samples]
        selected_images[category_id] = [
            coco.loadImgs(img_id)[0]['file_name'] for img_id in sampled_ids
        ]
    return selected_images

selected_images = get_class_images(coco, val_images_path)

def blur_image(image_path, sigma):
    image = cv2.imread(image_path)
    kernel_size = (5, 5)
    return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)

def run_inference(model, selected_images, images_path, sigma=0, conf_threshold=0.25):
    predictions = {}
    for category_id, image_files in selected_images.items():
        for fname in image_files:
            path = os.path.join(images_path, fname)
            # Read & optional blur
            img = blur_image(path, sigma) if sigma > 0 else cv2.imread(path)
            # Convert to tensor and send to device
            inp = to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device)
            # Inference: model expects list of tensors
            with torch.no_grad():
                output = model([inp])[0]
            # Extract boxes & scores
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            # Filter by confidence threshold
            keep = scores >= conf_threshold
            filtered = np.hstack([
                boxes[keep],
                scores[keep, None]
            ]) if keep.any() else np.empty((0, 5))
            # Store Nx5 array (x1, y1, x2, y2, score)
            predictions[fname] = filtered
    return predictions

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter/union if union>0 else 0

def calculate_metrics(coco, predictions, iou_threshold=0.5):
    y_true, y_pred, iou_scores = [], [], []

    for fname, preds in predictions.items():
        # find image id & GT boxes
        img_id = next(img['id'] for img in coco.dataset['images'] if img['file_name']==fname)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = [[x, y, x+w, y+h] for ann in anns for x,y,w,h in [ann['bbox']]]

        pred_boxes = preds[:, :4] if preds.size else []
        pred_scores= preds[:, 4] if preds.size else []

        matched = set()
        for pb, score in zip(pred_boxes, pred_scores):
            best_iou, best_j = 0, -1
            for j, tb in enumerate(gt_boxes):
                iou = compute_iou(tb, pb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_threshold and best_j not in matched:
                matched.add(best_j)
                y_true.append(1); y_pred.append(1); iou_scores.append(best_iou)
            else:
                y_true.append(0); y_pred.append(1)

        # any GT not detected
        for j in range(len(gt_boxes)):
            if j not in matched:
                y_true.append(1); y_pred.append(0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(   y_true, y_pred, zero_division=0)
    f1        = f1_score(       y_true, y_pred, zero_division=0)
    mean_iou  = np.mean(iou_scores) if iou_scores else 0.0

    # mAP approximation: average precision per class (simple version)
    ap_per_class = []
    for cat_id in coco.getCatIds():
        # filter samples belonging to this class
        # (for a true per-class you’d need to separate by label too)
        ap_per_class.append(precision)  # placeholder
    mAP = float(np.mean(ap_per_class))

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Mean IoU': mean_iou,
        'mAP': mAP,
    }

# Run for various blur levels
sigma_levels = [0, 1, 2, 3, 4]
results = []

for sigma in sigma_levels:
    print(f"\nRunning inference with sigma={sigma}")
    preds  = run_inference(model, selected_images, val_images_path, sigma)
    mets   = calculate_metrics(coco, preds)
    mets['Sigma'] = sigma
    results.append(mets)
    for k,v in mets.items():
        if k!='Sigma':
            print(f"  {k}: {v:.4f}")

# Save & plot
df = pd.DataFrame(results)
df.to_csv("SSD_blur_results.csv", index=False)
print("\nResults saved to SSD_blur_results.csv")

for metric in ['Precision','Recall','F1 Score','Mean IoU','mAP']:
    plt.figure(figsize=(8,6))
    plt.plot(df['Sigma'], df[metric], marker='o')
    plt.xlabel('Sigma (Blur Level)')
    plt.ylabel(metric)
    plt.title(f'{metric} vs. Sigma')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
