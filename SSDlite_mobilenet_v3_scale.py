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

# -------------------------------
# File paths and Initialization
# -------------------------------
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations
coco = COCO(annotation_path)

# Device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load SS DLite MobileNet V3 Large (pretrained on COCO)
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.to(device)
model.eval()

# Transform: BGR→RGB PIL→Tensor
to_tensor = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),  # → [0,1]
])

# -------------------------------
# Helper Functions
# -------------------------------
def get_class_images(coco, images_path, num_samples=5):
    selected = {}
    for cat_id in coco.getCatIds():
        img_ids = coco.getImgIds(catIds=cat_id)
        img_ids.sort()
        sel_ids = img_ids[:num_samples]
        selected[cat_id] = [
            coco.loadImgs(iid)[0]['file_name'] for iid in sel_ids
        ]
    return selected

def scale_image(path, scale_factor):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load {path}")
    h, w = img.shape[:2]
    new_w, new_h = int(w*scale_factor), int(h*scale_factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def run_inference(model, selected_images, images_path,
                  scale_factor=1.0, conf_threshold=0.25):
    """
    Scale input → detect → rescale boxes back → return dict[file→Nx5 array]
    """
    preds = {}
    for cat_id, files in selected_images.items():
        for fname in files:
            fullp = os.path.join(images_path, fname)
            img = (scale_image(fullp, scale_factor)
                   if scale_factor!=1.0 else
                   cv2.imread(fullp))
            # prepare tensor
            tensor = to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device)
            with torch.no_grad():
                out = model([tensor])[0]
            boxes  = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            # filter by score
            keep   = scores >= conf_threshold
            if keep.any():
                arr = np.hstack([boxes[keep] / scale_factor,
                                 scores[keep,None]])
            else:
                arr = np.empty((0,5))
            preds[fname] = arr
    return preds

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    uni = a1 + a2 - inter
    return inter/uni if uni>0 else 0

def calculate_metrics(coco, predictions, iou_thr=0.5):
    y_true, y_pred, ious = [], [], []

    for fname, pred_arr in predictions.items():
        # lookup GT boxes
        img_id = next(i['id'] for i in coco.dataset['images']
                      if i['file_name']==fname)
        anns   = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        gt_boxes = [[x, y, x+w, y+h] for ann in anns
                    for x,y,w,h in [ann['bbox']]]

        pb = pred_arr[:, :4] if pred_arr.size else []
        ps = pred_arr[:, 4]  if pred_arr.size else []

        matched = set()
        # match preds → GT
        for b, s in zip(pb, ps):
            best_iou, best_j = 0, -1
            for j, gt in enumerate(gt_boxes):
                iou = compute_iou(gt, b)
                if iou>best_iou:
                    best_iou, best_j = iou, j
            if best_iou>=iou_thr and best_j not in matched:
                matched.add(best_j)
                y_true.append(1); y_pred.append(1); ious.append(best_iou)
            else:
                y_true.append(0); y_pred.append(1)
        # false negatives
        for j in range(len(gt_boxes)):
            if j not in matched:
                y_true.append(1); y_pred.append(0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(   y_true, y_pred, zero_division=0)
    f1        = f1_score(       y_true, y_pred, zero_division=0)
    mean_iou  = np.mean(ious) if ious else 0.0

    # simple mAP = average of per-class precision (placeholder)
    ap_per_class = []
    for cid in coco.getCatIds():
        ap_per_class.append(precision)
    mAP = float(np.mean(ap_per_class))

    return {
        'Precision': precision,
        'Recall':    recall,
        'F1 Score':  f1,
        'Mean IoU':  mean_iou,
        'mAP':       mAP,
    }

# -------------------------------
# Main Inference and Evaluation
# -------------------------------
selected_images = get_class_images(coco, val_images_path)

# Define scale factors
scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3]
results = []

for sf in scale_factors:
    print(f"\nInference with scale factor {sf}")
    preds   = run_inference(model, selected_images, val_images_path,
                            scale_factor=sf)
    metrics = calculate_metrics(coco, preds)
    metrics['Scale Factor'] = sf
    results.append(metrics)
    print(f" Recall: {metrics['Recall']:.4f}")
    print(f" F1 Score: {metrics['F1 Score']:.4f}")
    print(f" Mean IoU: {metrics['Mean IoU']:.4f}")
    print(f" mAP: {metrics['mAP']:.4f}")

# Save to CSV
df = pd.DataFrame(results)
csv_path = "SSDlite_scale_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# Plotting
metrics_to_plot = ['Recall', 'F1 Score', 'Mean IoU', 'mAP']
fig, axs = plt.subplots(2, 2, figsize=(14,10))
axs = axs.flatten()
for i, m in enumerate(metrics_to_plot):
    axs[i].plot(df['Scale Factor'], df[m], marker='o')
    axs[i].set_xlabel('Scale Factor')
    axs[i].set_ylabel(m)
    axs[i].set_title(f'{m} vs. Scale Factor')
    axs[i].grid(True)
plt.tight_layout()
plt.show()
