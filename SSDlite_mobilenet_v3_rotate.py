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
# File paths
# -------------------------------
val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
annotation_path  = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"

# Load COCO annotations
coco = COCO(annotation_path)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load SSD-lite MobileNetV3-large (pretrained on COCO)
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
model.to(device)
model.eval()

# Pre-processing: BGR→RGB→PIL→Tensor
to_tensor = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
])

# -------------------------------
# Helper functions
# -------------------------------
def get_class_images(coco, images_path, num_samples=5):
    selected = {}
    for cid in coco.getCatIds():
        img_ids = coco.getImgIds(catIds=cid)
        img_ids.sort()
        sel = img_ids[:num_samples]
        selected[cid] = [coco.loadImgs(iid)[0]['file_name'] for iid in sel]
    return selected

def rotate_image_and_labels(image_path, annotations, angle):
    img = cv2.imread(image_path)
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0,0]); sin = np.abs(M[0,1])
    new_w = int(h*sin + w*cos)
    new_h = int(h*cos + w*sin)
    M[0,2] += (new_w/2) - center[0]
    M[1,2] += (new_h/2) - center[1]

    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))

    rotated_anns = []
    for ann in annotations:
        x,y,bb_w,bb_h = ann['bbox']
        corners = np.array([[x,y],
                            [x+bb_w,y],
                            [x,y+bb_h],
                            [x+bb_w,y+bb_h]])
        ones = np.ones((4,1))
        pts = np.hstack([corners, ones])
        rot = (M @ pts.T).T
        x_min, y_min = rot[:,0].min(), rot[:,1].min()
        x_max, y_max = rot[:,0].max(), rot[:,1].max()
        rotated_anns.append({
            'bbox': [x_min, y_min, x_max-x_min, y_max-y_min],
            'category_id': ann['category_id']
        })

    return rotated_img, rotated_anns

def run_inference(model, selected_images, images_path, angle=0, conf_threshold=0.25):
    preds = {}
    for cid, files in selected_images.items():
        for fname in files:
            path = os.path.join(images_path, fname)
            # load GT
            img_id = next(img['id'] for img in coco.dataset['images'] if img['file_name']==fname)
            anns   = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            # rotate
            rot_img, rot_anns = rotate_image_and_labels(path, anns, angle)
            # prepare
            inp = to_tensor(cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB)).to(device)
            with torch.no_grad():
                out = model([inp])[0]
            boxes  = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            keep = scores >= conf_threshold
            if keep.any():
                arr = np.hstack([boxes[keep], scores[keep,None]])
            else:
                arr = np.empty((0,5))
            preds[fname] = {
                'predictions':   arr,
                'ground_truth': rot_anns
            }
    return preds

def compute_iou(b1, b2):
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    uni = a1 + a2 - inter
    return inter/uni if uni>0 else 0

def calculate_metrics(coco, predictions, iou_threshold=0.5):
    y_true, y_pred, ious = [], [], []

    for fname, data in predictions.items():
        preds = data['predictions']
        gts   = data['ground_truth']
        gt_boxes = [[x,y,x+w,y+h] for x,y,w,h in [ann['bbox'] for ann in gts]]

        pb = preds[:, :4] if preds.size else []
        ps = preds[:, 4]  if preds.size else []

        matched = set()
        for b, s in zip(pb, ps):
            best_i, best_j = 0, -1
            for j, gt in enumerate(gt_boxes):
                i = compute_iou(gt, b)
                if i>best_i:
                    best_i, best_j = i, j
            if best_i>=iou_threshold and best_j not in matched:
                matched.add(best_j)
                y_true.append(1); y_pred.append(1); ious.append(best_i)
            else:
                y_true.append(0); y_pred.append(1)
        for j in range(len(gt_boxes)):
            if j not in matched:
                y_true.append(1); y_pred.append(0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(   y_true, y_pred, zero_division=0)
    f1        = f1_score(       y_true, y_pred, zero_division=0)
    mean_iou  = np.mean(ious) if ious else 0.0

    # simple per-class AP placeholder
    ap_per_class = [precision for _ in coco.getCatIds()]
    mAP = float(np.mean(ap_per_class))

    return {
        'Precision': precision,
        'Recall':    recall,
        'F1 Score':  f1,
        'Mean IoU':  mean_iou,
        'mAP':       mAP,
    }

# -------------------------------
# Main: rotate through angles
# -------------------------------
selected_images  = get_class_images(coco, val_images_path)
rotation_angles  = [0, 30, 60, 90, 120, 150, 180,210,240,270,300,330]
results = []

for angle in rotation_angles:
    print(f"\nInference with angle {angle}")
    preds   = run_inference(model, selected_images, val_images_path, angle)
    mets    = calculate_metrics(coco, preds)
    mets['Rotation Angle'] = angle
    results.append(mets)
    for k, v in mets.items():
        if k != 'Rotation Angle':
            print(f"  {k}: {v:.4f}")

# Save & plot
df = pd.DataFrame(results)
csv_path = "SSDlite_rotation_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

metrics_to_plot = ['Precision','Recall','F1 Score','Mean IoU','mAP']
for metric in metrics_to_plot:
    plt.figure(figsize=(8,6))
    plt.plot(df['Rotation Angle'], df[metric], marker='o')
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel(metric)
    plt.title(f'{metric} vs. Rotation Angle')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
