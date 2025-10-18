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

def add_noise(image_path, noise_std):
    # Read and convert image from BGR to RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    noisy = np.clip(img + noise, 0, 1)
    noisy = (noisy * 255).astype(np.uint8)
    return noisy

def run_inference(model, selected_images, images_path, noise_std=0.0, conf_threshold=0.25):
    preds = {}
    for cid, files in selected_images.items():
        for fname in files:
            path = os.path.join(images_path, fname)
            # load image (noisy or original)
            if noise_std > 0:
                img = add_noise(path, noise_std)
            else:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # prepare tensor
            tensor = to_tensor(img).to(device)
            with torch.no_grad():
                out = model([tensor])[0]

            boxes  = out['boxes'].cpu().numpy()
            scores = out['scores'].cpu().numpy()
            keep   = scores >= conf_threshold

            if keep.any():
                arr = np.hstack([boxes[keep], scores[keep, None]])
            else:
                arr = np.empty((0, 5))

            preds[fname] = arr
    return preds

def compute_iou(b1, b2):
    x1,y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2,y2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    uni = a1 + a2 - inter
    return inter/uni if uni>0 else 0

def calculate_metrics(coco, predictions, iou_threshold=0.5):
    y_true, y_pred, ious = [], [], []

    for fname, pred_arr in predictions.items():
        # get GT boxes
        img_id = next(img['id'] for img in coco.dataset['images']
                      if img['file_name']==fname)
        anns   = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        gt_boxes = [[x,y,x+w,y+h] for ann in anns for x,y,w,h in [ann['bbox']]]

        pb = pred_arr[:, :4] if pred_arr.size else []
        ps = pred_arr[:, 4]  if pred_arr.size else []

        matched = set()
        for b, s in zip(pb, ps):
            best_i, best_j = 0, -1
            for j, gt in enumerate(gt_boxes):
                iou = compute_iou(gt, b)
                if iou > best_i:
                    best_i, best_j = iou, j
            if best_i >= iou_threshold and best_j not in matched:
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
# Main: inference over noise levels
# -------------------------------
selected_images = get_class_images(coco, val_images_path)
noise_levels   = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
results = []

for ns in noise_levels:
    print(f"\nInference with noise std {ns}")
    preds   = run_inference(model, selected_images, val_images_path, noise_std=ns)
    mets    = calculate_metrics(coco, preds)
    mets['Noise Std'] = ns
    results.append(mets)
    for k, v in mets.items():
        if k != 'Noise Std':
            print(f"  {k}: {v:.4f}")

# Save & plot
df = pd.DataFrame(results)
csv_path = "SSD_noise_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU', 'mAP']
for metric in metrics_to_plot:
    plt.figure(figsize=(8,6))
    plt.plot(df['Noise Std'], df[metric], marker='o')
    plt.xlabel('Noise Std')
    plt.ylabel(metric)
    plt.title(f'{metric} vs. Noise Std')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
