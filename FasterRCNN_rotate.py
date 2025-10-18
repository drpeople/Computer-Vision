import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants: File paths (update these as needed)
VAL_IMAGES_PATH = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
ANNOTATION_PATH = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"


def get_device():
    """Return the torch device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_coco_annotations(annotation_path):
    """Load and return COCO annotations."""
    return COCO(annotation_path)


def load_model(device):
    """
    Load a pretrained Faster R-CNN model, set it to evaluation mode,
    and move it to the specified device.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    return model


def get_class_images(coco, num_samples=5):
    """
    For each category in the COCO dataset, select up to num_samples images.

    Returns:
        dict: Mapping from category_id to a list of image filenames.
    """
    selected_images = {}
    for category_id in coco.getCatIds():
        image_ids = sorted(coco.getImgIds(catIds=category_id))
        sampled_ids = image_ids[:min(num_samples, len(image_ids))]
        selected_images[category_id] = [
            coco.loadImgs(img_id)[0]['file_name'] for img_id in sampled_ids
        ]
    return selected_images


def rotate_image_and_labels(image_path, annotations, angle, interp=cv2.INTER_CUBIC):
    """
    Rotate an image and its ground-truth bounding boxes by a given angle.

    Uses cv2.boxPoints and cv2.boundingRect for a more robust rotated box.

    Args:
        image_path (str): Path to the image.
        annotations (list): List of annotation dicts with 'bbox' in [x, y, w, h].
        angle (float): Rotation angle in degrees (counterclockwise).
        interp: Interpolation flag for cv2.warpAffine.

    Returns:
        rotated_image (np.array): The rotated image.
        rotated_annotations (list): List of rotated annotations with updated 'bbox'.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix and new dimensions.
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Rotate image using cubic interpolation for better quality.
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), flags=interp)

    rotated_annotations = []
    for ann in annotations:
        x, y, bw, bh = ann['bbox']
        # Define the box corners.
        box = np.array([[x, y],
                        [x + bw, y],
                        [x, y + bh],
                        [x + bw, y + bh]], dtype=np.float32)
        # Convert box corners to homogeneous coordinates.
        ones = np.ones((box.shape[0], 1), dtype=np.float32)
        points = np.hstack([box, ones])
        # Apply the rotation.
        rotated_points = np.dot(rotation_matrix, points.T).T

        # Use cv2.boundingRect to get the axis-aligned rectangle.
        rotated_points = rotated_points.astype(np.float32)
        x_new, y_new, w_new, h_new = cv2.boundingRect(rotated_points)
        rotated_annotations.append({
            'bbox': [x_new, y_new, w_new, h_new],
            'category_id': ann['category_id']
        })

    return rotated_image, rotated_annotations


def soft_nms(boxes, scores, sigma=0.5, Nt=0.3, threshold=0.001, method='gaussian'):
    """
    Apply Soft-NMS to a set of bounding boxes.

    Args:
        boxes (np.array): Array of bounding boxes (N, 4) in [x1, y1, x2, y2] format.
        scores (np.array): Array of confidence scores for each box (N,).
        sigma (float): Sigma for Gaussian penalty.
        Nt (float): IoU threshold for penalty.
        threshold (float): Boxes with scores below this will be discarded.
        method (str): 'gaussian' or 'linear' for score decay.

    Returns:
        np.array: Array of predictions in the form [x1, y1, x2, y2, score].
    """
    N = boxes.shape[0]
    boxes = boxes.copy()
    scores = scores.copy()

    for i in range(N):
        maxpos = i + np.argmax(scores[i:])
        boxes[[i, maxpos]] = boxes[[maxpos, i]]
        scores[[i, maxpos]] = scores[[maxpos, i]]
        box_i = boxes[i]
        for j in range(i + 1, N):
            iou = compute_iou(box_i, boxes[j])
            if method == 'linear':
                weight = 1 - iou if iou > Nt else 1
            elif method == 'gaussian':
                weight = np.exp(-(iou * iou) / sigma)
            else:
                weight = 0 if iou > Nt else 1
            scores[j] *= weight

    keep_indices = np.where(scores > threshold)[0]
    if keep_indices.size:
        return np.hstack((boxes[keep_indices], scores[keep_indices].reshape(-1, 1)))
    else:
        return np.empty((0, 5))


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two boxes given as [x1, y1, x2, y2].
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


def run_inference(model, selected_images, images_path, device, angle=0, conf_threshold=0.5,
                  use_soft_nms=False, soft_nms_params=None):
    """
    Run inference on selected images after rotating them by a specified angle.

    For each image, the ground-truth annotations are rotated along with the image.

    Returns:
        dict: Mapping from image filename to a dict with:
              'predictions': array of predictions ([x1, y1, x2, y2, score]),
              'ground_truth': list of rotated annotations.
    """
    transform = T.Compose([T.ToTensor()])
    predictions = {}
    for category_images in selected_images.values():
        for image_file in category_images:
            image_path = os.path.join(images_path, image_file)
            # Retrieve ground truth annotations.
            image_info = next((img for img in coco.dataset['images'] if img['file_name'] == image_file), None)
            if image_info is None:
                print(f"Skipping {image_file}: no image info found.")
                continue
            image_id = image_info['id']
            ann_ids = coco.getAnnIds(imgIds=image_id)
            anns = coco.loadAnns(ann_ids)

            try:
                rotated_image, rotated_annotations = rotate_image_and_labels(image_path, anns, angle)
            except Exception as e:
                print(f"Skipping {image_file}: {e}")
                continue

            # Convert rotated image to RGB and to tensor.
            rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
            image_tensor = transform(rotated_image).to(device)

            with torch.no_grad():
                outputs = model([image_tensor])

            boxes = outputs[0]['boxes'].detach().cpu().numpy()
            scores = outputs[0]['scores'].detach().cpu().numpy()

            # Optionally apply Soft-NMS.
            if use_soft_nms:
                if soft_nms_params is None:
                    soft_nms_params = {'sigma': 0.5, 'Nt': 0.3, 'threshold': conf_threshold, 'method': 'gaussian'}
                preds = soft_nms(boxes, scores, **soft_nms_params)
            else:
                valid_indices = np.where(scores >= conf_threshold)[0]
                if valid_indices.size:
                    preds = np.hstack((boxes[valid_indices], scores[valid_indices].reshape(-1, 1)))
                else:
                    preds = np.empty((0, 5))

            predictions[image_file] = {
                'predictions': preds,
                'ground_truth': rotated_annotations
            }
    return predictions


def calculate_metrics(coco, predictions, iou_threshold=0.5):
    """
    Calculate performance metrics (precision, recall, F1, and mean IoU) based on predictions
    and rotated ground truth.
    """
    y_true = []
    y_pred = []
    iou_scores = []
    # For simplicity, mAP is not fully computed here.
    ap_per_class = {}

    for image_file, data in predictions.items():
        preds = data['predictions']
        anns = data['ground_truth']
        true_boxes = [[ann['bbox'][0], ann['bbox'][1],
                       ann['bbox'][0] + ann['bbox'][2],
                       ann['bbox'][1] + ann['bbox'][3]] for ann in anns]
        pred_boxes = preds[:, :4] if preds.size > 0 else []
        pred_scores = preds[:, 4] if preds.size > 0 else []

        matched_gt = set()
        for pb, score in zip(pred_boxes, pred_scores):
            max_iou = 0
            best_match = -1
            for idx, tb in enumerate(true_boxes):
                iou = compute_iou(tb, pb)
                if iou > max_iou:
                    max_iou = iou
                    best_match = idx
            if max_iou >= iou_threshold and best_match not in matched_gt:
                matched_gt.add(best_match)
                y_true.append(1)
                y_pred.append(1)
                iou_scores.append(max_iou)
            else:
                y_true.append(0)
                y_pred.append(1)

        for idx in range(len(true_boxes)):
            if idx not in matched_gt:
                y_true.append(1)
                y_pred.append(0)

    # Here mAP is set to zero; consider using COCOeval for a more robust mAP calculation.
    for category_id in coco.getCatIds():
        ap_per_class[category_id] = 0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    mean_ap = np.mean(list(ap_per_class.values()))

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Mean IoU': mean_iou,
        'mAP': mean_ap,
    }


def plot_results(results_df, metrics_to_plot):
    """Plot specified metrics versus the rotation angle."""
    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(results_df['Rotation Angle'], results_df[metric], marker='o', label=metric)
        plt.xlabel('Rotation Angle (degrees)')
        plt.ylabel(metric)
        plt.title(f'{metric} vs. Rotation Angle')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    device = get_device()
    global coco  # Used in run_inference
    coco = load_coco_annotations(ANNOTATION_PATH)
    model = load_model(device)
    selected_images = get_class_images(coco, num_samples=5)

    # Define rotation angles (in degrees) to evaluate.
    rotation_angles = [0, 30, 60, 90, 120, 150, 180]
    results = []

    # Enable Soft-NMS if desired.
    use_soft_nms = True  # Try enabling soft_nms to see if it helps.

    # You might also experiment with a lower conf_threshold for rotated images.
    conf_threshold = 0.4

    for angle in tqdm(rotation_angles, desc="Processing rotation angles"):
        print(f"\nRunning inference with rotation angle = {angle}")
        predictions = run_inference(model, selected_images, VAL_IMAGES_PATH, device,
                                    angle=angle, conf_threshold=conf_threshold, use_soft_nms=use_soft_nms)
        metrics = calculate_metrics(coco, predictions)
        metrics['Rotation Angle'] = angle
        results.append(metrics)
        for metric, value in metrics.items():
            if metric != 'Rotation Angle':
                print(f"{metric}: {value:.4f}")

    results_df = pd.DataFrame(results)
    csv_path = "FasterRCNN_rotation_results_improved.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU', 'mAP']
    plot_results(results_df, metrics_to_plot)


if __name__ == '__main__':
    main()
