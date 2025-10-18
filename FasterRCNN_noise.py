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
    Load a pretrained Faster R-CNN model, set it to evaluation mode, and move to device.
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


def add_noise(image_path, noise_std):
    """
    Read an image from image_path and add Gaussian noise in the normalized [0, 1] domain.

    Args:
        noise_std (float): Standard deviation of the Gaussian noise (normalized scale).

    Raises:
        ValueError: If the image cannot be loaded.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert image to float32 and scale to [0, 1]
    image = image.astype(np.float32) / 255.0
    # Generate Gaussian noise with mean=0 and std=noise_std
    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    # Add noise and clip to maintain [0, 1] range
    noisy_image = np.clip(image + noise, 0, 1)
    # Scale back to [0, 255] and convert to uint8
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return noisy_image


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
            else:  # original NMS
                weight = 0 if iou > Nt else 1
            scores[j] = scores[j] * weight

    keep_indices = np.where(scores > threshold)[0]
    if keep_indices.size:
        return np.hstack((boxes[keep_indices], scores[keep_indices].reshape(-1, 1)))
    else:
        return np.empty((0, 5))


def run_inference(model, selected_images, images_path, device, noise_std=0,
                  conf_threshold=0.5, use_soft_nms=False, soft_nms_params=None):
    """
    Run inference on selected images with optional Gaussian noise and Soft-NMS.

    Args:
        model: The detection model.
        selected_images (dict): Mapping from category_id to list of image filenames.
        images_path (str): Directory path containing the images.
        device: The torch device.
        noise_std (float): Standard deviation for Gaussian noise (normalized scale).
        conf_threshold (float): Confidence threshold for predictions.
        use_soft_nms (bool): Whether to apply Soft-NMS to predictions.
        soft_nms_params (dict): Parameters for soft_nms; keys: sigma, Nt, threshold, method.

    Returns:
        dict: Mapping from image filename to an array of predictions ([x1, y1, x2, y2, score]).
    """
    transform = T.Compose([T.ToTensor()])
    predictions = {}
    for category_images in selected_images.values():
        for image_file in category_images:
            image_path = os.path.join(images_path, image_file)
            try:
                if noise_std > 0:
                    image = add_noise(image_path, noise_std)
                else:
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError("Image not found.")
            except Exception as e:
                print(f"Skipping {image_file}: {e}")
                continue

            # Convert image from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image).to(device)

            with torch.no_grad():
                outputs = model([image_tensor])

            boxes = outputs[0]['boxes'].detach().cpu().numpy()
            scores = outputs[0]['scores'].detach().cpu().numpy()

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
            predictions[image_file] = preds
    return predictions


def calculate_metrics(coco, predictions, iou_threshold=0.5):
    """
    Calculate performance metrics (precision, recall, F1, and mean IoU) based on predictions and ground truth.

    Args:
        coco: The COCO annotations object.
        predictions (dict): Mapping from image filename to prediction arrays.
        iou_threshold (float): IoU threshold for a true positive.

    Returns:
        dict: Dictionary containing metrics.
    """
    y_true = []
    y_pred = []
    iou_scores = []

    for image_file, preds in predictions.items():
        image_info = next((img for img in coco.dataset['images'] if img['file_name'] == image_file), None)
        if image_info is None:
            continue
        image_id = image_info['id']
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        true_boxes = [
            [ann['bbox'][0], ann['bbox'][1],
             ann['bbox'][0] + ann['bbox'][2],
             ann['bbox'][1] + ann['bbox'][3]]
            for ann in anns
        ]
        matched_gt = set()

        for pb in preds[:, :4] if preds.shape[0] > 0 else []:
            max_iou = 0
            best_match = None
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

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mean_iou = np.mean(iou_scores) if iou_scores else 0

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Mean IoU': mean_iou,
    }


def plot_results(results_df, metrics_to_plot):
    """Plot specified metrics versus the noise level."""
    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(results_df['Noise Level'], results_df[metric], marker='o', label=metric)
        plt.xlabel('Noise Level')
        plt.ylabel(metric)
        plt.title(f'{metric} vs. Noise Level')
        plt.grid(True)
        plt.legend()
        plt.show()


def main():
    device = get_device()
    coco = load_coco_annotations(ANNOTATION_PATH)
    model = load_model(device)
    selected_images = get_class_images(coco, num_samples=5)

    # Define the noise levels in normalized domain (e.g., 0, 0.05, 0.1, 0.2, 0.3)
    noise_levels = [0.0, 0.05, 0.1,0.15, 0.2, 0.25, 0.3]
    results = []

    # Set to True to use Soft-NMS; adjust soft_nms_params if needed.
    use_soft_nms = True

    for noise_std in tqdm(noise_levels, desc="Processing noise levels"):
        print(f"\nRunning inference with noise std = {noise_std}")
        predictions = run_inference(model, selected_images, VAL_IMAGES_PATH, device,
                                    noise_std=noise_std, conf_threshold=0.5, use_soft_nms=use_soft_nms)
        metrics = calculate_metrics(coco, predictions)
        metrics['Noise Level'] = noise_std
        results.append(metrics)
        for metric, value in metrics.items():
            if metric != 'Noise Level':
                print(f"{metric}: {value:.4f}")

    # Save and plot results
    results_df = pd.DataFrame(results)
    csv_path = "FasterRCNN_noise_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'Mean IoU']
    plot_results(results_df, metrics_to_plot)


if __name__ == '__main__':
    main()
