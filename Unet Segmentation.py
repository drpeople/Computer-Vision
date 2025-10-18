import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2

# Set parameters
IMG_SIZE = (128, 128)
TEST_IMAGES_DIR = "path_to/ISIC_Test_Images"
TEST_MASKS_DIR = "path_to/ISIC_Test_Masks"
MODEL_PATH = "path_to/pretrained_unet_model.h5"


# Function to load and preprocess images
def load_images(image_dir, mask_dir, img_size):
    images = []
    masks = []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        # Load image
        img_path = os.path.join(image_dir, img_file)
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)

        # Load mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = load_img(mask_path, target_size=img_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0  # Normalize to [0, 1]
        masks.append(mask)

    return np.array(images), np.array(masks)


# Load data
print("Loading test data...")
test_images, test_masks = load_images(TEST_IMAGES_DIR, TEST_MASKS_DIR, IMG_SIZE)
print(f"Loaded {len(test_images)} test images.")

# Load the pre-trained U-Net model
print("Loading model...")
model = load_model(MODEL_PATH)

# Run predictions
print("Running predictions...")
predictions = model.predict(test_images)

# Threshold predictions to create binary masks
predictions = (predictions > 0.5).astype(np.uint8)


# Visualize results
def plot_results(images, true_masks, pred_masks, num_samples=5):
    plt.figure(figsize=(15, num_samples * 5))
    for i in range(num_samples):
        idx = np.random.randint(0, len(images))

        # Plot original image
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(images[idx])
        plt.title("Original Image")
        plt.axis("off")

        # Plot true mask
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(true_masks[idx].squeeze(), cmap="gray")
        plt.title("True Mask")
        plt.axis("off")

        # Plot predicted mask
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred_masks[idx].squeeze(), cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Plot results
print("Plotting results...")
plot_results(test_images, test_masks, predictions)
