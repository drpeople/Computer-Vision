# ## BLUR
# import cv2
# import matplotlib.pyplot as plt
#
# # === CONFIGURATION ===
# image_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017\000000007386.jpg"  # <-- Replace with your image path
# sigma_levels = [0, 1, 2, 3, 4]
#
# # === FUNCTION TO APPLY BLUR ===
# def blur_image(image, sigma):
#     kernel_size = (5, 5)
#     return cv2.GaussianBlur(image, kernel_size, sigmaX=sigma, sigmaY=sigma)
#
# # === LOAD IMAGE ===
# image_bgr = cv2.imread(image_path)
# if image_bgr is None:
#     raise FileNotFoundError(f"Image not found at: {image_path}")
#
# # === APPLY BLUR AND CONVERT TO RGB ===
# blurred_images = []
# for sigma in sigma_levels:
#     blurred = blur_image(image_bgr, sigma) if sigma > 0 else image_bgr.copy()
#     image_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
#     blurred_images.append((image_rgb, sigma))
#
# # === PLOT FULL WIDTH, MINIMAL SPACING ===
# fig, axes = plt.subplots(1, len(sigma_levels), figsize=(30, 10))  # Massive width & height
#
# for ax, (img, sigma) in zip(axes, blurred_images):
#     ax.imshow(img)
#     ax.axis('off')
#     ax.text(0.5, -0.1, f"Sigma = {sigma}", fontsize=20, ha='center', transform=ax.transAxes)
#
# plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.02)
# plt.show()



### NOISE

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # === CONFIGURATION ===
# image_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017\000000007386.jpg"  # <-- Replace with your actual image path
# noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]  # Standard deviations for Gaussian noise
#
# # === FUNCTION TO ADD GAUSSIAN NOISE ===
# def add_noise(image_bgr, noise_std):
#     image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     image = image.astype(np.float32) / 255.0
#     noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
#     noisy_image = np.clip(image + noise, 0, 1)
#     noisy_image = (noisy_image * 255).astype(np.uint8)
#     return noisy_image
#
# # === LOAD IMAGE ===
# image_bgr = cv2.imread(image_path)
# if image_bgr is None:
#     raise FileNotFoundError(f"Image not found at: {image_path}")
#
# # === APPLY NOISE AND COLLECT RESULTS ===
# noised_images = []
# for std in noise_levels:
#     noisy = add_noise(image_bgr, std) if std > 0 else cv2.cvtColor(image_bgr.copy(), cv2.COLOR_BGR2RGB)
#     noised_images.append((noisy, std))
#
# # === PLOT — LARGE IMAGES, LABELS UNDERNEATH ===
# fig, axes = plt.subplots(1, len(noise_levels), figsize=(30, 10))
# fig.suptitle("Effect of Gaussian Noise on Image", fontsize=22, weight='bold')
#
# for ax, (img, std) in zip(axes, noised_images):
#     ax.imshow(img)
#     ax.axis('off')
#     ax.text(0.5, -0.1, f"Noise Std = {std}", fontsize=18, ha='center', transform=ax.transAxes)
#
# plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.05, wspace=0.02)
# plt.show()


#  ROTATE
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # === CONFIGURATION ===
# image_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017\000000007386.jpg"  # <-- Replace with your actual image path
# rotation_angles = [0, 30, 45, 60, 90]
#
# # === FUNCTION TO ROTATE IMAGE AROUND CENTER ===
# def rotate_image(image_bgr, angle):
#     (h, w) = image_bgr.shape[:2]
#     center = (w // 2, h // 2)
#
#     # Compute rotation matrix and new bounding dimensions
#     rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
#     cos = np.abs(rot_matrix[0, 0])
#     sin = np.abs(rot_matrix[0, 1])
#     new_w = int((h * sin) + (w * cos))
#     new_h = int((h * cos) + (w * sin))
#
#     # Adjust the rotation matrix for translation
#     rot_matrix[0, 2] += (new_w / 2) - center[0]
#     rot_matrix[1, 2] += (new_h / 2) - center[1]
#
#     # Perform the actual rotation
#     rotated = cv2.warpAffine(image_bgr, rot_matrix, (new_w, new_h))
#     return cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
#
# # === LOAD IMAGE ===
# image_bgr = cv2.imread(image_path)
# if image_bgr is None:
#     raise FileNotFoundError(f"Image not found at: {image_path}")
#
# # === ROTATE AND COLLECT RESULTS ===
# rotated_images = []
# for angle in rotation_angles:
#     rotated_img = rotate_image(image_bgr, angle)
#     rotated_images.append((rotated_img, angle))
#
# # === PLOT ===
# fig, axes = plt.subplots(1, len(rotation_angles), figsize=(30, 10))
# fig.suptitle("Effect of Image Rotation", fontsize=22, weight='bold')
#
# for ax, (img, angle) in zip(axes, rotated_images):
#     ax.imshow(img)
#     ax.axis('off')
#     ax.text(0.5, -0.1, f"Angle = {angle}°", fontsize=18, ha='center', transform=ax.transAxes)
#
# plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.05, wspace=0.02)
# plt.show()

# ### SCALE
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # === CONFIGURATION ===
# image_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017\000000007386.jpg"  # <-- Replace with your actual image path
# scale_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
#
# # === FUNCTION TO SCALE IMAGE ===
# def scale_image(image_bgr, scale_factor):
#     height = int(image_bgr.shape[0] * scale_factor)
#     width = int(image_bgr.shape[1] * scale_factor)
#     resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
#     return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#
# # === LOAD IMAGE ===
# image_bgr = cv2.imread(image_path)
# if image_bgr is None:
#     raise FileNotFoundError(f"Image not found at: {image_path}")
#
# # === SCALE AND COLLECT IMAGES ===
# scaled_images = []
# for scale in scale_factors:
#     scaled_img = scale_image(image_bgr, scale)
#     scaled_images.append((scaled_img, scale))
#
# # === PLOT (LARGE IMAGES, MULTI-ROW FOR MANY SCALES) ===
# cols = 5
# rows = (len(scaled_images) + cols - 1) // cols
# fig, axes = plt.subplots(rows, cols, figsize=(30, 6 * rows))
# axes = axes.flatten()
#
# for i in range(len(axes)):
#     ax = axes[i]
#     if i < len(scaled_images):
#         img, scale = scaled_images[i]
#         ax.imshow(img)
#         ax.axis('off')
#         ax.text(0.5, -0.1, f"Scale = {scale}", fontsize=16, ha='center', transform=ax.transAxes)
#     else:
#         ax.axis('off')
#
# fig.suptitle("Effect of Image Scaling", fontsize=26, weight='bold')
# plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05, wspace=0.05, hspace=0.25)
# plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIGURATION ===
image_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017\000000007386.jpg"  # <-- Replace with your image path
rotation_angles = [0, 30, 45, 60, 90]

# === FUNCTION TO ROTATE IMAGE ===
def rotate_image_keep_size(image_bgr, angle, target_size):
    (h, w) = image_bgr.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rot_matrix[0, 0])
    sin = np.abs(rot_matrix[0, 1])

    # Compute new size to avoid cropping
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix for translation
    rot_matrix[0, 2] += (new_w / 2) - center[0]
    rot_matrix[1, 2] += (new_h / 2) - center[1]

    # Rotate with large canvas
    rotated = cv2.warpAffine(image_bgr, rot_matrix, (new_w, new_h))

    # Resize back to original target size
    resized = cv2.resize(rotated, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

# === LOAD ORIGINAL IMAGE ===
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")
original_size = image_bgr.shape[:2]  # (height, width)

# === ROTATE AND RESIZE BACK TO ORIGINAL SIZE ===
rotated_images = []
for angle in rotation_angles:
    rotated_img = rotate_image_keep_size(image_bgr, angle, original_size)
    rotated_images.append((rotated_img, angle))

# === PLOT ===
fig, axes = plt.subplots(1, len(rotation_angles), figsize=(30, 10))
fig.suptitle("Effect of Image Rotation (Same Output Size)", fontsize=22, weight='bold')

for ax, (img, angle) in zip(axes, rotated_images):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.5, -0.1, f"Angle = {angle}°", fontsize=18, ha='center', transform=ax.transAxes)

plt.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.05, wspace=0.02)
plt.show()
