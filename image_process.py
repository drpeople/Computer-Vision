import os
def get_all_images(folder):
    """
    Recursively get all image file paths from the given folder.
    Args:
        folder (str): Path to the folder containing images.
    Returns:
        list: List of file paths to all images in the folder.
    """
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []

    for root, _, files in os.walk(folder):  # Recursively walk through directories
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths


# Example usage
#folder_path = r"C:\Users\goker\PycharmProjects\DiplomProject\sampled_images" # Replace with the path to your folder
folder_path = r"C:\Users\goker\PycharmProjects\DiplomProject\ILSVRC2012_img_val_subset" # Replace with the path to your folder
image_list = get_all_images(folder_path)

print(f"Found {len(image_list)} images.")
print("First 5 images:", image_list[:5])  # Display first 5 images