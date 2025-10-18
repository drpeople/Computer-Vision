## COCO DATASET
# import os
# from collections import OrderedDict
#
# import matplotlib.pyplot as plt
# from pycocotools.coco import COCO
#
# # === USER‐CONFIGURATION ===
# val_images_path = r"C:\Users\goker\PycharmProjects\DiplomProject\val2017"
# annotation_path   = r"C:\Users\goker\PycharmProjects\DiplomProject\instances_val2017.json"
# # ===========================
#
# # Load COCO annotations
# coco = COCO(annotation_path)
#
# # Total number of images in the dataset
# all_img_ids = coco.getImgIds()
# total_images = len(all_img_ids)
#
# # All category IDs and names
# cat_ids    = coco.getCatIds()
# num_classes = len(cat_ids)
# cat_info  = coco.loadCats(cat_ids)
# cat_names = [c['name'] for c in cat_info]
#
# # Count images per class (an image is counted once per category if it contains ≥1 instance)
# images_per_class = OrderedDict()
# for cid, cname in zip(cat_ids, cat_names):
#     img_ids = coco.getImgIds(catIds=cid)
#     images_per_class[cname] = len(set(img_ids))
#
# # --- Print summary ---
# print(f"Total images in dataset: {total_images}")
# print(f"Number of classes:     {num_classes}\n")
#
# print("Images per class:")
# for cname, count in images_per_class.items():
#     print(f"  - {cname:<15} : {count}")
#
# # --- Plot bar chart ---
# plt.figure(figsize=(12, 6))
# plt.bar(images_per_class.keys(), images_per_class.values())
# plt.xticks(rotation=90, fontsize=10)
# plt.ylabel("Number of images", fontsize=12)
# plt.title("COCO val2017: Images per Class", fontsize=14)
# plt.tight_layout()
# plt.show()


## IMAGENET
# import os
# from collections import Counter, OrderedDict
#
# import matplotlib.pyplot as plt
# from PIL import Image
# from torch.utils.data import Dataset
#
# # === USER CONFIGURATION ===
# base_folder = r"C:\Users\goker\PycharmProjects\DiplomProject\ILSVRC2012_img_val_subset"
# # Load human-readable ImageNet labels
# # (same as in your main script)
# import requests
# imagenet_classes = requests.get(
#     "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# ).text.splitlines()
# # ==========================
#
# class ImageDataset(Dataset):
#     def __init__(self, base_folder, transform=None):
#         self.image_paths = []
#         self.labels      = []
#         self.transform   = transform
#         for subfolder in os.listdir(base_folder):
#             subfolder_path = os.path.join(base_folder, subfolder)
#             if not os.path.isdir(subfolder_path):
#                 continue
#             # folder name is the integer label
#             label_idx = int(subfolder)
#             for img_name in os.listdir(subfolder_path):
#                 if img_name.lower().endswith(('.jpg','jpeg','png')):
#                     self.image_paths.append(os.path.join(subfolder_path, img_name))
#                     self.labels.append(label_idx)
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         img = Image.open(self.image_paths[idx]).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, self.labels[idx]
#
# def summarize_dataset(base_folder):
#     # instantiate without transforms
#     ds = ImageDataset(base_folder, transform=None)
#
#     total_images = len(ds)
#     label_counts = Counter(ds.labels)
#
#     num_classes = len(label_counts)
#
#     # sort by label idx for consistent ordering
#     images_per_class = OrderedDict(
#         (imagenet_classes[label], count)
#         for label, count in sorted(label_counts.items())
#     )
#
#     # --- print summary ---
#     print(f"Total images in dataset: {total_images}")
#     print(f"Number of classes:       {num_classes}\n")
#     print("Images per class:")
#     for cls_name, cnt in images_per_class.items():
#         print(f"  - {cls_name:<30} : {cnt}")
#
#     # --- plot bar chart ---
#     plt.figure(figsize=(10, 8))
#     plt.barh(
#         list(images_per_class.keys()),
#         list(images_per_class.values()),
#         edgecolor='black'
#     )
#     plt.xlabel("Number of images", fontsize=12)
#     plt.title("Images per Class in ILSVRC2012 Subset", fontsize=14)
#     plt.tight_layout()
#     plt.show()
#
# if __name__ == "__main__":
#     summarize_dataset(base_folder)

## PASCAL VOC
# import os
# from collections import Counter, OrderedDict
#
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
#
# # === USER CONFIGURATION ===
# VOC_ROOT     = r"C:\Users\goker\PycharmProjects\DiplomProject\voc"
# IMAGES_DIR   = os.path.join(VOC_ROOT, "JPEGImages")
# MASKS_DIR    = os.path.join(VOC_ROOT, "SegmentationClass")
# VAL_TXT_PATH = os.path.join(VOC_ROOT, "ImageSets", "Segmentation", "val.txt")
# # ==========================
#
# # Pascal VOC 21 class names (0=background, 1–20 are object classes)
# VOC_CLASSES = [
#     "background", "aeroplane", "bicycle", "bird", "boat",
#     "bottle",     "bus",       "car",     "cat",  "chair",
#     "cow",        "diningtable","dog",    "horse","motorbike",
#     "person",     "pottedplant","sheep",  "sofa", "train",
#     "tvmonitor"
# ]
#
# def load_val_ids(txt_path):
#     with open(txt_path, "r") as f:
#         return [line.strip() for line in f if line.strip()]
#
# def count_images_per_class(val_ids, masks_dir):
#     # counter[class_id] = number of images where class_id appears
#     counter = Counter()
#     for img_id in val_ids:
#         mask_path = os.path.join(masks_dir, img_id + ".png")
#         mask = np.array(Image.open(mask_path), dtype=np.int32)
#         # find unique labels, ignore void label 255
#         labels = np.unique(mask)
#         valid = labels[(labels >= 0) & (labels <= 20)]
#         for cls in valid:
#             counter[int(cls)] += 1
#     return counter
#
# def summarize_and_plot():
#     # 1) load val split IDs
#     val_ids = load_val_ids(VAL_TXT_PATH)
#     total_images = len(val_ids)
#
#     # 2) count images per class
#     images_per_cls = count_images_per_class(val_ids, MASKS_DIR)
#
#     # 3) compute how many classes actually appear
#     classes_present = sorted(images_per_cls.keys())
#     num_classes_present = len(classes_present)
#
#     # 4) build ordered mapping name -> count (include zeros if never appears)
#     ordered = OrderedDict()
#     for idx, name in enumerate(VOC_CLASSES):
#         ordered[name] = images_per_cls.get(idx, 0)
#
#     # --- print summary ---
#     print(f"Total images in validation split: {total_images}")
#     print(f"Number of VOC classes present:    {num_classes_present} / {len(VOC_CLASSES)}\n")
#
#     print("Images per class:")
#     for cls_name, cnt in ordered.items():
#         print(f"  - {cls_name:<15} : {cnt}")
#
#     # --- plot bar chart ---
#     plt.figure(figsize=(12, 6))
#     plt.bar(ordered.keys(), ordered.values(), edgecolor="black")
#     plt.xticks(rotation=90)
#     plt.ylabel("Number of images")
#     plt.title("Pascal VOC val: Images per Class")
#     plt.tight_layout()
#     plt.show()
#
# if __name__ == "__main__":
#     summarize_and_plot()


## Leaf
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
#
# # === USER CONFIGURATION ===
# test_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
# # ==========================
#
# # 1) Gather all mask filenames
# mask_files = [
#     fn for fn in os.listdir(test_masks_dir)
#     if fn.lower().endswith(('.png', '.jpg', '.jpeg'))
# ]
# total_images = len(mask_files)
#
# # 2) Define classes
# classes = {
#     0: "background",
#     1: "leaf"
# }
#
# # 3) Count images containing each class
# counts = {name: 0 for name in classes.values()}
#
# for mask_fn in mask_files:
#     mask_path = os.path.join(test_masks_dir, mask_fn)
#     mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
#     # leaf pixels assumed > 0
#     has_leaf = np.any(mask > 0)
#     has_bg   = np.any(mask == 0)
#     if has_bg:
#         counts["background"] += 1
#     if has_leaf:
#         counts["leaf"] += 1
#
# # 4) Print summary
# print(f"Total images in test set: {total_images}")
# print(f"Number of classes:         {len(classes)}\n")
#
# print("Images per class:")
# for cls_name, cnt in counts.items():
#     print(f"  - {cls_name:<10} : {cnt}")
#
# # 5) Plot bar chart
# plt.figure(figsize=(6, 4))
# plt.bar(counts.keys(), counts.values(), edgecolor="black")
# plt.ylabel("Number of images")
# plt.title("Leaf Segmentation Test Set: Images per Class")
# plt.tight_layout()
# plt.show()

## LEAF TRAINING
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === USER CONFIGURATION ===
train_masks_dir = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\data\masks"
val_masks_dir   = r"C:\Users\goker\PycharmProjects\DiplomProject\leaf\test\masks"
# test_masks_dir = r"...\leaf\test\masks"  # if you have a separate test split
# ==========================

CLASSES = {
    0: "background",
    1: "leaf"
}

def summarize_split(split_name, masks_dir, threshold=0):
    """
    Prints and plots:
      - total images in `masks_dir`
      - number of classes (always 2 here)
      - images per class, where a class counts if its pixels > threshold
    """
    mask_files = [
        fn for fn in os.listdir(masks_dir)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    total_images = len(mask_files)
    counts = {name: 0 for name in CLASSES.values()}

    for fn in mask_files:
        mask = np.array(
            Image.open(os.path.join(masks_dir, fn)).convert("L"),
            dtype=np.uint8
        )
        has_leaf = np.any(mask > threshold)
        has_bg   = np.any(mask == 0)

        if has_bg:
            counts["background"] += 1
        if has_leaf:
            counts["leaf"] += 1

    print(f"\n=== {split_name} Split ===")
    print(f"Total images:   {total_images}")
    print(f"Number of classes: {len(CLASSES)}\n")
    print("Images per class:")
    for cls, cnt in counts.items():
        print(f"  - {cls:<10} : {cnt}")

    # Bar chart
    plt.figure(figsize=(5, 4))
    plt.bar(counts.keys(), counts.values(), edgecolor="black")
    plt.title(f"{split_name}: Images per Class")
    plt.ylabel("Number of images")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    summarize_split("Train", train_masks_dir)
    summarize_split("Validation", val_masks_dir)
    # summarize_split("Test", test_masks_dir)  # uncomment if you have a separate test set
