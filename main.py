import os
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing

def main():
    # Function to visualize bounding boxes on an image
    import os
    def visualize_bounding_boxes(image_path, annotation_path, class_labels):
        image = cv2.imread(image_path)
        try:
            annotations = pd.read_csv(annotation_path, delimiter=' ', header=None)
        except pd.errors.EmptyDataError:
            print(f'{annotation_path}, " is empty')

        for _, row in annotations.iterrows():
            class_label = row[0]
            class_label = class_labels[int(class_label)]
            x_center, y_center, width, height = row[1:].values
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (218, 144, 38), 2)
            cv2.putText(image, str(class_label), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (239, 13, 23), 2)

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    images_dir = r'C:\Users\goker\PycharmProjects\pythonProject1\train\images'
    annot_dir = r'C:\Users\goker\PycharmProjects\pythonProject1\train\labels'

    class_labels = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold', 'Target Spot',
                    'Black Spot']

    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    random_images = random.sample(image_files, 5)

    for image_file in random_images:
        image_path = os.path.join(images_dir, image_file)
        annotation_path = os.path.join(annot_dir, image_file.replace('.jpg', '.txt'))

        print("Image:", image_file)
        visualize_bounding_boxes(image_path, annotation_path, class_labels)

    class_labels = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold', 'Target Spot',
                    'Black Spot']

    labels_dir = r'C:\Users\goker\PycharmProjects\pythonProject1\train\labels'

    annotation_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    class_counts = {label: 0 for label in class_labels}
    empty_files = []

    for annotation_file in annotation_files:
        annotation_path = os.path.join(labels_dir, annotation_file)
        try:
            annotations = pd.read_csv(annotation_path, delimiter=' ', header=None)
        except pd.errors.EmptyDataError:
            empty_files.append(annotation_file)

        for class_label in annotations[0]:
            class_counts[class_labels[class_label]] += 1

    print("Label Distribution:")
    for class_label, count in class_counts.items():
        print(f"{class_label}: {count} instances")

    print(f'There are {len(empty_files)} empty files')

    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Label Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    from ultralytics import YOLO
    import os
    from IPython.display import display, Image
    from IPython import display

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')

    # Train the model using the 'yolov8n.pt' dataset for 300 epochs
    results = model.train(data=r'C:\Users\goker\PycharmProjects\pythonProject1\data.yaml', epochs=300, imgsz=640,batch=-1)
    #results = model.train(data=r'C:\Users\goker\PycharmProjects\pythonProject1\data.yaml', epochs=1, imgsz=640,batch=-1)

    import matplotlib.image as mpimg
    img = mpimg.imread(r'C:\Users\goker\PycharmProjects\pythonProject1\runs\detect\train16\F1_curve.png')
    imgplot = plt.imshow(img)
    plt.show()

    img = mpimg.imread(r'C:\Users\goker\PycharmProjects\pythonProject1\runs\detect\train16\results.png')
    imgplot = plt.imshow(img)
    plt.show()

    img = mpimg.imread(r'C:\Users\goker\PycharmProjects\pythonProject1\runs\detect\train16\confusion_matrix.png')
    imgplot = plt.imshow(img)
    plt.show()

    img = mpimg.imread(r'C:\Users\goker\PycharmProjects\pythonProject1\runs\detect\train16\val_batch0_pred.jpg')
    imgplot = plt.imshow(img)
    plt.show()


    model = YOLO(r'C:\Users\goker\PycharmProjects\pythonProject1\runs\detect\train16\weights\best.pt')
    #results = model.predict(source="0")
    results = model.predict(source=r'C:\Users\goker\PycharmProjects\pythonProject1\test\images',imgsz=640, conf = 0.4,save=True)

    from glob import glob
    from PIL import Image
    # for image_path in glob(r'C:\Users\goker\PycharmProjects\pythonProject1\runs\detect\predict\*.jpg')[10:20]:
    #     plt.imshow(Image.open(image_path));
    #     plt.axis("off");
    #     plt.show()
    #     print("\n")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

