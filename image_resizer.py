import os
import cv2
from tqdm import tqdm


def read_yolo_annotations(annot_file):
    with open(annot_file, 'r') as file:
        lines = file.readlines()
    annotations = [line.strip().split() for line in lines]
    return annotations


def resize_image(image_path, target_size):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, target_size)
    return resized_img


def resize_annotations(annotations, original_size, target_size):
    original_width, original_height = original_size
    target_width, target_height = target_size
    resized_annotations = []

    for class_id, x_center, y_center, width, height in annotations:
        x_center = float(x_center) * (target_width / original_width)
        y_center = float(y_center) * (target_height / original_height)
        width = float(width) * (target_width / original_width)
        height = float(height) * (target_height / original_height)
        resized_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    return resized_annotations


def save_yolo_annotations(resized_annotations, output_file):
    with open(output_file, 'w') as file:
        for annotation in resized_annotations:
            file.write(annotation + '\n')


def process_images_and_annotations(image_folder, annot_folder, output_image_folder, output_annot_folder, target_size):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_annot_folder, exist_ok=True)

    for image_filename in tqdm(os.listdir(image_folder), total=len(os.listdir(image_folder)),
                               desc="Resizing: "):
        if image_filename.endswith(('.jpg', '.PNG', '.jpeg')):
            image_path = os.path.join(image_folder, image_filename)
            annot_filename = os.path.splitext(image_filename)[0] + '.txt'
            annot_path = os.path.join(annot_folder, annot_filename)

            if os.path.isfile(annot_path):
                annotations = read_yolo_annotations(annot_path)
                img = resize_image(image_path, target_size)
                resized_annotations = resize_annotations(annotations, img.shape[:2], target_size)

                output_image_path = os.path.join(output_image_folder, image_filename)
                output_annot_path = os.path.join(output_annot_folder, annot_filename)

                cv2.imwrite(output_image_path, img)
                save_yolo_annotations(resized_annotations, output_annot_path)


# Example usage:
image_folder = '/home/deltlo36/PycharmProjects/WasteDetection/datasets/zerowaste/images/augment_train'
annot_folder = '/home/deltlo36/PycharmProjects/WasteDetection/datasets/zerowaste/labels/augment_train'
output_image_folder = '/home/deltlo36/PycharmProjects/WasteDetection/images'
output_annot_folder = '/home/deltlo36/PycharmProjects/WasteDetection/annotations'
target_size = (512, 512)  # Set your desired image size here.

process_images_and_annotations(image_folder, annot_folder, output_image_folder, output_annot_folder, target_size)
