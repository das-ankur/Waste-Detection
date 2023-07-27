import os
import shutil
from glob import glob
import cv2
import imgaug.augmenters as iaa
from tqdm import tqdm


def augment_image(image):
    # Define the augmentation sequence
    seq = iaa.Sequential([
        # Color Augmentations with reduced ranges and probabilities
        iaa.MultiplyBrightness((0.9, 1.1)),
        iaa.MultiplySaturation((0.9, 1.1)),
        iaa.ChangeColorTemperature((9000, 11000)),
        iaa.LinearContrast((0.9, 1.1), per_channel=True),
        iaa.AddToHueAndSaturation((-10, 10)),

        # Blur Augmentations with reduced ranges
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.AverageBlur(k=(2, 5)),
        iaa.MedianBlur(k=(3, 7)),

        # Noise Augmentations with reduced ranges
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
        iaa.AdditiveLaplaceNoise(scale=(0, 0.05 * 255)),
        iaa.SaltAndPepper(p=(0, 0.01)),
        iaa.CoarseSaltAndPepper(p=(0.005, 0.05), size_percent=(0.01, 0.05)),
    ])
    # Convert image to numpy array
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply augmentation
    augmented_image_np = seq(image=image_np)
    # Convert back to BGR format
    augmented_image = cv2.cvtColor(augmented_image_np, cv2.COLOR_RGB2BGR)
    return augmented_image


def augment():
    annotation_files = glob(input_annotation_dir + '/*.txt')
    for file in tqdm(annotation_files, total=len(annotation_files), desc="Augmeting...."):
        base_path, filename = os.path.split(file)
        base_name = filename[:-4]
        src_img = cv2.imread(os.path.join(input_image_dir, base_name + '.PNG'))
        with open(file, 'r') as fp:
            annotations = fp.readlines()
            count = 0
            for annotation in annotations:
                class_id, x, y, w, h = annotation.split()
                for i in range(oversampling_factor[class_id]):
                    aug_img = augment_image(src_img)
                    cv2.imwrite(os.path.join(output_image_dir, base_name + "_" + str(count) + '.PNG'), aug_img)
                    with open(os.path.join(output_annotation_dir, base_name + "_" + str(count) + '.txt'), 'w') as f:
                        f.write(annotation)
                    count += 1


input_image_dir = "/home/deltlo36/PycharmProjects/WasteDetection/datasets/zerowaste/images/train/"
input_annotation_dir = "/home/deltlo36/PycharmProjects/WasteDetection/datasets/zerowaste/labels/train/"
output_image_dir = "/home/deltlo36/PycharmProjects/WasteDetection/datasets/zerowaste/images/augment_train/"
output_annotation_dir = "/home/deltlo36/PycharmProjects/WasteDetection/datasets/zerowaste/labels/augment_train/"
oversampling_factor = {
    "0": 10,
    "1": 0,
    "2": 50,
    "3": 2
}
augment()
