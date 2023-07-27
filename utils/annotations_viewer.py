import os
import random
from PIL import Image
import matplotlib.pyplot as plt


def plot_bounding_boxes(data_directory, output_directory, num_images=5):
    image_files = [f for f in os.listdir(data_directory) if f.endswith(".PNG")]
    random.shuffle(image_files)
    image_files = image_files[:num_images]

    for image_file in image_files:
        image_path = os.path.join(data_directory, image_file)
        annotation_path = os.path.join(data_directory, os.path.splitext(image_file)[0] + ".txt")

        image = Image.open(image_path)
        image_width, image_height = image.size

        with open(annotation_path, 'r') as f:
            lines = f.read().splitlines()

        plt.figure()
        plt.imshow(image)

        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.split())

            x_min = (x_center - width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            x_max = (x_center + width / 2) * image_width
            y_max = (y_center + height / 2) * image_height

            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              linewidth=2, edgecolor='r', facecolor='none'))
            plt.text(x_min, y_min - 2, f"Class: {int(class_id)}", fontsize=10, color='r')

        plt.axis('off')
        save_path = os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_bbox.jpg")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == "__main__":
    data_directory = "/home/deltlo36/PycharmProjects/WasteDetection/datasets/zerowaste/images/augment_train"
    output_directory = "/home/deltlo36/PycharmProjects/WasteDetection/utils/tempdir"

    os.makedirs(output_directory, exist_ok=True)
    plot_bounding_boxes(data_directory, output_directory, num_images=5)
