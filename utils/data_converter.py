import json
import os


def convert_to_yolov5(categories, images, annotations, output_dir):
    for image_data in images:
        image_filename = image_data['file_name']
        image_width = image_data['width']
        image_height = image_data['height']

        image_annotations = [ann for ann in annotations if ann['image_id'] == image_data['id']]

        txt_filename = os.path.splitext(image_filename)[0] + '.txt'
        txt_filepath = os.path.join(output_dir, txt_filename)

        with open(txt_filepath, 'w') as f:
            for ann in image_annotations:
                category_id = ann['category_id']
                category = next((cat for cat in categories if cat['id'] == category_id), None)
                if category is None:
                    continue

                x, y, bbox_width, bbox_height = ann['bbox']
                x_center = (x + bbox_width / 2) / image_width
                y_center = (y + bbox_height / 2) / image_height
                width = bbox_width / image_width
                height = bbox_height / image_height

                line = f"{int(category['id']) - 1} {x_center} {y_center} {width} {height}\n"
                f.write(line)


if __name__ == "__main__":
    json_path = "/home/deltlo36/PycharmProjects/WasteDetection/raw_datasets/zerowaste/test/labels.json"
    with open(json_path, 'r') as fp:
        dataset = json.load(fp)

    output_directory = '/home/deltlo36/PycharmProjects/WasteDetection/datasets/zerowaste/test'
    os.makedirs(output_directory, exist_ok=True)
    convert_to_yolov5(dataset['categories'], dataset['images'], dataset['annotations'], output_directory)
