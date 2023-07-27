import glob
import os


def change_file_extension(folder_path, old_extension, new_extension):
    image_files = glob.glob(os.path.join(folder_path, "**", f"*{old_extension}"))
    for old_file_path in image_files:
        new_file_path = old_file_path[:-len(old_extension)] + new_extension
        os.rename(old_file_path, new_file_path)


if __name__ == "__main__":
    folder_path = "/home/deltlo36/PycharmProjects/WasteDetection/datasets/zerowaste/images"
    old_extension = ".PNG"
    new_extension = ".png"

    change_file_extension(folder_path, old_extension, new_extension)
