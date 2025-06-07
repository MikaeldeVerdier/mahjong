import os
from PIL import Image

import files

input_dir = "ssd/dataset/data/dataset100"
output_dir = "yolo/dataset/data/dataset100"


def normalize_boxes(input_dir, output_dir):
    if not os.path.exists(output_dir):
        files.create_path(output_dir)

    annotations = files.load(input_dir)
    for i, annotation in enumerate(annotations):
        image_name = annotation["image"]

        old_path = os.path.join(input_dir, image_name)
        new_path = os.path.join(output_dir, image_name)
        files.copy_file(old_path, new_path)

        img = Image.open(os.path.join(input_dir, annotation["image"]))
        width, height = img.size

        for anno in annotation["annotations"]:
            anno["coordinates"]["x"] /= width
            anno["coordinates"]["y"] /= height
            anno["coordinates"]["width"] /= width
            anno["coordinates"]["height"] /= height

    files.save(annotations, output_dir)


if __name__ == "__main__":
    normalize_boxes(input_dir, output_dir)
