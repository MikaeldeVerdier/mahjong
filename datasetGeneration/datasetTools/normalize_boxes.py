from PIL import Image

import files

input_dir = "ssd/dataset/data/dataset100"
output_dir = "ssd/dataset/data/dataset100_normalized"

def normalize_boxes(input_dir, output_dir):
    files.create_path(output_dir)

    annotations = files.load_annotations(input_dir)
    for anno_i, annotation in enumerate(annotations):
        image_name = annotation["image"]

        old_path = files.join_paths(input_dir, image_name)
        new_path = files.join_paths(output_dir, image_name)
        files.copy_file(old_path, new_path)

        img = Image.open(files.join_paths(input_dir, annotation["image"]))
        width, height = img.size

        # width = 288
        # height = 512

        for anno in annotation["annotations"]:
            anno["coordinates"]["x"] /= width
            anno["coordinates"]["y"] /= height
            anno["coordinates"]["width"] /= width
            anno["coordinates"]["height"] /= height

    files.save_annotations(annotations, output_dir)


if __name__ == "__main__":
    normalize_boxes(input_dir, output_dir)
