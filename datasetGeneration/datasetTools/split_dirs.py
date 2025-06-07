import os

import files

input_dir = "yolo/dataset/data/dataset100"
output_dirs = [
    "yolo/dataset/data/dataset100/train",
    "yolo/dataset/data/dataset100/val"
]
split = [0.9, 0.1]
split_annotations = True


def split_dirs(input_dir, output_dirs, split, split_annotations):
    annotations = files.load(input_dir)

    for i, output_dir in enumerate(output_dirs):
        if not os.path.exists(output_dir):
            files.create_path(output_dir)

        start_index = int(len(annotations) * sum(split[:i]))
        end_index = int(len(annotations) * sum(split[:i + 1]))
        split_annotations = annotations[start_index:end_index]

        for i, annotation in enumerate(split_annotations):
            image_name = annotation["image"]

            old_path = os.path.join(input_dir, image_name)
            new_path = os.path.join(output_dir, image_name)
            files.copy_file(old_path, new_path)

        if split_annotations:
            files.save(split_annotations, output_dir)


if __name__ == "__main__":
    split_dirs(input_dir, output_dirs, split, split_annotations)
