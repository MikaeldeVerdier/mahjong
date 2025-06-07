import os

import files

input_dir = "yolo/dataset/data/dataset100/train"
output_dir = "yolo/dataset/data/dataset100/train_yolo"

labels = [
    "Bamboo 1", "Bamboo 2", "Bamboo 3", "Bamboo 4", "Bamboo 5", "Bamboo 6", "Bamboo 7", "Bamboo 8", "Bamboo 9",
    "Dot 1", "Dot 2", "Dot 3", "Dot 4", "Dot 5", "Dot 6", "Dot 7", "Dot 8", "Dot 9",
    "Character 1", "Character 2", "Character 3", "Character 4", "Character 5", "Character 6", "Character 7", "Character 8", "Character 9",
    "East Wind", "South Wind", "West Wind", "North Wind",
    "Red Dragon", "Green Dragon", "White Dragon",
    "East Flower", "South Flower", "West Flower", "North Flower",
    "East Season", "South Season", "West Season", "North Season",
    "Back"
]


def convert_to_yolo(input_dir, output_dir, labels):
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    if not os.path.exists(images_dir):
        files.create_path(images_dir)
    if not os.path.exists(labels_dir):
        files.create_path(labels_dir)

    annotations = files.load(input_dir)
    for i, annotation in enumerate(annotations):
        annotation_text = ""
        for anno in annotation["annotations"]:
            label_id = labels.index(anno["label"])
            cx, cy, w, h = anno["coordinates"].values()

            annotation_text += f"{label_id} {cx} {cy} {w} {h}\n"

        annotation_text = annotation_text.strip()
        image_name = annotation["image"]

        new_name = ".".join(image_name.split(".")[:-1]) + ".txt"
        new_path = os.path.join(labels_dir, new_name)
        files.create_file(new_path, annotation_text)

        old_image_path = os.path.join(input_dir, image_name)
        new_image_path = os.path.join(images_dir, image_name)
        files.copy_file(old_image_path, new_image_path)


if __name__ == "__main__":
    convert_to_yolo(input_dir, output_dir, labels)
