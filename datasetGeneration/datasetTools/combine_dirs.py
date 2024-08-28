import os

import files

input_dirs = [
    "ssd/dataset/data/realDataset",
    "ssd/dataset/data/synthDataset"
]
output_dir = "ssd/dataset/data/hybridDataset"
combine_annotations = True

new_annotations = []

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for input_dir in input_dirs:
    dir_identifier = hash(input_dir)

    annotations = files.load(input_dir)
    for i, annotation in enumerate(annotations):
        old_name = annotation['image']
        new_name = f"{dir_identifier}_{annotation['image']}"

        old_path = os.path.join(input_dir, old_name)
        new_path = os.path.join(output_dir, new_name)

        files.copy_file(old_path, new_path)

        annotations[i]["image"] = new_name

    if combine_annotations:
        new_annotations += annotations

if combine_annotations:
    files.save(new_annotations, output_dir)
