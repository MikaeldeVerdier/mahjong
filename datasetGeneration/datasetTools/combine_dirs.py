import os

import files

input_dirs = [
    "ssd/dataset/data/realDataset",
    "ssd/dataset/data/synthDataset"
]
output_dir = "/Users/mikaeldeverdier/mahjong/mahjong/ssd/dataset/data/hybridDataset"
combine_annotations = True

new_annotations = []

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for input_dir in input_dirs:
    dir_identifier = hash(input_dir)

    annotations = files.load(input_dir)
    for i, annotation in enumerate(annotations):
        annotations[i]["image"] = f"{dir_identifier}_{annotation['image']}"

    if combine_annotations:
        new_annotations += annotations

    for file in os.listdir(input_dir):
        if file == ".DS_Store":
            continue

        if not file.endswith(".createml.json"):
            old_path = os.path.join(input_dir, file)
            new_path = os.path.join(output_dir, f"{dir_identifier}_{file}")

            files.copy_file(old_path, new_path)

if combine_annotations:
    files.save(new_annotations, output_dir)
