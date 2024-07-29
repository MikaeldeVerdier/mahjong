import os

import files

input_dirs = [
    "/path/to/input_dir1",
    "/path/to/input_dir2"
]
output_dir = "/path/to/output_dir"
combine_annotations = False

new_annotations = []

for input_dir in input_dirs:
    dir_identifier = hash(input_dir)

    annotations = files.load(input_dir)
    for i, annotation in enumerate(annotations):
        annotations[i]['image'] = f"{dir_identifier}_{annotation['image']}"

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
