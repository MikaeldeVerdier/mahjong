import os

import files

input_dirs = [
    "/path/to/input_dir1",
    "/path/to/input_dir2"
]
output_dir = "/path/to/output_dir"
combine_annotations = False

annotations = []

for input_dir in input_dirs:
    dir_identifier = hash(input_dir)

    if combine_annotations:
        annotations += files.load(input_dir)

    for file in os.listdir(input_dir):
        if file == ".DS_Store":
            continue

        if not file.endswith(".createml.json"):
            files.copy_file(file, input_dir, output_dir)

if combine_annotations:
    files.save(annotations, output_dir)
