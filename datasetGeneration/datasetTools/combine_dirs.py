import files

input_dirs = [
    "ssd/dataset/data/dataset100",
    "ssd/dataset/data/dataset1"
]
output_dir = "ssd/dataset/data/dataset101"
combine_annotations = True

def combine_dirs(input_dirs, output_dir, combine_annotations):
    new_annotations = []

    files.create_path(output_dir)

    for dir_identifier, input_dir in enumerate(input_dirs):
        # dir_identifier = hash(input_dir)

        annotations = files.load_annotations(input_dir)
        for anno_i, annotation in enumerate(annotations):
            old_name = annotation["image"]
            new_name = f"{dir_identifier}_{annotation['image']}"

            old_path = files.join_paths(input_dir, old_name)
            new_path = files.join_paths(output_dir, new_name)
            files.copy_file(old_path, new_path)  # Copy whole directory instead?

            annotation["image"] = new_name

        if combine_annotations:
            new_annotations += annotations

    if combine_annotations:
        files.save_annotations(new_annotations, output_dir)


if __name__ == "__main__":
    combine_dirs(input_dirs, output_dir, combine_annotations)
