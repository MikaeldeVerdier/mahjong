import os
import shutil
import json

def join_paths(*args):  # alternative to os.path.join but only using forward slashes
    args = [arg.replace("\\", "/") for arg in args if len(arg)]  # replace backslashes with forward slashes
    sep = "/"  # os.sep

    return sep.join(args)


def load_annotations(input_dir):
    with open(join_paths(input_dir, "_annotations.createml.json"), "r") as json_file:
        return json.load(json_file)
    

def save_annotations(annotations, output_dir):
    with open(join_paths(output_dir, "_annotations.createml.json"), "w") as json_file:
        json.dump(annotations, json_file)


def create_path(path):
        sep = "/"  # os.sep
        dirs = path.split(sep)

        cur_path = ""
        for dir in dirs:
            cur_path = join_paths(cur_path, dir)
            if not os.path.exists(cur_path):
                os.mkdir(cur_path)


def create_file(path, content):
     with open(path, "w") as file:
        file.write(content)


def copy_file(old_path, new_path):
    shutil.copy(old_path, new_path)
