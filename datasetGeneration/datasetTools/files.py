import os
import shutil
import json

def load(input_dir):
    with open(os.path.join(input_dir, "_annotations.createml.json"), "r") as json_file:
        return json.load(json_file)
    

def save(annotations, output_dir):
    with open(os.path.join(output_dir, "_annotations.createml.json"), "w") as json_file:
        json.dump(annotations, json_file)


def create_path(path):
        dirs = path.split(os.sep)

        cur_path = ""
        for dir in dirs:
            cur_path = os.path.join(cur_path, dir)
            if not os.path.exists(cur_path):
                os.mkdir(cur_path)


def create_file(path, content):
     with open(path, "w") as file:
        file.write(content)


def copy_file(old_path, new_path):
    shutil.copy(old_path, new_path)
