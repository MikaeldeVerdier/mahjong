import os
import json

def load(input_dir):
    with open(os.path.join(input_dir, "_annotations.createml.json"), "r") as json_file:
        return json.load(json_file)
    

def save(annotations, output_dir):
    with open(os.path.join(output_dir, "_annotations.createml.json"), "w") as json_file:
        json.dump(annotations, json_file)


"""
def move_file(name, input_dir, output_dir):
    input_path = os.path.join(input_dir, name)
    img = Image.open(input_path)

    new_path = os.path.join(output_dir, name)
    img.save(new_path)
"""
