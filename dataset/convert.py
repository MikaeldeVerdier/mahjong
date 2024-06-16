import sys
import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

sys.path.append(".")
from box_utils import convert_to_centroids

class Converter:
    def __init__(self, root):
        self.root = root

    def convert_to_createml(self, save_paths, divisions=["train", "val", "test"]):
        subsets = {}
        for subset in divisions:
            subset_layout_path = os.path.join(self.root, f"ImageSets/Main/{subset}.txt")

            if os.path.exists(subset_layout_path):
                with open(subset_layout_path, "r") as subset_layout_file:
                    subsets[subset] = subset_layout_file.read().split("\n")

        annotations = {division: [] for division in divisions}
        annotations_path = os.path.join(self.root, "Annotations")

        for annotation in os.listdir(annotations_path):
            path = os.path.join(annotations_path, annotation)
            root = ET.parse(path).getroot()

            # folder = root.find("folder").text
            img_name = root.find("filename").text
            names = [name.text for name in root.findall("object/name")]
            boxes = [{coord.tag: float(coord.text) for coord in bnd_box} for bnd_box in root.findall("object/bndbox")]
            boxes = np.array([[box["xmin"], box["ymin"], box["xmax"], box["ymax"]] for box in boxes])
            centroid_boxes = convert_to_centroids(boxes)  # Don't know why this abs is needed (w, h are negative otherwise)

            # from box_utils import plot_ious
            # plot_ious(centroid_boxes, np.empty(shape=(0, 4)), Image.open(os.path.join(self.root, "JPEGImages", img_name)), scale_coords=False)

            createml_annotation = {
                "image": img_name,
                "annotations": [
                    {
                        "label": label,
                        "coordinates": {
                            "x": coords[0],
                            "y": coords[1],
                            "w": coords[2],
                            "h": coords[3]
                        }
                    }
                for label, coords in zip(names, centroid_boxes)]
            }

            file_name = annotation.replace(".xml", "")
            for division in divisions:
                if division == "none" or file_name in subsets[division]:
                    annotations[division].append(createml_annotation)

        self.save_annotations(annotations, save_paths)

        return annotations
    
    def save_annotations(self, annotations_dict, dir_paths):
        for dir_path, annotations in zip(dir_paths, annotations_dict.values()):
            os_dir_path = os.path.join(os.getcwd(), dir_path)
            if not os.path.exists(os_dir_path):
                os.mkdir(os_dir_path)

            anontations_path = os.path.join(dir_path, "_annotations.createml.json")
            with open(anontations_path, "w") as json_file:
                json.dump(annotations, json_file)

            image_paths = [os.path.join(self.root, "JPEGImages", annotation["image"]) for annotation in annotations]
            for image_path in image_paths:
                img = Image.open(image_path)
                
                new_path = os.path.join(dir_path, image_path.split("/")[-1])
                img.save(new_path)

input_root = "VOCdevkit/VOC2012"
converter = Converter(input_root)

output_dirs = ["dataset/data/VOC2012_ALL"]
dataset = converter.convert_to_createml(output_dirs, divisions=["none"])
