import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from matplotlib.patches import Rectangle

def convert_class_MjT(class_name):
    replacements = {
        "s": "bambu",
        "p": "cirkel",
        "m": "tecken",
        "z": "vind"
    }
    z_dict = {
        "1": "östan",
        "2": "sunnan",
        "3": "västan",
        "4": "nordan",
        "5": "vit",
        "6": "grön",
        "7": "röd",
        None: "drake"
    }

    split = list(class_name)[::-1]
    if split[0] == "z":
        if int(split[1]) > 4:
            split[0] = "drake"
        split[1] = z_dict[split[1]]
    tile = "_".join([replacements[word] if not word.isdigit() and word not in z_dict.values() else word for word in split])

    return tile


def convert_class_SG(class_name):
    replacements = {
        "bamboo": "bambu",
        "dot": "cirkel",
        "character": "tecken",
        "wind": "vind",
        "east": "östan",
        "south": "sunnan",
        "west": "västan",
        "north": "nordan",
        "dragon": "drake",
        "red": "röd",
        "green": "grön",
        "white": "vit",
        "redflower": "blomma",
        "blueflower": "blomma"
    }
    flower_dict = {
        "1": "öst",
        "2": "syd",
        "3": "väst",
        "4": "norr"
    }

    split = class_name.split(" ")[::-1]

    if split[0] not in replacements:
        return None

    if "flower" in split[0]:
        split[1] = flower_dict[split[1]]
    tile = "_".join([replacements[word] if not word.isdigit() and word not in flower_dict.values() else word for word in split])

    return tile


def plot_infos(img, infos):
    _, ax = plt.subplots(1)

    ax.imshow(img)

    for info in infos:
        top_left_x, top_left_y = info[1].abs_coords[:2]
        width, height = info[1].size_coords[2:]

        rect = Rectangle((top_left_x, top_left_y), width, height, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

        ax.text(top_left_x + 5, top_left_y - 7, f"{info[0]} ({info[2]:.5f})", fontsize=5, backgroundcolor="y")

    plt.savefig("save_folder/box.png", dpi=200)
