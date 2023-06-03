import matplotlib.pyplot as plt
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


# Incorrect
def plot_b_boxes(img, b_boxes):
    _, ax = plt.subplots()

    ax.imshow(img)

    for b_box in b_boxes:
        ax.add_patch(Rectangle((b_box.abs_coords[0], b_box.abs_coords[1]), b_box.size_coords[2], b_box.size_coords[3]))

    plt.savefig("box.png", dpi=200)
