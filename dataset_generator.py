import csv
import os
import shutil
import requests
import random

from PIL import Image

def download_file(url, save_path):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

    print("File downloaded successfully.")


def create_files(path, suits, iteratable):
    for suit in suits:
        suit_path = f"{path}{suit}"
        os.makedirs(suit_path, exist_ok=True)

        for i in iteratable:
            i_path = f"{path}{suit}/{suit[0]}{i}/"
            os.makedirs(i_path, exist_ok=True)


def setup_files(path):
    suits = ["cirkel", "bambu", "tecken"]
    create_files(path, suits, range(1, 10))

    dragons = ["drake"]
    create_files(path, dragons, ["röd", "grön", "vit"])

    winds = ["vind"]
    create_files(path, winds, ["östan", "sunnan", "västan", "nordan"])

    flowers = ["blomma"]
    create_files(path, flowers, ["öst", "syd", "väst", "norr"])

    back = ["baksida"]
    create_files(path, back, ["bak"])


def find_dir_path_substr(substring, directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if substring in file:
                return root


def split_image(url, path, grid_size):
    download_file(url, path)

    img = Image.open(path)
    width, height = img.size
    cell_width = width // grid_size[0]
    cell_height = height // grid_size[1]

    num = 13
    print(num)

    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            row = 3
            col = 8
            cell_image = img.crop((col * cell_width, row * cell_height, (col + 1) * cell_width, (row + 1) * cell_height))

            path = f"{find_dir_path_substr(f'{row}_{col}.jpg', 'dataset')}/{num}{row}_{col}.jpg"
            if not "None" in path:
                print(f"{row}_{col}")
                cell_image.save(path)


def from_annotations1(path, destination):
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

    with open(f"{path}/_annotations.csv", "r") as csvfile:
        render = csv.reader(csvfile)
        next(render)

        for row in render:
            split = row[3].split(" ")[::-1]
            if split[0] not in replacements:
                continue

            if "flower" in split[0]:
                split[1] = flower_dict[split[1]]
            tile = "_".join([replacements[word] if not word.isdigit() and word not in flower_dict.values() else word for word in split])
            shutil.move(f"{path}/{row[0]}", f"{destination}/{tile}/{row[0]}")


def from_annotations2(path, destination):
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

    with open(f"{path}/_annotations.csv", "r") as csvfile:
        render = csv.reader(csvfile)
        next(render)

        for row in render:
            split = list(row[3])[::-1]
            if row[0] not in os.listdir(path):
                continue

            if split[0] == "z":
                if int(split[1]) > 4:
                    split[0] = "drake"
                split[1] = z_dict[split[1]]
            tile = "_".join([replacements[word] if not word.isdigit() and word not in z_dict.values() else word for word in split])
            shutil.move(f"{path}/{row[0]}", f"{destination}/{tile}/{row[0]}")


def split_validation(path):
    new_path = path.replace("dataset", "validation")
    os.makedirs(new_path)
    subdirs = os.listdir(path)
    for subdir in subdirs:
        if subdir == ".DS_Store":
            continue

        new_subdir_path = os.path.join(new_path, subdir)
        if not os.path.exists(new_subdir_path):
            os.makedirs(new_subdir_path)

        subdir_path = os.path.join(path, subdir)
        files = os.listdir(subdir_path)

        for file in files:
            if ".jpg" in file:
                if random.random() > 0.8:
                    shutil.move(os.path.join(subdir_path, file), os.path.join(new_subdir_path, file))


if __name__ == "__main__":
    # path = "/Users/mikaeldeverdier/mahjong/dataset"
    # setup_files(path)

    # split_image("https://upload.wikimedia.org/wikipedia/commons/4/42/Mahjong_eg_Euro.jpg", "dataset/grid", (9, 5))
    # # from_annotations2("/Users/mikaeldeverdier/mahjong/mahjong-tiles.v7i.tensorflow/train", "/Users/mikaeldeverdier/mahjong/dataset new")

    # split_validation("/Users/mikaeldeverdier/mahjong/dataset new")

    pass
