import csv
import os
import random
import numpy as np
from PIL import Image

from funcs import convert_class_SG, convert_class_MjT, plot_info
from model import SSD_Model
from default_box import CellBox

classes = [
            "inget", "baksida",
            "bambu_1", "bambu_2", "bambu_3", "bambu_4", "bambu_5", "bambu_6", "bambu_7", "bambu_8", "bambu_9",
            "cirkel_1", "cirkel_2", "cirkel_3", "cirkel_4", "cirkel_5", "cirkel_6", "cirkel_7", "cirkel_8", "cirkel_9",
            "tecken_1", "tecken_2", "tecken_3", "tecken_4", "tecken_5", "tecken_6", "tecken_7", "tecken_8", "tecken_9",
            "vind_nordan", "vind_sunnan", "vind_västan", "vind_östan",
            "drake_grön", "drake_röd", "drake_vit",
            "blomma_norr", "blomma_syd", "blomma_väst", "blomma_öst"
           ]
class_amount = len(classes)

input_shape = (300, 300, 3)
max_output = 100
batch_size = 256
training_iterations = 10000

def preprocess_image(path):
    img = Image.open(path)
    img = img.resize(input_shape[:-1])
    img = img.convert("RGB")
    img = np.moveaxis(np.array(img, dtype="float32"), 0, 1)
    img /= 255.0

    return img


def prepare_dataset(model, paths, convert_classes):
    dataset = []
    for path, convert_class in zip(paths, convert_classes):
        with open(f"{path}/_annotations.csv", "r") as csvfile:
            render = csv.reader(csvfile)
            next(render)

            a = list(render)

            i = 0
            while i < len(a):
                b_boxes = []
                class_indices = []

                i2 = 0
                for img in a[i:]:
                    if img[0] != a[i][0]:
                        break

                    class_name = convert_class(img[3])
                    if class_name in classes:
                        b_boxes.append(CellBox(abs_coords=map(lambda coord: float(coord) / float(img[1 + int(coord) % 2]), img[4:])))

                        class_index = classes.index(class_name)
                        class_indices.append(class_index)

                    i2 += 1

                if b_boxes and class_indices:
                    img_path = os.path.join(path, a[i][0])

                    indices, boxes = model.match_boxes(b_boxes)
                    offsets = [[b_box.create_offset(box) for box in matched_boxes] for b_box, matched_boxes in zip(b_boxes, boxes)]

                    locations = np.zeros((len(model.default_boxes), 4))
                    confidences = np.zeros((len(model.default_boxes), class_amount), dtype="int32")
                    for index, offset, class_index in zip(indices, offsets, class_indices):
                        locations[index] = offset
                        confidences[index, class_index] = 1

                    data = [preprocess_image(img_path), locations, confidences]
                    dataset.append(data)

                i += i2

    return dataset


def retrain(model, dataset, iteration_amount, epochs=1):
    for i in range(iteration_amount):
        x, y_loc, y_conf = zip(*random.sample(dataset, batch_size))

        y = {"locations": np.array(y_loc, dtype="int32"), "confidences": np.array(y_conf)}
        model.train([np.array(x)], y, epochs)

        if not int(i  % (iteration_amount / 10)):
            print(f"Training iteration {i} completed!")


def inference(model, path):
    image = preprocess_image(path)

    found_classes, found_boxes, confs = model.get_preds(image)
    labeled_classes = np.array(classes)[found_classes]
    scaled_boxes = [CellBox(box) for box in found_boxes * np.tile(input_shape[:2], (len(found_boxes), 2))]

    class_infos = list(zip(labeled_classes, scaled_boxes, confs))

    return class_infos


def evaluate(model):
    pass


if __name__ == "__main__":
    model = SSD_Model(input_shape, class_amount, max_output=max_output)

    # dataset = prepare_dataset(model, ["datasets/SG-mahjong.v1i.tensorflow/train", "datasets/mahjong detection for MjT.v4-resize.tensorflow/train"], [convert_class_SG, convert_class_MjT])
    # retrain(model, dataset, training_iterations)

    # model.save_model("model")
    # model.plot_metrics()

    img_path = "datasets/SG-mahjong.v1i.tensorflow/test/local_sg_mahjong_with_animal_tiles_travel_size_1551811793_7e51a0d2_progressive_jpg.rf.0d5c4a5f78e69e58dc9788ae8d89e67d.jpg"
    infos = inference(model, img_path)

    img = Image.open(img_path).resize(input_shape[:-1])
    plot_info(img, infos)
