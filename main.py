import csv
import os
import random
import numpy as np
from PIL import Image

from funcs import convert_class_SG, convert_class_MjT
from model import NeuralNetwork

input_shape = (300, 300, 3)
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
max_output = 532
batch_size = 256
training_iterations = 10

def preprocess_image(path):
    img = Image.open(path)
    img = img.resize(input_shape[:-1])
    img = img.convert("RGB")
    img = np.moveaxis(np.array(img, dtype="float32"), 0, 1)
    img /= 255.0

    return img


def prepare_dataset(paths, convert_classes):
    dataset = []

    for path, convert_class in zip(paths, convert_classes):
        with open(f"{path}/_annotations.csv", "r") as csvfile:
            render = csv.reader(csvfile)
            next(render)

            a = list(render)

            i = 0
            while i < len(a):
                locations = []
                confidences = []

                i2 = 0
                for img in a[i:]:
                    if img[0] != a[i][0]:
                        break

                    class_name = convert_class(img[3])
                    if class_name in classes:
                        locations.append(img[4:])

                        class_index = classes.index(class_name)
                        confidences.append(np.eye(class_amount)[class_index])

                    i2 += 1

                img_path = os.path.join(path, a[i][0])

                locations += [[0] * 4] * (max_output - len(locations))
                confidences += [[0] * class_amount] * (max_output - len(confidences))
                data = [preprocess_image(img_path), locations, confidences]
                dataset.append(data)

                i += i2

    return dataset


def retrain_network(nn, dataset, iteration_amount, epochs=1):
    for i in range(iteration_amount):
        x, y_loc, y_conf = zip(*random.sample(dataset, batch_size))

        y = {"locations": np.array(y_loc, dtype="int32"), "confidences": np.array(y_conf)}
        nn.train([np.array(x)], y, epochs)

        if not int(i  % (iteration_amount / 10)):
            print(f"Training iteration {i} completed!")


if __name__ == "__main__":
    nn = NeuralNetwork(input_shape, class_amount, max_output=max_output)

    dataset = prepare_dataset(["datasets/dataset1", "datasets/dataset2"], [convert_class_SG, convert_class_MjT])
    retrain_network(nn, dataset, training_iterations)

    nn.save_model("model")
    nn.plot_metrics()

    # Inference:

    # image = preprocess_image("/Users/mikaeldeverdier/mahjong/dataset new no_green/drake_vit/4cell_3_6.jpg")
    # found_classes, found_boxes, confs = nn.get_preds(image)
    # found_boxes = found_boxes.tolist()
    # labeled_classes = np.array(classes)[found_classes].tolist()

    # class_infos = list(zip(labeled_classes, found_boxes, confs))
