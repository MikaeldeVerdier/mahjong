import csv
import os
import random
import numpy as np
from PIL import Image

import config
from funcs import convert_class_SG, convert_class_MjT, plot_infos
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

def preprocess_image(path):
    img = Image.open(path)
    img = img.resize(input_shape[:-1])
    img = img.convert("RGB")
    img = np.moveaxis(np.array(img, dtype="float32"), 0, 1)
    img /= 255.0

    return img


def prepare_training(model, image, gt_boxes, class_indices):
    image = preprocess_image(image)

    pos_indices = model.match_boxes(gt_boxes)

    locations = np.zeros((len(model.default_boxes), 4))
    confidences = np.zeros((len(model.default_boxes), class_amount), dtype="int32")

    for pos_index, gt_match in pos_indices:
        offset = gt_boxes[gt_match].calculate_offset(model.default_boxes[pos_index])

        locations[pos_index] = offset
        confidences[pos_index, class_indices[gt_match]] = 1

    mask = np.ones(len(confidences), dtype="bool")
    mask[np.array(pos_indices)[:, 0]] = False
    confidences[mask, 0] = 1

    return image, locations, confidences


def prepare_dataset(model, paths, convert_classes, training=False):
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
                        locations.append(CellBox(abs_coords=map(lambda coord: float(coord) / float(img[1 + int(coord) % 2]), img[4:])))

                        class_index = classes.index(class_name)
                        confidences.append(class_index)

                    i2 += 1

                if locations and confidences:
                    image = os.path.join(path, a[i][0])

                    if training:
                        image, locations, confidences = prepare_training(model, image, locations, confidences)

                    data = [image, locations, confidences]
                    dataset.append(data)

                i += i2

    return dataset


def retrain(model, dataset, iteration_amount, epochs):
    for i in range(iteration_amount):
        x, y_loc, y_conf = zip(*random.sample(dataset, config.BATCH_SIZE))

        x = np.array(x)
        y = {"locations": np.array(y_loc), "confidences": np.array(y_conf)}
        model.train(x, y, epochs)

        if not int(i  % (iteration_amount / 10)):
            print(f"Training iteration {i} completed!")
        
        if not int(i % config.SAVING_FREQUENCY):
            model.save_model("model")


def inference(model, path, conf_threshold=0.1):
    image = preprocess_image(path)

    found_classes, found_boxes, confs = model.get_preds(image, conf_threshold=conf_threshold)
    labeled_classes = np.array(classes)[found_classes]
    scaled_boxes = [CellBox(abs_coords=box).scale_box(input_shape[:-1]) for box in found_boxes]

    class_infos = list(zip(labeled_classes, scaled_boxes, confs))

    return class_infos


def evaluate(model, dataset, iou_threshold=0.5, conf_threshold=0.5):
    amount_true_pos = 0
    amount_false_pos = 0
    amount_false_neg = 0

    for img, gt_boxes, gt_classes in dataset:
        iteration_true_pos = 0

        gt_boxes = [gt_box.scale_box(input_shape[:-1]) for gt_box in gt_boxes]

        class_infos = inference(model, img, conf_threshold=conf_threshold)

        for pred_class, box, _ in class_infos:
            if any(box.calculate_iou(gt_box) >= iou_threshold and classes[gt_class] == pred_class for gt_class, gt_box in zip(gt_classes, gt_boxes)):
                iteration_true_pos += 1
            else:
                amount_false_pos += 1
        
        amount_false_neg += len(gt_boxes) - iteration_true_pos
        amount_true_pos += iteration_true_pos

    metric_value = amount_true_pos / (amount_true_pos + amount_false_pos + amount_false_neg)

    return metric_value


if __name__ == "__main__":
    model = SSD_Model(input_shape, class_amount)

    training_dataset = prepare_dataset(model, ["datasets/SG-mahjong.v1i.tensorflow/train"], [convert_class_SG], training=True)
    retrain(model, training_dataset, config.TRAINING_ITERATIONS, config.EPOCHS)

    model.save_model("model")
    model.plot_metrics()

    testing_dataset = prepare_dataset(model, ["datasets/SG-mahjong.v1i.tensorflow/test"], [convert_class_SG])
    metric_value = evaluate(model, testing_dataset)
    print(f"The model got a score of {metric_value}!")

    # img_path = "datasets/SG-mahjong.v1i.tensorflow/test/local_sg_mahjong_with_animal_tiles_travel_size_1551811793_7e51a0d2_progressive_jpg.rf.0d5c4a5f78e69e58dc9788ae8d89e67d.jpg"
    # infos = inference(model, img_path)

    # img = Image.open(img_path).resize(input_shape[:-1])
    # plot_infos(img, infos)
    pass
