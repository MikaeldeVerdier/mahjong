import os
import random
import numpy as np
from PIL import Image

import files
import config
from model import SSD_Model
from default_box import CellBox

labels = [
    "Bamboo 1", "Bamboo 2", "Bamboo 3", "Bamboo 4", "Bamboo 5", "Bamboo 6", "Bamboo 7", "Bamboo 8", "Bamboo 9",
    "Dot 1", "Dot 2", "Dot 3", "Dot 4", "Dot 5", "Dot 6", "Dot 7", "Dot 8", "Dot 9",
    "Character 1", "Character 2", "Character 3", "Character 4", "Character 5", "Character 6", "Character 7", "Character 8", "Character 9",
    "East Wind", "South Wind", "West Wind", "North Wind",
    "Red Dragon", "Green Dragon", "White Dragon",
    "East Flower", "South Flower", "West Flower", "North Flower",
    "East Season", "South Season", "West Season", "North Season",
    "Back"
]
label_amount = len(labels)

input_shape = (288, 512, 3)

def preprocess_image(path):
    img = Image.open(path)
    og_img_size = img.size

    img = img.resize(input_shape[:-1])
    img = img.convert("RGB")
    img = np.moveaxis(np.array(img, dtype="float32"), 0, 1)
    img /= 255.0

    return img, og_img_size


def prepare_training(model, gt_boxes, label_indices):
    pos_indices = model.match_boxes(gt_boxes)

    locations = np.zeros((len(model.default_boxes), 4))
    confidences = np.zeros((len(model.default_boxes), label_amount), dtype="int32")

    for pos_index, gt_match in pos_indices:
        offset = gt_boxes[gt_match].calculate_offset(model.default_boxes[pos_index])

        locations[pos_index] = offset
        confidences[pos_index, label_indices[gt_match]] = 1

    mask = np.ones(len(confidences), dtype="bool")
    if len(gt_boxes):
        mask[np.array(pos_indices)[:, 0]] = False
    confidences[mask, 0] = 1

    return locations, confidences


def prepare_dataset(model, path, training=False):
    dataset = []
    annotations = files.load(path)

    for annotation in annotations:
        img_path = os.path.join(path, annotation["image"])
        image, og_img_size = preprocess_image(img_path)

        locations = []
        confidences = []
        for label in annotation["annotations"]:
            scaled_coords = [val / og_img_size[key in ["y", "height"]] for key, val in label["coordinates"].items()]
            locations.append(CellBox(size_coords=scaled_coords))

            confidences.append(labels.index(label["label"]))

        if training:
            locations, confidences = prepare_training(model, locations, confidences)

        dataset.append([image, locations, confidences])

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

    found_labels, found_boxes, confs = model.get_preds(image, conf_threshold=conf_threshold)
    labeled_labels = np.array(labels)[found_labels]
    scaled_boxes = [CellBox(abs_coords=box).scale_box(input_shape[:-1]) for box in found_boxes]

    label_infos = list(zip(labeled_labels, scaled_boxes, confs))

    return label_infos


def evaluate(model, dataset, iou_threshold=0.5, conf_threshold=0.5):
    amount_true_pos = 0
    amount_false_pos = 0
    amount_false_neg = 0

    for img, gt_boxes, gt_labels in dataset:
        iteration_true_pos = 0

        gt_boxes = [gt_box.scale_box(input_shape[:-1]) for gt_box in gt_boxes]

        label_infos = inference(model, img, conf_threshold=conf_threshold)

        for pred_label, box, _ in label_infos:
            if any(box.calculate_iou(gt_box) >= iou_threshold and labels[gt_label] == pred_label for gt_label, gt_box in zip(gt_labels, gt_boxes)):
                iteration_true_pos += 1
            else:
                amount_false_pos += 1
        
        amount_false_neg += len(gt_boxes) - iteration_true_pos
        amount_true_pos += iteration_true_pos

    metric_value = amount_true_pos / (amount_true_pos + amount_false_pos + amount_false_neg)

    return metric_value


if __name__ == "__main__":
    model = SSD_Model(input_shape, label_amount)

    prepared_dataest = random.shuffle(prepare_dataset(model, "dataset", training=True))

    training_dataset = prepared_dataest[int(len(prepared_dataest) * config.TESTING_SPLIT)]
    retrain(model, training_dataset, config.TRAINING_ITERATIONS, config.EPOCHS)

    model.save_model("model")
    model.plot_metrics()

    metadata_changes = {
        "Iterations trained": len(model.metrics["loss"]),
        "Accuracy": "unknown"
    }
    model.convert(labels, metadata_changes=metadata_changes)

    # testing_dataset = prepare_dataset(model, "datasets/SG-mahjong.v1i.tensorflow/test")
    # metric_value = evaluate(model, testing_dataset)
    # print(f"The model got a score of {metric_value}!")

    # img_path = "datasets/SG-mahjong.v1i.tensorflow/test/local_sg_mahjong_with_animal_tiles_travel_size_1551811793_7e51a0d2_progressive_jpg.rf.0d5c4a5f78e69e58dc9788ae8d89e67d.jpg"
    # infos = inference(model, img_path)

    # img = Image.open(img_path).resize(input_shape[:-1])
    # plot_infos(img, infos)
    pass
