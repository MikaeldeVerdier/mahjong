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
    img_size = img.size

    img = img.resize(input_shape[:-1])
    img = img.convert("RGB")

    return img, img_size


def prepare_training(model, image, gt_boxes, label_indices):
    image_arr = np.moveaxis(np.array(image, dtype="float32"), 0, 1)
    image_arr /= 255.0

    pos_indices = model.match_boxes(gt_boxes)

    locations = np.zeros((len(model.default_boxes), 4))
    confidences = np.zeros((len(model.default_boxes), label_amount + 1), dtype="int32")

    for pos_index, gt_match in set(pos_indices):
        offset = gt_boxes[gt_match].calculate_offset(model.default_boxes[pos_index])

        locations[pos_index] = offset
        confidences[pos_index, label_indices[gt_match] + 1] = 1

    mask = np.ones(len(confidences), dtype="bool")
    if len(gt_boxes):
        mask[np.array(pos_indices)[:, 0]] = False
    confidences[mask, 0] = 1

    return image_arr, locations, confidences


def prepare_dataset(model, path, training_ratio=0):
    dataset = []
    annotations = files.load(path)

    amount_training = int(len(annotations) * training_ratio)
    for i, annotation in enumerate(annotations):
        img_path = os.path.join(path, annotation["image"])
        image, image_size = preprocess_image(img_path)

        locations = []
        confidences = []
        for label in annotation["annotations"]:
            box = CellBox(size_coords=label["coordinates"].values())
            scaled_box = box.scale_box((1 / image_size[0], 1 / image_size[1]))
            # scaled_coords = [val / image_size[key in ["y", "height"]] for key, val in label["coordinates"].items()]
            # scaled_box2 = CellBox(size_coords=scaled_coords)
            locations.append(scaled_box)

            confidences.append(labels.index(label["label"]))

        if i < amount_training:
            image, locations, confidences = prepare_training(model, image, locations, confidences)

        dataset.append([image, locations, confidences])

    return dataset, amount_training


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


def inference(model, image):  # PIL Image
    locations, confidences = model.mlmodel.predict({"input_1": image}).values()
    labeled_labels = np.array(labels)[np.argmax(confidences, axis=-1) - 1]
    scaled_boxes = [CellBox(abs_coords=box).scale_box(input_shape[:-1]) for box in locations]

    label_infos = list(zip(labeled_labels, scaled_boxes, confidences))

    return label_infos


def evaluate(model, dataset, iou_threshold=0.5):
    amount_true_pos = 0
    amount_false_pos = 0
    amount_false_neg = 0

    for image, gt_boxes, gt_labels in dataset:
        iteration_true_pos = 0

        gt_boxes = [gt_box.scale_box(input_shape[:-1]) for gt_box in gt_boxes]

        label_infos = inference(model, image)

        for pred_label, box, _ in label_infos:
            if any(box.calculate_iou(gt_box) >= iou_threshold and labels[gt_label] == pred_label for gt_label, gt_box in zip(gt_labels, gt_boxes)):
                iteration_true_pos += 1  # Can be right twice for same gt...
            else:
                amount_false_pos += 1
        
        amount_false_neg += max(0, len(gt_boxes) - iteration_true_pos)  # ... resulting in this max being needed. Could use abs instead to punish this behavior.
        amount_true_pos += iteration_true_pos

    metric_value = amount_true_pos / (amount_true_pos + amount_false_pos + amount_false_neg)

    return metric_value


if __name__ == "__main__":
    model = SSD_Model(input_shape, label_amount)

    dataset, split_index = prepare_dataset(model, "dataset", training_ratio=config.TRAINING_SPLIT)
    training_dataset = dataset[:split_index]
    testing_dataset = dataset[split_index:]

    retrain(model, training_dataset, config.TRAINING_ITERATIONS, config.EPOCHS)

    model.save_model("model")
    model.plot_metrics()

    model.convert_to_mlmodel(labels)

    testing_score = evaluate(model, testing_dataset)
    print(f"The model got a testing score of {testing_score}")

    metadata_changes = {
        "additional": {
            "Iterations trained": str(len(model.metrics["loss"])),
            "Testing score": str(np.round(testing_score, 5))
        }
    }
    model.save_mlmodel(metadata_changes=metadata_changes)
