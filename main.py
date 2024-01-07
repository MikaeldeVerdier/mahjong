import os
import random
import numpy as np
import albumentations as A
from PIL import Image

import box_utils
import files
import config
from model import SSD_Model

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

input_shape = (512, 288, 3)

def preprocess_image(path):
    img = Image.open(path)
    img_size = img.size

    img = img.resize(input_shape[:-1][::-1])
    img = img.convert("RGB")

    return img, img_size


def data_augment(image, boxes, labels):
    boxes = box_utils.convert_to_coordinates(boxes)
    boxes = np.maximum(boxes, 0)
    boxes = np.minimum(boxes, 1)

    transform = A.Compose([
        A.OneOf([
            A.Blur(p=0.5),
            A.MotionBlur(p=0.5),
            A.PixelDropout(p=0.5)
        ]),
        A.OneOf([
            A.Affine(p=0.5),
            A.Perspective(p=0.5)
        ]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.BBoxSafeRandomCrop(p=0.5)
    ], bbox_params=A.BboxParams(format="albumentations", min_visibility=0.4, label_fields=["class_labels"]))

    result = transform(image=image, bboxes=boxes, class_labels=labels)

    img = result["image"]
    bboxes = np.array(result["bboxes"])
    labels = result["class_labels"]

    if bboxes.shape == (0,):
        bboxes = np.empty((0, 4))

    return img, bboxes, labels


def prepare_training(model, image, gt_boxes, label_indices):
    image_arr = np.array(image)
    data = [[image_arr, gt_boxes, label_indices]] + [data_augment(image_arr, gt_boxes, label_indices) for _ in range(config.AUGMENTATION_AMOUNT)]

    generated_data = []
    for image_arr, gt_box, labels in data:
        processed_image = model.preprocess_function(image_arr)

        matches = box_utils.match(gt_box, model.default_boxes)

        locations = np.zeros((len(model.default_boxes), 4))
        confidences = np.zeros((len(model.default_boxes), label_amount + 1), dtype="int32")

        for gt_index, default_index in enumerate(matches):
            offset = box_utils.calculate_offset(gt_box[gt_index], model.default_boxes[default_index])

            locations[default_index] = offset
            confidences[default_index, labels[gt_index] + 1] = 1

        confidences[np.sum(confidences, axis=-1) == 0, 0] = 1

        generated_data.append([processed_image, locations, confidences])

    return generated_data


def prepare_dataset(model, path, training_ratio=0):
    dataset = [[], []]
    annotations = files.load(path)

    amount_training = int(len(annotations) * training_ratio)
    for i, annotation in enumerate(annotations):
        img_path = os.path.join(path, annotation["image"])
        image, image_size = preprocess_image(img_path)

        locations = np.empty(shape=(0, 4))
        confidences = []
        for label in annotation["annotations"]:
            box = np.array(list(label["coordinates"].values()))
            scaled_box = box_utils.scale_box(box, (1 / image_size[0], 1 / image_size[1]))
            # scaled_coords = [val / image_size[key in ["y", "height"]] for key, val in label["coordinates"].items()]
            # scaled_box2 = CellBox(size_coords=scaled_coords)
            locations = np.concatenate([locations, scaled_box[None]])

            confidences.append(labels.index(label["label"]))

        if i < amount_training:
            dataset[0] += prepare_training(model, image, locations, confidences)
        else:
            dataset[1].append([image, locations, confidences])

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


def inference(model, image):  # PIL Image
    locations, confidences = model.mlmodel.predict({"image": image}).values()
    predicted_labels = np.array(labels)[np.argmax(confidences, axis=-1)]
    # scaled_boxes = box_utils.scale_box(locations, input_shape[:-1])
    label_confs = confidences[np.arange(len(confidences)), np.argmax(confidences, axis=-1)]

    label_infos = [predicted_labels, locations, label_confs]

    return label_infos


def evaluate(model, dataset, iou_threshold=0.5):
    amount_true_pos = 0
    amount_false_pos = 0
    amount_false_neg = 0

    for image, gt_boxes, gt_labels in dataset:
        predicted_labels, predicted_boxes, _ = inference(model, image)

        ious = box_utils.calculate_iou(gt_boxes, predicted_boxes)

        amount_false_pos += len(predicted_labels)
        for iou, gt_label in zip(ious, gt_labels):
            if np.count_nonzero(predicted_labels[iou > iou_threshold] == labels[gt_label]):
                amount_true_pos += 1
                amount_false_pos -= 1
            else:
                amount_false_neg += 1

    metric_value = amount_true_pos / (amount_true_pos + amount_false_pos + amount_false_neg) if any([amount_true_pos, amount_false_pos, amount_false_neg]) else 0

    return metric_value


if __name__ == "__main__":
    model = SSD_Model(input_shape, label_amount)

    training_dataset, testing_dataset = prepare_dataset(model, "dataset", training_ratio=config.TRAINING_SPLIT)

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
