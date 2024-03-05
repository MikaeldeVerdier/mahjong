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


def augment_data(image, boxes, labels):
    coords = box_utils.convert_to_coordinates(boxes)
    coords = np.maximum(coords, 0)
    coords = np.minimum(coords, 1)  # I don't really like this

    h, w = image.shape[:-1]
    transform = A.Compose([
        # A.OneOf([
        #     A.Blur(p=0.25),
        #     A.MotionBlur(p=0.25),
        #     A.PixelDropout(p=0.25)
        # ]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.ISONoise(p=0.5),
        A.RandomSizedBBoxSafeCrop(height=h, width=w, p=0.5)
        # A.OneOf([
        #     A.Affine(p=0.25),
        #     A.Perspective(p=0.25)
        # ])
    ], bbox_params=A.BboxParams(format="albumentations", min_visibility=0.4, label_fields=["class_labels"]))

    result = transform(image=image, bboxes=coords, class_labels=labels)

    transformed_img = result["image"]
    transformed_boxes = np.array(result["bboxes"])
    transformed_labels = result["class_labels"]

    if transformed_boxes.shape == (0,):
        transformed_boxes = np.empty_like(boxes)
    
    centroids = box_utils.convert_to_centroids(transformed_boxes)
    data = [transformed_img, centroids, transformed_labels]

    # box_utils.plot_ious(centroids, np.empty_like(transformed_boxes), Image.fromarray(transformed_img, mode="RGB"))

    return data


def prepare_training(image, default_boxes, preprocess_function, gt_boxes, label_indices):
    image_arr = np.array(image)
    data = [[image_arr, gt_boxes, label_indices]]
    for _ in range(config.AUGMENTATION_AMOUNT):
        new_data = augment_data(image_arr, gt_boxes, label_indices)

        if not any([np.array_equal(new_data[0], entry[0]) for entry in data]):
            data.append(new_data)
    
    # [augment_data(image_arr, gt_boxes, label_indices) for _ in range(config.AUGMENTATION_AMOUNT)]

    generated_data = []
    for augmented_image_arr, gt_box, labels in data:
        processed_image = preprocess_function(augmented_image_arr)

        matches = box_utils.match(gt_box, default_boxes, threshold=0.6)

        locations = np.zeros((len(default_boxes), 4), dtype="float32")
        confidences = np.zeros((len(default_boxes), label_amount + 1), dtype="uint8")

        for gt_index, default_index in enumerate(matches):
            offset = box_utils.calculate_offset(gt_box[gt_index], default_boxes[default_index], sq_variances=config.SQ_VARIANCES)

            locations[default_index] = offset
            confidences[default_index, labels[gt_index] + 1] = 1

        confidences[np.sum(confidences, axis=-1) == 0, 0] = 1

        generated_data.append([processed_image, locations, confidences])

    return generated_data


def prepare_dataset(path, training_ratio=0, default_boxes=None, preprocess_function=None, used_ratio=1, start_index=0):
    dataset = [[], []]
    annotations = files.load(path)

    starting_index = int(len(annotations) * start_index)
    amount_used = int(len(annotations) * used_ratio)
    annotations = annotations[starting_index: starting_index + amount_used]

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
            dataset[0] += prepare_training(image, default_boxes, preprocess_function, locations, confidences)
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


def inference(mlmodel, image):  # PIL Image
    locations, confidences = mlmodel.predict({"image": image, "confidenceThreshold": 0.2}).values()
    predicted_labels = np.array(labels)[np.argmax(confidences, axis=-1)]
    # scaled_boxes = box_utils.scale_box(locations, input_shape[:-1])
    label_confs = np.max(confidences, axis=-1)

    label_infos = [predicted_labels, locations, label_confs]

    return label_infos


# def evaluate(mlmodel, dataset, iou_threshold=0.5):
#     amount_true_pos = 0
#     amount_false_pos = 0
#     amount_false_neg = 0

#     for image, gt_boxes, gt_labels in dataset:
#         predicted_labels, predicted_boxes, predicted_confidences = inference(mlmodel, image)

#         # box_utils.plot_ious(gt_boxes, predicted_boxes, image, labels=predicted_labels, confidences=predicted_confidences)

#         ious = box_utils.calculate_iou(gt_boxes, predicted_boxes)

#         amount_false_pos += len(predicted_labels)
#         for iou, gt_label in zip(ious, gt_labels):
#             if np.count_nonzero(predicted_labels[iou > iou_threshold] == labels[gt_label]):
#                 amount_true_pos += 1
#                 amount_false_pos -= 1
#             else:
#                 amount_false_neg += 1

#     metric_value = amount_true_pos / (amount_true_pos + amount_false_pos + amount_false_neg) if any([amount_true_pos, amount_false_pos, amount_false_neg]) else 0

#     return metric_value

def calculate_precision_recall(pred, gt, iou_threshold=0.5):
    # true_positives = 0
    # false_positives = 0
    # false_negatives = 0

    # for pred_box in pred:
    #     ious = box_utils.calculate_iou(np.array([pred_box[1]]), np.array([g[1] for g in gt]))
    #     max_iou = np.max(ious)

    #     if max_iou >= iou_threshold:
    #         true_positives += 1
    #     else:
    #         false_positives += 1

    # false_negatives = max(len(gt) - true_positives, 0)

    ious = box_utils.calculate_iou(np.array(pred), np.array(gt))
    max_ious = np.max(ious, axis=-1)

    true_positives = np.count_nonzero(max_ious >= iou_threshold)
    false_positives = len(max_ious) - true_positives
    false_negatives = np.maximum(len(gt) - true_positives, 0)

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    return precision, recall


def calculate_average_precision(predictions, ground_truth, iou_threshold=0.5):
    precision_values = []
    recall_values = []

    for confidence_threshold in np.arange(1, -0.1, -0.1):  # Typically this would rather be np.arange(0, 1.1, 0.1)
        confident_boxes = [pred[1] for pred in predictions if pred[2] >= confidence_threshold] or np.empty(shape=(0, 4))
        gt_boxes = [gt[1] for gt in ground_truth]

        precision, recall = calculate_precision_recall(confident_boxes, gt_boxes, iou_threshold)
        precision_values.append(precision)
        recall_values.append(recall)

    precision_values = np.array(precision_values)
    recall_values = np.array(recall_values)

    interpolated_precision = np.maximum.accumulate(precision_values[::-1])[::-1]  # [np.max(precision_values[i:]) for i in range(len(precision_values))]
    recall_diff = np.diff(recall_values, prepend=0)

    average_precision = np.sum(interpolated_precision * recall_diff)

    return average_precision


def evaluate(mlmodel, dataset):
    all_preds = []
    all_gts = []

    for image, gt_boxes, gt_labels in dataset:
        pred = inference(mlmodel, image)

        # box_utils.plot_ious(gt_boxes, pred[1], image, labels=pred[0], confidences=pred[2])

        all_preds.append(list(zip(*pred)))
        all_gts.append(list(zip(*[np.array(labels)[gt_labels], gt_boxes])))

    ap_values = []
    for label in labels:
        pred_class = [pred for preds in all_preds for pred in preds if pred[0] == label]
        gt_class = [gt for gts in all_gts for gt in gts if gt[0] == label]

        average_precision = calculate_average_precision(pred_class, gt_class)
        ap_values.append(average_precision)

    mean_average_precision = np.mean(ap_values)

    return mean_average_precision


if __name__ == "__main__":
    model = SSD_Model(input_shape, label_amount)

    div = 4
    for i in range(div):
        training_dataset, testing_dataset = prepare_dataset("dataset", training_ratio=config.TRAINING_SPLIT, default_boxes=model.default_boxes, preprocess_function=model.preprocess_function, used_ratio=1 / div, start_index=i / div)

        retrain(model, training_dataset, int(config.TRAINING_ITERATIONS / div), config.EPOCHS)

    model.save_model("model")
    model.plot_metrics()

    model.convert_to_mlmodel(labels)

    testing_score = evaluate(model.mlmodel, testing_dataset)
    print(f"The model got a testing score of {testing_score}")

    metadata_changes = {
        "additional": {
            "Iterations trained": str(len(model.metrics["loss"])),
            "Testing score": str(np.round(testing_score, 5))
        }
    }
    model.save_mlmodel(metadata_changes=metadata_changes)
