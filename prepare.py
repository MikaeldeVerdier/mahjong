import os
import numpy as np
from PIL import Image
# import albumentations as A

import box_utils
import augment
import files
import config
import cv2

def preprocess_image(path):
    # img = Image.open(path)

    # img = img.resize(input_shape[:-1][::-1])
    # img = img.convert("RGB")

    img = cv2.imread(path)  # Seems to be around 50% faster than PIL
    # img = cv2.resize(img, input_shape[:-1][::-1], interpolation=cv2.INTER_LINEAR)  # Could reintroduce this or something similar to make augmentation faster
    img = img[:, :, ::-1]

    return img


def augment_data(image, boxes, labels, input_shape):
    if not config.USE_AUGMENTATION:
        image = cv2.resize(image, input_shape[:-1][::-1])

        return image, boxes, labels

    augmentor = augment.ssd_augmentation(input_shape[1], input_shape[0])  # This doesn't need to be reinitialized every time (currently uses a single-cache against this)
    transformed_img, transformed_boxes, transformed_labels = augmentor(image, boxes, np.array(labels))

    if transformed_boxes.shape == (0,):
        transformed_boxes = np.empty(shape=(0, 4))

    # box_utils.plot_ious(transformed_boxes, np.empty(shape=(0, 4)), Image.fromarray(transformed_img, mode="RGB"))

    return transformed_img, transformed_boxes, transformed_labels


def prepare_training(image_path, gt_boxes, label_indices, input_shape, label_amount, default_boxes):
    image_arr = preprocess_image(image_path)
    # image_arr = np.array(image)

    augmented_image_arr, gt_box, labels = augment_data(image_arr, gt_boxes, label_indices, input_shape)
    # box_utils.plot_ious(gt_box, np.empty(shape=(0, 4)), Image.fromarray(augmented_image_arr, mode="RGB"))

    # processed_image = preprocess_function(augmented_image_arr)  # Writes over augmented_image_arr (processed_image == augmented_image_arr[:, :, ::-1])

    matches, neutral_indices = box_utils.match(gt_box, default_boxes)

    locations = np.zeros((len(default_boxes), 4))
    confidences = np.zeros((len(default_boxes), label_amount + 1))

    for gt_index, default_index in enumerate(matches):
        offset = box_utils.calculate_offset(gt_box[gt_index], default_boxes[default_index], variances=config.VARIANCES)

        # box_utils.plot_ious(gt_box[gt_index][None], default_boxes[default_index], Image.fromarray(augmented_image_arr, mode="RGB"))

        locations[default_index] = offset
        confidences[default_index, labels[gt_index] + 1] = 1

    confidences[np.sum(confidences, axis=-1) == 0, 0] = 1
    confidences[neutral_indices] = np.zeros(label_amount + 1)

    gt = np.concatenate([confidences, locations], axis=-1)
    generated_data = [augmented_image_arr, gt]

    return generated_data


def prepare_testing(image_path, gt_boxes, label_indices, input_shape):
    image = Image.open(image_path)
    image = image.resize(input_shape[:-1][::-1])

    return image, gt_boxes, label_indices


def prepare_dataset(path, labels, training_ratio=0, exclude_difficult=False):
    dataset = [[], []]
    annotations = files.load(path)

    amount_training = int(len(annotations) * training_ratio)
    for i, annotation in enumerate(annotations):
        img_path = os.path.join(path, annotation["image"])
        image_size = Image.open(img_path).size

        locations = np.empty(shape=(0, 4))
        confidences = []
        for label in annotation["annotations"]:
            if exclude_difficult and label["difficulty"]:
                continue

            box = np.array(list(label["coordinates"].values()))
            scaled_box = box_utils.scale_box(box, (1 / image_size[0], 1 / image_size[1]))
            # scaled_coords = [val / image_size[key in ["y", "height"]] for key, val in label["coordinates"].items()]
            # scaled_box2 = CellBox(size_coords=scaled_coords)
            locations = np.concatenate([locations, scaled_box[None]])

            confidences.append(labels.index(label["label"]))

        dataset[int(i >= amount_training)].append([img_path, locations, confidences])

    return dataset
