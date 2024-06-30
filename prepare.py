import os
import numpy as np
from PIL import Image
# import albumentations as A

import box_utils
import augmentation
import files
import config
import cv2

def preprocess_image(path, input_shape):
    # img = Image.open(path)

    # img = img.resize(input_shape[:-1][::-1])
    # img = img.convert("RGB")

    img = cv2.imread(path)  # Seems to be around 50% faster than PIL
    img = cv2.resize(img, input_shape[:-1][::-1], interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1]

    return img


def augment_data(image, boxes, labels):
    if not config.USE_AUGMENTATION:
        return image, boxes, labels

    augmentor = augmentation.SSDAugmentation(size=image.shape[0])  # Doesn't support non-square images right now

    bgr_image = image[:, :, ::-1]  # Equivalent to cv2.cvtColor(image, cv2.RGB2BGR)
    coords = box_utils.convert_to_coordinates(boxes)

    transformed_img, transformed_boxes, transformed_labels = augmentor(bgr_image, coords, np.array(labels))

    if transformed_boxes.shape == (0,):
        transformed_boxes = np.empty(shape=(0, 4))
    
    rgb_image = transformed_img[:, :, ::-1]
    centroids = box_utils.convert_to_centroids(transformed_boxes)
    data = [rgb_image, centroids, transformed_labels]

    # box_utils.plot_ious(centroids, np.empty(shape=(0, 4)), Image.fromarray(np.uint8(rgb_image), mode="RGB"))

    return data


def prepare_training(image_path, gt_boxes, label_indices, input_shape, label_amount, default_boxes):
    image_arr = preprocess_image(image_path, input_shape)
    # image_arr = np.array(image)

    augmented_image_arr, gt_box, labels = augment_data(image_arr, gt_boxes, label_indices)
    # box_utils.plot_ious(gt_box, np.empty(shape=(0, 4)), Image.fromarray(np.uint8(augmented_image_arr), mode="RGB"))

    # processed_image = preprocess_function(augmented_image_arr)  # Writes over augmented_image_arr (processed_image == augmented_image_arr[:, :, ::-1])

    matches, neutral_indices = box_utils.match(gt_box, default_boxes)

    locations = np.zeros((len(default_boxes), 4))
    confidences = np.zeros((len(default_boxes), label_amount + 1))

    for gt_index, default_index in enumerate(matches):
        offset = box_utils.calculate_offset(gt_box[gt_index], default_boxes[default_index], variances=config.VARIANCES)

        locations[default_index] = offset
        confidences[default_index, labels[gt_index] + 1] = 1

    confidences[np.sum(confidences, axis=-1) == 0, 0] = 1
    confidences[neutral_indices] = np.zeros(label_amount + 1)

    gt = np.concatenate([confidences, locations], axis=-1)
    generated_data = [augmented_image_arr, gt]

    return generated_data


def prepare_testing(image_path, gt_boxes, label_indices, input_shape):
    image = preprocess_image(image_path, input_shape)
    image = Image.fromarray(image, mode="RGB")

    return image, gt_boxes, label_indices


def prepare_dataset(path, labels, training_ratio=0):
    dataset = [[], []]
    annotations = files.load(path)

    amount_training = int(len(annotations) * training_ratio)
    for i, annotation in enumerate(annotations):
        img_path = os.path.join(path, annotation["image"])
        image_size = Image.open(img_path).size

        locations = np.empty(shape=(0, 4))
        confidences = []
        for label in annotation["annotations"]:
            box = np.array(list(label["coordinates"].values()))
            scaled_box = box_utils.scale_box(box, (1 / image_size[0], 1 / image_size[1]))
            # scaled_coords = [val / image_size[key in ["y", "height"]] for key, val in label["coordinates"].items()]
            # scaled_box2 = CellBox(size_coords=scaled_coords)
            locations = np.concatenate([locations, scaled_box[None]])

            confidences.append(labels.index(label["label"]))

        dataset[int(i >= amount_training)].append([img_path, locations, confidences])

    return dataset
