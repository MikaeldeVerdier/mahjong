import os
import numpy as np
from PIL import Image
# import albumentations as A

import box_utils
import augmentation
import files
import config

def preprocess_image(path, input_shape):
    img = Image.open(path)
    img_size = img.size

    img = img.resize(input_shape[:-1][::-1])
    img = img.convert("RGB")

    return img, img_size


def augment_data(image, boxes, labels):
    bgr_image = np.float32(image[:, :, ::-1])  # Equivalent to cv2.cvtColor(image, cv2.RGB2BGR)
    coords = box_utils.scale_box(box_utils.convert_to_coordinates(boxes), image.shape[:-1])

    result = [bgr_image, coords, labels]
    augmentations = [
        augmentation.random_contrast,
        augmentation.random_contrast,
        augmentation.random_hue,
        augmentation.random_lighting_noise,
        augmentation.random_saturation,
        augmentation.random_vertical_flip,
        augmentation.random_horizontal_flip,
        augmentation.random_expand,
        augmentation.random_crop
    ]

    for augment in augmentations:
        result = augment(*result)

    if result[0].shape != image.shape:
        resize_func = augmentation.resize_to_fixed_size(image.shape[0], image.shape[1])
        result = resize_func(*result)

    transformed_img, transformed_boxes, transformed_labels = result

    if transformed_boxes.shape == (0,):
        transformed_boxes = np.empty(shape=(0, 4))
    
    rgb_image = transformed_img[:, :, ::-1]
    centroids = box_utils.convert_to_centroids(box_utils.scale_box(transformed_boxes, (1 / image.shape[0], 1 / image.shape[1])))
    data = [rgb_image, centroids, transformed_labels]

    # box_utils.plot_ious(centroids, np.empty(shape=(0, 4)), Image.fromarray(np.uint8(transformed_img[:, :, ::-1]), mode="RGB"), scale_coords=False)

    return data


def prepare_training(image, label_amount, default_boxes, preprocess_function, gt_boxes, label_indices):
    image_arr = np.array(image)
    data = [[image_arr, gt_boxes, label_indices]]
    for _ in range(config.AUGMENTATION_AMOUNT):
        new_data = augment_data(image_arr, gt_boxes, label_indices)

        if not any([np.array_equal(new_data[0], entry[0]) for entry in data]):
            data.append(new_data)
    
    # [augment_data(image_arr, gt_boxes, label_indices) for _ in range(config.AUGMENTATION_AMOUNT)]

    generated_data = []
    for augmented_image_arr, gt_box, labels in data:
        # box_utils.plot_ious(gt_box, np.empty(shape=(0, 4)), Image.fromarray(np.uint8(augmented_image_arr), mode="RGB"))

        processed_image = preprocess_function(augmented_image_arr)

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
        generated_data.append([processed_image, gt])

    return generated_data


def prepare_dataset(path, labels, input_shape, training_ratio=0, default_boxes=None, preprocess_function=None, used_ratio=1, start_index=0):
    dataset = [[], []]
    annotations = files.load(path)

    starting_index = int(len(annotations) * start_index)
    amount_used = int(len(annotations) * used_ratio)
    annotations = annotations[starting_index: starting_index + amount_used]

    amount_training = int(len(annotations) * training_ratio)
    for i, annotation in enumerate(annotations):
        img_path = os.path.join(path, annotation["image"])
        image, image_size = preprocess_image(img_path, input_shape)

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
            dataset[0] += prepare_training(image, len(labels), default_boxes, preprocess_function, locations, confidences)
        else:
            dataset[1].append([image, locations, confidences])

    return dataset
