import os
import numpy as np
from PIL import Image
import albumentations as A

import box_utils
import files
import config

def preprocess_image(path, input_shape):
    img = Image.open(path)
    img_size = img.size

    img = img.resize(input_shape[:-1][::-1])
    img = img.convert("RGB")

    return img, img_size


def augment_data(image, boxes, labels):
    coords = box_utils.convert_to_coordinates(boxes)
    coords = np.clip(coords, 0, 1)  # Is weird that only augmented images have clipped boxed...

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
        processed_image = preprocess_function(augmented_image_arr)

        matches, neutral_indices = box_utils.match(gt_box, default_boxes)

        locations = np.zeros((len(default_boxes), 4))
        confidences = np.zeros((len(default_boxes), label_amount + 1))

        for gt_index, default_index in enumerate(matches):
            offset = box_utils.calculate_offset(gt_box[gt_index], default_boxes[default_index], sq_variances=config.SQ_VARIANCES)

            locations[default_index] = offset
            confidences[default_index, labels[gt_index] + 1] = 1

        confidences[np.sum(confidences, axis=-1) == 0, 0] = 1
        confidences[neutral_indices] = np.zeros(label_amount + 1)

        generated_data.append([processed_image, locations, confidences])

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
