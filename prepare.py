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

    img = img.resize(input_shape[:-1][::-1])
    img = img.convert("RGB")

    return img


def augment_data(image, boxes, labels, augmentations):
    bgr_image = np.float32(image[:, :, ::-1])  # Equivalent to cv2.cvtColor(image, cv2.RGB2BGR)
    coords = box_utils.scale_box(box_utils.convert_to_coordinates(boxes), image.shape[:-1])

    result = [bgr_image, coords, labels]
    for augment in augmentations:
        result = augment(*result, p=1)

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


def custom_cache(max_size=1):
    def decorator(func):
        cache = {}

        def cache_func(*args):
            key = ""
            for arg in args:
                if isinstance(arg, np.ndarray):
                    key += str(hash(tuple(arg.flatten())))
                else:
                    if arg.__hash__:
                        key += str(hash(arg))
                    else:
                        continue

            if key in cache:
                return cache[key]
            else:
                val = func(*args)
                cache[key] = val

                if len(cache) > max_size:
                    cache.popitem()

                return val

        return cache_func

    return decorator


@custom_cache()
def prepare_training(image_path, gt_boxes, label_indices, augmentations, input_shape, label_amount, default_boxes, preprocess_function):
    image = preprocess_image(image_path, input_shape)
    image_arr = np.array(image)

    augmented_image_arr, gt_box, labels = augment_data(image_arr, gt_boxes, label_indices, augmentations)

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
    generated_data = [processed_image, gt]

    return generated_data


def prepare_dataset(path, labels, training_ratio=0):
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
    probabilities = [0.5] * len(augmentations)

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

        if i < amount_training:
            for _ in range(config.AUGMENTATION_AMOUNT):
                mask = np.random.rand(len(augmentations)) < probabilities
                chosen_augmentations = [augmentation_func for augmentation_func, selected in zip(augmentations, mask) if selected]
                new_data = [img_path, locations, confidences, chosen_augmentations]

                if not any([np.array_equal(new_data[0], entry[0]) for entry in dataset[0]]):
                    dataset[0].append(new_data)
        else:
            dataset[1].append([img_path, locations, confidences])

    return dataset
