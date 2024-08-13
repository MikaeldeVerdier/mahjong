import cv2
import numpy as np
import random
from PIL import Image

import box_utils

def convert_data_type(desired):  # Could just use classes here like a normal person
    def transform(image, boxes, labels):
        if desired == "float32":
            image = image.astype(np.float32)
        elif desired == "uint8":
            image = np.round(image, decimals=0).astype(np.uint8)

        return image, boxes, labels

    return transform


def convert_color(source, desired):
    def transform(image, boxes, labels):
        if source == "RGB" and desired == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif source == "HSV" and desired == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

        return image, boxes, labels

    return transform


def random_brightness(delta=32, p=0.5):
    def transform(image, boxes, labels):
        if np.random.rand() > p:  # Make this positive statement format instead?
            return image, boxes, labels

        alpha = np.random.uniform(-delta, delta)
        image = np.clip(image + alpha, 0, 255)

        return image, boxes, labels

    return transform


def random_contrast(lower_bound=0.5, upper_bound=1.5, p=0.5):
    def transform(image, boxes, labels):
        if np.random.rand() > p:
            return image, boxes, labels

        alpha = np.random.uniform(lower_bound, upper_bound)
        image = np.clip(127.5 + alpha * (image - 127.5), 0, 255)

        return image, boxes, labels

    return transform


def random_saturation(lower_bound=0.5, upper_bound=1.5, p=0.5):
    def transform(image, boxes, labels):
        if np.random.rand() > p:
            return image, boxes, labels

        alpha = np.random.uniform(lower_bound, upper_bound)
        image[:, :, 1] = np.clip(image[:, :, 1] * alpha, 0, 255)

        return image, boxes, labels

    return transform


def random_hue(delta=18, p=0.5):
    def transform(image, boxes, labels):
        if np.random.rand() > p:
            return image, boxes, labels

        alpha = np.random.uniform(-delta, delta)
        image[:, :, 0] = (image[:, :, 0] + alpha) % 180

        return image, boxes, labels

    return transform


def random_channel_swap(p=0.5):
    perms = [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

    def transform(image, boxes, labels):
        if np.random.rand() > p:
            return image, boxes, labels

        chosen_perm = random.choice(perms)
        image = image[:, :, chosen_perm]

        return image, boxes, labels

    return transform


def photometric_distort():
    sequence_1 = [
        convert_data_type("float32"),
        random_brightness(),
        random_contrast(),
        convert_data_type("uint8"),
        convert_color("RGB", "HSV"),
        convert_data_type("float32"),
        random_saturation(),
        random_hue(),
        convert_data_type("uint8"),
        convert_color("HSV", "RGB"),
        random_channel_swap(p=0)
    ]
    sequence_2 = [
        convert_data_type("float32"),
        random_brightness(),
        convert_data_type("uint8"),
        convert_color("RGB", "HSV"),
        convert_data_type("float32"),
        random_saturation(),
        random_hue(),
        convert_data_type("uint8"),
        convert_color("HSV", "RGB"),
        convert_data_type("float32"),
        random_contrast(),
        convert_data_type("uint8"),
        random_channel_swap(p=0)
    ]

    def transform(image, boxes, labels):
        chosen_sequence = random.choice([sequence_1, sequence_2])

        for t in chosen_sequence:
            image, boxes, labels = t(image, boxes, labels)

        return image, boxes, labels

    return transform


def convert_boxes(source, destination):  # Is this one really necessary? Could just do it before and after data augmentation
    def transform(image, boxes, labels):
        if source == "centroids" and destination == "coordinates":
            boxes = box_utils.convert_to_coordinates(boxes)
        elif source == "coordinates" and destination == "centroids":
            boxes = box_utils.convert_to_centroids(boxes)

        return image, boxes, labels

    return transform


def convert_coords(source, destination):
    def transform(image, boxes, labels):
        height, width, _ = image.shape

        if source == "relative" and destination == "absolute":
            boxes = box_utils.scale_box(boxes, (width, height))
        elif source == "absolute" and destination == "relative":
            boxes = box_utils.scale_box(boxes, (1 / width, 1 / height))

        return image, boxes, labels

    return transform


def random_expand(lower_bound=1, upper_bound=4, background_color=[123, 117, 104], p=0.5):
    def transform(image, boxes, labels):
        if np.random.rand() > p:
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = np.random.uniform(lower_bound, upper_bound)

        top = int(np.random.uniform(0, height * (ratio - 1)))
        left = int(np.random.uniform(0, width * (ratio - 1)))
        new_height = int(height * ratio)
        new_width = int(width * ratio)

        expanded_image = np.zeros((new_height, new_width, depth), dtype=image.dtype)
        expanded_image[:] = background_color
        expanded_image[top:(top + height), left:(left + width)] = image
        image = expanded_image

        boxes += (left, top, left, top)

        return image, boxes, labels

    return transform


def random_sample_crop(min_scale=0.3, max_scale=1, min_ar=0.5, max_ar=2, amount_max_trials=50, p_per_max_trial=0.857):
    sample_options = [
        (0.1, None),
        (0.3, None),
        (0.7, None),
        (0.9, None),
        (None, None)
    ]

    def transform(image, boxes, labels):
        height, width, _ = image.shape

        while np.random.rand() < p_per_max_trial:
            min_iou, max_iou = random.choice(sample_options)
            if min_iou is None:
                min_iou = 0
            if max_iou is None:
                max_iou = 1

            for _ in range(amount_max_trials):
                current_image = image.copy()

                new_h = int(np.random.uniform(height * min_scale, height * max_scale))  # np.random.randint ?
                new_w = int(np.random.uniform(width * min_scale, width * max_scale))

                new_ar = new_h / new_w
                if new_ar < min_ar or new_ar > max_ar:
                    continue  # ar outside or allowed range

                top = int(np.random.uniform(0, height - new_h))
                left = int(np.random.uniform(0, width - new_w))

                rect = np.array([left, top, left + new_w, top + new_h])

                centroids_boxes = box_utils.convert_to_centroids(boxes)
                centroids_rect = box_utils.convert_to_centroids(rect[None])
                ious = box_utils.calculate_iou(centroids_boxes, centroids_rect)[:, 0]

                if np.min(ious) <= min_iou or np.max(ious) > max_iou:  # different from amdegroot's?
                    continue  # iou of some box is outside of allowed range

                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2]]
                centers = centroids_boxes[:, :2]

                mask_1 = (rect[0] < centers[:, 0]) & (rect[1] < centers[:, 1])
                mask_2 = (rect[2] > centers[:, 0]) & (rect[3] > centers[:, 1])
                mask = mask_1 & mask_2

                if not np.any(mask):
                    continue  # no valid boxes

                image = current_image  # What is this ocd for everything to take and return with the same variable names

                boxes = boxes[mask]
                boxes[:, :2] = np.maximum(boxes[:, :2], rect[:2])
                boxes[:, :2] -= rect[:2]
                boxes[:, 2:] = np.minimum(boxes[:, 2:], rect[2:])
                boxes[:, 2:] -= rect[:2]

                labels = labels[mask]

                return image, boxes, labels  # double break

        return image, boxes, labels

    return transform


def random_flip(dim="horizontal", p=0.5):
    def transform(image, boxes, labels):
        if np.random.rand() > p:
            return image, boxes, labels

        img_height, img_width, _ = image.shape

        if dim == "horizontal":
            image = image[:, ::-1]
            boxes[:, [0, 2]] = img_width - boxes[:, [2, 0]]
        elif dim == "vertical":
            image = image[::-1]
            boxes[:, [1, 3]] = img_height - boxes[:, [3, 1]]

        return image, boxes, labels

    return transform


def random_rotate(p=0.5):
    def transform(image, boxes, labels):
        if np.random.rand() > p:
            return image, boxes, labels

        angle = np.random.uniform(0, 360)

        image_pil = Image.fromarray(image)
        rotated_image_pil = image_pil.rotate(angle, expand=True)
        rotated_image = np.array(rotated_image_pil)

        height, width = image.shape[:2]
        new_height, new_width = rotated_image.shape[:2]

        angle_rad = np.deg2rad(angle)

        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        center_x = width / 2
        center_y = height / 2

        new_center_x = new_width / 2
        new_center_y = new_height / 2

        rotated_bboxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box

            corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

            corners_translated = corners - np.array([center_x, center_y])

            rotated_corners = np.dot(corners_translated, np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]))
            rotated_corners = rotated_corners + np.array([new_center_x, new_center_y])

            new_xmin = np.min(rotated_corners[:, 0])
            new_ymin = np.min(rotated_corners[:, 1])
            new_xmax = np.max(rotated_corners[:, 0])
            new_ymax = np.max(rotated_corners[:, 1])

            rotated_bboxes.append((new_xmin, new_ymin, new_xmax, new_ymax))

        image = rotated_image
        boxes = rotated_bboxes

        return image, boxes, labels

    return transform


def random_perspective_warp(min_factor=0, max_factor=0.2, p=0.5):
    def transform(image, boxes, labels):
        if np.random.rand() > p:
            return image, boxes, labels

        height, width = image.shape[:2]

        factor = np.random.uniform(min_factor, max_factor)

        src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst_points = src_points + np.random.uniform(-width * factor, width * factor, src_points.shape).astype(np.float32)

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.warpPerspective(image, M, (width, height))

        warped_bboxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)

            warped_points = cv2.perspectiveTransform(np.array([points]), M)[0]

            new_xmin = np.min(warped_points[:, 0])
            new_ymin = np.min(warped_points[:, 1])
            new_xmax = np.max(warped_points[:, 0])
            new_ymax = np.max(warped_points[:, 1])

            warped_bboxes.append((new_xmin, new_ymin, new_xmax, new_ymax))

        boxes = warped_bboxes

        return image, boxes, labels

    return transform


def random_homography_transform(p=0.5):
    def transform(image, boxes, labels):
        if np.random.rand() > p:
            return image, boxes, labels

        height, width = image.shape[:2]

        src_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        dst_points = src_points + np.random.uniform(-width * 0.3, width * 0.3, src_points.shape).astype(np.float32)

        H, _ = cv2.findHomography(src_points, dst_points)
        image = cv2.warpPerspective(image, H, (width, height))

        transformed_bboxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            points = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)

            transformed_points = cv2.perspectiveTransform(np.array([points]), H)[0]

            new_xmin = np.min(transformed_points[:, 0])
            new_ymin = np.min(transformed_points[:, 1])
            new_xmax = np.max(transformed_points[:, 0])
            new_ymax = np.max(transformed_points[:, 1])

            transformed_bboxes.append((new_xmin, new_ymin, new_xmax, new_ymax))

        boxes = transformed_bboxes

        return image, boxes, labels

    return transform


def random_motion_blur(min_kernel_size=1, max_kernel_size=10, p=0.5):
    def transform(image, boxes, labels):
        if np.random.rand() > p:
            return image, boxes, labels

        kernel_size = int(np.random.uniform(min_kernel_size, max_kernel_size))

        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel /= kernel_size

        image = cv2.filter2D(image, -1, kernel)

        return image, boxes, labels

    return transform


def random_resize(width, height):
    interpolation_options = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]

    def transform(image, boxes, labels):
        chosen_interpolation = random.choice(interpolation_options)
        image = cv2.resize(image, (width, height), interpolation=chosen_interpolation)

        return image, boxes, labels

    return transform


def single_cache(f):  # Quite a weird thing, feels like it indicates that the structure could be improved to avoid this.
    value = None

    def inner(*args):
        nonlocal value
        value = value or f(*args)

        return value

    return inner


@single_cache
def ssd_augmentation(image_width, image_height):
    transforms = [
        photometric_distort(),
        convert_boxes("centroids", "coordinates"),
        convert_coords("relative", "absolute"),
        random_expand(),
        random_sample_crop(),
        random_flip("horizontal"),
        random_rotate(),
        random_perspective_warp(),
        random_homography_transform(),
        random_motion_blur(),
        convert_coords("absolute", "relative"),
        convert_boxes("coordinates", "centroids"),
        random_resize(image_width, image_height)
    ]

    def transform(image, boxes, labels):
        for t in transforms:
            image, boxes, labels = t(image, boxes, labels)

        return image, boxes, labels

    return transform
