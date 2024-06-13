# All functions in this file are taken from https://towardsdatascience.com/implementing-single-shot-detector-ssd-in-keras-part-iv-data-augmentation-59c9f230a910 (https://github.com/Socret360/object-detection-in-keras)

import random
import numpy as np
import cv2

from box_utils import calculate_iou

def random_brightness(image, bboxes=None, classes=None, min_delta=-32, max_delta=32, p=0.5):
    """ Changes the brightness of an image by adding/subtracting a delta value to/from each pixel.
    The image format is assumed to be BGR to match Opencv's standard.
    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - min_delta: minimum delta value.
        - max_delta: maximum delta value.
        - p: The probability with which the brightness is changed
    Returns:
        - image: The modified image
        - bboxes: The unmodified bounding boxes
        - classes: The unmodified bounding boxes
    Raises:
        - min_delta is less than -255.0
        - max_delta is larger than 255.0
        - p is smaller than zero
        - p is larger than 1
    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert min_delta >= -255.0, "min_delta must be larger than -255.0"
    assert max_delta <= 255.0, "max_delta must be less than 255.0"
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, bboxes, classes

    temp_image = image.copy()
    d = random.uniform(min_delta, max_delta)
    temp_image += d
    temp_image = np.clip(temp_image, 0, 255)
    return temp_image, bboxes, classes


def random_contrast(image, bboxes=None, classes=None, min_delta=0.5, max_delta=1.5, p=0.5):
    """ Changes the contrast of an image by increasing/decreasing each pixel by a factor of delta.
    The image format is assumed to be BGR to match Opencv's standard.
    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - min_delta: minimum delta value.
        - max_delta: maximum delta value.
        - p: The probability with which the contrast is changed
    Returns:
        - image: The modified image
        - bboxes: The unmodified bounding boxes
        - classes: The unmodified bounding boxes
    Raises:
        - min_delta is less than 0
        - max_delta is less than min_delta
        - p is smaller than zero
        - p is larger than 1
    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert min_delta >= 0.0, "min_delta must be larger than zero"
    assert max_delta >= min_delta, "max_delta must be larger than min_delta"
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, bboxes, classes

    temp_image = image.copy()
    d = random.uniform(min_delta, max_delta)
    temp_image *= d
    temp_image = np.clip(temp_image, 0, 255)
    return temp_image, bboxes, classes


def random_hue(image, bboxes=None, classes=None, min_delta=-18, max_delta=18, p=0.5):
    """ Changes the Hue of an image by adding/subtracting a delta value
    to/from each value in the Hue channel of the image. The image format
    is assumed to be BGR to match Opencv's standard.
    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - min_delta: minimum delta value.
        - max_delta: maximum delta value.
        - p: The probability with which the Hue is adjusted
    Returns:
        - image: The modified image
        - bboxes: The unmodified bounding boxes
        - classes: The unmodified bounding boxes
    Raises:
        - min_delta is less than -360.0
        - max_delta is larger than 360.0
        - p is smaller than zero
        - p is larger than 1
    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert min_delta >= -360.0, "min_delta must be larger than -360.0"
    assert max_delta <= 360.0, "max_delta must be less than 360.0"
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, bboxes, classes

    temp_image = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2HSV)
    temp_image = np.array(temp_image, dtype=np.float32)
    d = random.uniform(min_delta, max_delta)
    temp_image[:, :, 0] += d
    temp_image = np.clip(temp_image, 0, 360)
    temp_image = cv2.cvtColor(np.uint8(temp_image), cv2.COLOR_HSV2BGR)
    temp_image = np.array(temp_image, dtype=np.float32)
    return temp_image, bboxes, classes


def random_lighting_noise(image, bboxes=None, classes=None, p=0.5):
    """ Changes the lighting of the image by randomly swapping the channels.
    The image format is assumed to be BGR to match Opencv's standard.
    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - p: The probability with which the contrast is changed
    Returns:
        - image: The modified image
        - bboxes: The unmodified bounding boxes
        - classes: The unmodified bounding boxes
    Raises:
        - p is smaller than zero
        - p is larger than 1
    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, bboxes, classes

    temp_image = image.copy()
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0)
    ]
    selected_perm = random.randint(0, len(perms) - 1)
    perm = perms[selected_perm]
    temp_image = temp_image[:, :, perm]
    return temp_image, bboxes, classes


def random_saturation(image, bboxes=None, classes=None, min_delta=0.5, max_delta=1.5, p=0.5):
    """ Changes the saturation of an image by increasing/decreasing each
    value in the saturation channel by a factor of delta. The image format
    is assumed to be BGR to match Opencv's standard.
    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - min_delta: minimum delta value.
        - max_delta: maximum delta value.
    Returns:
        - image: The modified image
        - bboxes: The unmodified bounding boxes
        - classes: The unmodified bounding boxes
    Raises:
        - min_delta is less than 0
        - max_delta is less than min_delta
        - p is smaller than zero
        - p is larger than 1
    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert min_delta >= 0.0, "min_delta must be larger than zero"
    assert max_delta >= min_delta, "max_delta must be larger than min_delta"
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, bboxes, classes

    temp_image = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2HSV)
    temp_image = np.array(temp_image, dtype=np.float32)
    d = random.uniform(min_delta, max_delta)
    temp_image[:, :, 1] *= d
    temp_image = cv2.cvtColor(np.uint8(temp_image), cv2.COLOR_HSV2BGR)
    temp_image = np.array(temp_image, dtype=np.float32)
    return temp_image, bboxes, classes


def random_vertical_flip(image, bboxes, classes, p=0.5):
    """ Randomly flipped the image vertically. The image format is assumed to be BGR to match Opencv's standard.
    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes in corners format (xmin, ymin, xmax, ymax).
        - classes: the list of classes associating with each bounding boxes.
        - p: The probability with which the image is flipped vertically
    Returns:
        - image: The modified image
        - bboxes: The modified bounding boxes
        - classes: The unmodified bounding boxes
    Raises:
        - p is smaller than zero
        - p is larger than 1
    Webpage References:
        - https://www.kdnuggets.com/2018/09/data-augmentation-bounding-boxes-image-transforms.html/2
    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """

    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, bboxes, classes

    temp_bboxes = bboxes.copy()
    image_center = np.array(image.shape[:2])[::-1]/2
    image_center = np.hstack((image_center, image_center))
    temp_bboxes[:, [1, 3]] += 2*(image_center[[1, 3]] - temp_bboxes[:, [1, 3]])
    boxes_height = abs(temp_bboxes[:, 1] - temp_bboxes[:, 3])
    temp_bboxes[:, 1] -= boxes_height
    temp_bboxes[:, 3] += boxes_height
    return np.array(cv2.flip(np.uint8(image), 0), dtype=np.float32), temp_bboxes, classes


def random_horizontal_flip(image, bboxes, classes, p=0.5):
    """ Randomly flipped the image horizontally. The image format is assumed to be BGR to match Opencv's standard.
    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes in corners format (xmin, ymin, xmax, ymax).
        - classes: the list of classes associating with each bounding boxes.
        - p: The probability with which the image is flipped horizontally
    Returns:
        - image: The modified image
        - bboxes: The modified bounding boxes
        - classes: The unmodified bounding boxes
    Raises:
        - p is smaller than zero
        - p is larger than 1
    Webpage References:
        - https://www.kdnuggets.com/2018/09/data-augmentation-bounding-boxes-image-transforms.html/2
    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"

    if (random.random() > p):
        return image, bboxes, classes

    temp_bboxes = bboxes.copy()
    image_center = np.array(image.shape[:2])[::-1]/2
    image_center = np.hstack((image_center, image_center))
    temp_bboxes[:, [0, 2]] += 2*(image_center[[0, 2]] - temp_bboxes[:, [0, 2]])
    boxes_width = abs(temp_bboxes[:, 0] - temp_bboxes[:, 2])
    temp_bboxes[:, 0] -= boxes_width
    temp_bboxes[:, 2] += boxes_width
    return np.array(cv2.flip(np.uint8(image), 1), dtype=np.float32), temp_bboxes, classes


def random_expand(image, bboxes, classes, min_ratio=1, max_ratio=4, mean=[104, 117, 123] , p=0.5):  # old means = [0.406, 0.456, 0.485] (mean / 255)
    """ Randomly expands an image and bounding boxes by a ratio between min_ratio and max_ratio. The image format is assumed to be BGR to match Opencv's standard.
    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - min_ratio: The minimum value to expand the image. Defaults to 1.
        - max_ratio: The maximum value to expand the image. Defaults to 4.
        - p: The probability with which the image is expanded
    Returns:
        - image: The modified image
        - bboxes: The modified bounding boxes
        - classes: The unmodified bounding boxes
    Raises:
        - p is smaller than zero
        - p is larger than 1
    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"
    assert min_ratio > 0, "min_ratio must be larger than zero"
    assert max_ratio > min_ratio, "max_ratio must be larger than min_ratio"

    if (random.random() > p):
        return image, bboxes, classes

    height, width, depth = image.shape
    ratio = random.uniform(min_ratio, max_ratio)
    left = random.uniform(0, width * ratio - width)
    top = random.uniform(0, height * ratio - height)
    temp_image = np.zeros(
        (int(height * ratio), int(width * ratio), depth),
        dtype=image.dtype
    )
    temp_image[:, :, :] = mean
    temp_image[int(top):int(top+height), int(left):int(left+width)] = image
    temp_bboxes = bboxes.copy()
    temp_bboxes[:, :2] += (int(left), int(top))
    temp_bboxes[:, 2:] += (int(left), int(top))
    return temp_image, temp_bboxes, classes


def random_crop(image, bboxes, classes, min_size=0.1, max_size=1, min_ar=1, max_ar=2, overlap_modes=[None, [0.1, None], [0.3, None], [0.7, None], [0.9, None], [None, None]], max_attempts=100, p=0.5):
    """ Randomly crops a patch from the image.
    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - min_size: the maximum size a crop can be
        - max_size: the maximum size a crop can be
        - min_ar: the minimum aspect ratio a crop can be
        - max_ar: the maximum aspect ratio a crop can be
        - overlap_modes: the list of overlapping modes the function can randomly choose from.
        - max_attempts: the max number of attempts to generate a patch.
    Returns:
        - image: the modified image
        - bboxes: the modified bounding boxes
        - classes: the modified classes
    Webpage References:
        - https://www.telesens.co/2018/06/28/data-augmentation-in-ssd/
    Code References:
        - https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py
    """
    assert p >= 0, "p must be larger than or equal to zero"
    assert p <= 1, "p must be less than or equal to 1"
    assert min_size > 0, "min_size must be larger than zero."
    assert max_size <= 1, "max_size must be less than or equals to one."
    assert max_size > min_size, "max_size must be larger than min_size."
    assert max_ar > min_ar, "max_ar must be larger than min_ar."
    assert max_attempts > 0, "max_attempts must be larger than zero."

    if (random.random() > p):
        return image, bboxes, classes

    height, width, channels = image.shape
    overlap_mode = random.choice(overlap_modes)

    if overlap_mode == None:
        return image, bboxes, classes

    min_iou, max_iou = overlap_mode

    if min_iou == None:
        min_iou = float(-np.inf)

    if max_iou == None:
        max_iou = float(np.inf)

    temp_image = image.copy()

    for i in range(max_attempts):
        crop_w = random.uniform(min_size * width, max_size * width)
        crop_h = random.uniform(min_size * height, max_size * height)
        crop_ar = crop_h / crop_w

        if crop_ar < min_ar or crop_ar > max_ar:  # crop ar does not match criteria, next attempt
            continue

        crop_left = random.uniform(0, width-crop_w)
        crop_top = random.uniform(0, height-crop_h)

        crop_rect = np.array([crop_left, crop_top, crop_left + crop_w, crop_top + crop_h], dtype=np.float32)
        crop_rect = np.expand_dims(crop_rect, axis=0)
        crop_rect = np.tile(crop_rect, (bboxes.shape[0], 1))

        ious = calculate_iou(crop_rect, bboxes)

        if ious.min() < min_iou and ious.max() > max_iou:
            continue

        bbox_centers = np.zeros((bboxes.shape[0], 2), dtype=np.float32)
        bbox_centers[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
        bbox_centers[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2

        cx_in_crop = (bbox_centers[:, 0] > crop_left) * (bbox_centers[:, 0] < crop_left + crop_w)
        cy_in_crop = (bbox_centers[:, 1] > crop_top) * (bbox_centers[:, 1] < crop_top + crop_h)
        boxes_in_crop = cx_in_crop * cy_in_crop

        if not boxes_in_crop.any():
            continue

        temp_image = temp_image[int(crop_top): int(crop_top+crop_h), int(crop_left): int(crop_left+crop_w), :]
        temp_classes = np.array(classes)
        temp_classes = temp_classes[boxes_in_crop]
        temp_bboxes = bboxes[boxes_in_crop]
        crop_rect = np.array([crop_left, crop_top, crop_left + crop_w, crop_top + crop_h], dtype=np.float32)
        crop_rect = np.expand_dims(crop_rect, axis=0)
        crop_rect = np.tile(crop_rect, (temp_bboxes.shape[0], 1))
        temp_bboxes[:, :2] = np.maximum(temp_bboxes[:, :2], crop_rect[:, :2])  # if bboxes top left is out of crop then use crop's xmin, ymin
        temp_bboxes[:, :2] -= crop_rect[:, :2]  # translate xmin, ymin to fit crop
        temp_bboxes[:, 2:] = np.minimum(temp_bboxes[:, 2:], crop_rect[:, 2:])
        temp_bboxes[:, 2:] -= crop_rect[:, :2]  # translate xmax, ymax to fit crop
        return temp_image, temp_bboxes, temp_classes.tolist()

    return image, bboxes, classes


def resize_to_fixed_size(width, height):
    """ Resize the input image and bounding boxes to fixed size.

    Args:
        - image: numpy array representing the input image.
        - bboxes: numpy array representing the bounding boxes.
        - classes: the list of classes associating with each bounding boxes.
        - width: minimum delta value.
        - height: maximum delta value.

    Returns:
        - image: The modified image
        - bboxes: The unmodified bounding boxes
        - classes: The unmodified bounding boxes

    Raises:
        - width is less than 0
        - height is less than 0
    """
    assert width >= 0, "width must be larger than 0"
    assert height >= 0, "height must be larger than 0"

    def _augment(image, bboxes, classes=None):
        temp_image = np.uint8(image)
        o_height, o_width, _ = temp_image.shape
        height_scale, width_scale = height / o_height, width / o_width
        temp_image = cv2.resize(temp_image, (width, height))
        temp_image = np.array(temp_image, dtype=np.float32)
        temp_bboxes = bboxes.copy()
        temp_bboxes[:, [0, 2]] *= width_scale
        temp_bboxes[:, [1, 3]] *= height_scale
        temp_bboxes[:, [0, 2]] = np.clip(temp_bboxes[:, [0, 2]], 0, width)
        temp_bboxes[:, [1, 3]] = np.clip(temp_bboxes[:, [1, 3]], 0, height)

        return temp_image, temp_bboxes, classes

    return _augment
