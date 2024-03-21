import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config

# All functions asume centroids for bounding boxes (except convert_to_centroids)

def default_boxes(k, m, aspect_ratios, f, scales=[0.1, 0.5], im_aspect_ratio=1):
	def s(k, m, s_min=scales[0], s_max=scales[1]):
		return s_min + (s_max - s_min) / (m - 1) * (k - 1)

	scale_i = s(k, m)
	extra_box_scale = np.sqrt(scale_i * s(k + 1, m))

	def create_boxes(scale, aspect_ratio, f):
		w = scale * np.sqrt(aspect_ratio)
		h = scale / np.sqrt(aspect_ratio)

		return [[(i + 0.5) / f[0], (j + 0.5) / f[1], w, h] for i in range(f[0]) for j in range(f[1])]

	anchor_boxes = np.array([create_boxes(scale_i, ar, f) for ar in aspect_ratios] + [create_boxes(extra_box_scale, 1 * im_aspect_ratio, f)])

	return anchor_boxes


def convert_to_centroids(boxes):
	min_xs = boxes[:, 0]
	min_ys = boxes[:, 1]
	max_xs = boxes[:, 2]
	max_ys = boxes[:, 3]

	w = max_xs - min_xs
	h = max_ys - min_ys
	cx = min_xs + w * 0.5
	cy = min_ys + h * 0.5

	boxes = np.transpose([cx, cy, w, h])

	return boxes


def convert_to_coordinates(boxes):
	cxs = boxes[:, 0]
	cys = boxes[:, 1]
	ws = boxes[:, 2]
	hs = boxes[:, 3]

	min_x = cxs - ws * 0.5
	min_y = cys - hs * 0.5
	max_x = min_x + ws
	max_y = min_y + hs

	boxes = np.transpose([min_x, min_y, max_x, max_y])  # np.moveaxis([min_x, min_y, max_x, max_y], 0, -1) instead?

	return boxes


def scale_box(boxes, scalars):
	new_boxes = boxes * np.tile(scalars, 2)

	return new_boxes


def calculate_iou(boxes1, boxes2):
    coords1 = convert_to_coordinates(boxes1)
    coords2 = convert_to_coordinates(boxes2)

    x1 = np.maximum(coords1[:, 0][:, None], coords2[:, 0])
    y1 = np.maximum(coords1[:, 1][:, None], coords2[:, 1])
    x2 = np.minimum(coords1[:, 2][:, None], coords2[:, 2])
    y2 = np.minimum(coords1[:, 3][:, None], coords2[:, 3])

    intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    box_area = np.abs(boxes1[:, 2] * boxes1[:, 3])
    other_box_area = np.abs(boxes2[:, 2] * boxes2[:, 3])

    union_area = box_area[:, None] + other_box_area - intersection_area

    iou = intersection_area / union_area

    return iou


def match(boxes1, boxes2, threshold=0.5):
	ious = calculate_iou(boxes1, boxes2)

	matches = np.argmax(ious, axis=-1)[:, None].tolist()
	for gt_index, default_index in zip(*np.where(ious > threshold)):  # Could probably be simplified to avoid this zip 
		matches[gt_index].append(default_index)

	return matches


def plot_ious(gts, boxes, img, labels=None, confidences=None, name="boxes.png", scale_coords=True):  # Should scale_coords even be an option? I never use scaled coords
	if labels is None:
		labels = [""] * len(boxes)
	if confidences is None:
		confidences = [""] * len(boxes)

	_, ax = plt.subplots()

	plt.imshow(img)
	w, h = img.size

	ious = calculate_iou(gts, boxes)

	if scale_coords:
		gts = scale_box(gts, (w, h))
		boxes = scale_box(boxes, (w, h))

	gt_coords = convert_to_coordinates(gts)
	coords = convert_to_coordinates(boxes)

	iou_font = {"color": "green", "size": 5}
	label_font = {"color": "black", "size": 5}
	for gt, gt_coord in zip(gts, gt_coords):
		ax.add_artist(Rectangle((gt_coord[0], gt_coord[1]), gt[2], gt[3], linewidth=1, edgecolor="g", facecolor="none"))

	for box, coord, iou, label, confidence in zip(boxes, coords, np.transpose(ious), labels, confidences):
		ax.add_artist(Rectangle((coord[0], coord[1]), box[2], box[3], linewidth=1, edgecolor="r", facecolor="none"))  # Could do opacity based on confidence

		left, top = box[0], coord[1]
		plt.text(left, top - 5, f"IOU: {np.max(iou):.3f}", horizontalalignment="center", fontdict=iou_font)  # Doesn't say which gt the iou is for but should be fine

		bottom = coord[3]
		desc = "" if label is None else f"{label} ({confidence:.3f})" if confidence else f"{label}"
		plt.text(left, bottom + 10, desc, horizontalalignment="center", fontdict=label_font)

	plt.savefig(f"{config.SAVE_FOLDER_PATH}/{name}", dpi=300)
	plt.close()


def calculate_offset(gt, boxes, sq_variances=None):
	cx = (gt[0] - boxes[:, 0]) / boxes[:, 2]
	cy = (gt[1] - boxes[:, 1]) / boxes[:, 3]
	w = np.log(gt[2] / boxes[:, 2])
	h = np.log(gt[3] / boxes[:, 3])

	offset = np.moveaxis([cy, cx, h, w], 0, -1)

	if sq_variances is not None:
		offset /= np.sqrt(sq_variances)

	return offset


"""
def calculate_offset(gt, boxes):
	cx = (gt[0] - boxes[:, 0]) / boxes[:, 2]
	cy = (gt[1] - boxes[:, 1]) / boxes[:, 3]
	w = np.log(gt[2] / boxes[:, 2])
	h = np.log(gt[3] / boxes[:, 3])

	offset = np.moveaxis([cx, cy, w, h], 0, -1)

	return offset
"""


"""  # Unfinished, unecessary
def apply_offset(boxes, offsets):
	y = boxes[:, 0]
	o = offsets[:, :, 0]
	cx = boxes[:, 0] + offsets[:, :, 0] * boxes[:, 2]
	cy = boxes[:, 1] + offsets[:, :, 1] * boxes[:, 3]
	w = boxes[:, 2] * np.exp(offsets[:, :, 2])
	h = boxes[:, 3] * np.exp(offsets[:, :, 3])
	
	box = np.moveaxis([cx, cy, w, h], 0, -1)

	return box
"""
