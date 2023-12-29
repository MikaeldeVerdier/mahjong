import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import config

# TODO: Consider reworking this entire file

def default_boxes(k, m, aspect_ratios, f):
	def s(k, m, s_min=0.07, s_max=0.6):
		return s_min + (s_max - s_min) / (m - 1) * (k - 1)

	scale_i = s(k, m)
	extra_box_scale = (scale_i * s(k + 1, m)) ** 0.5

	anchor_boxes = [DefaultBox(scale_i, aspect_ratio, f) for aspect_ratio in aspect_ratios] + [DefaultBox(extra_box_scale, 1, f)]

	return anchor_boxes


class DefaultBox:
	def __init__(self, scale, aspect_ratio, f):
		self.w = scale * np.sqrt(aspect_ratio)
		self.h = scale / np.sqrt(aspect_ratio)

		self.cells = [CellBox(size_coords=[(i + 0.5) / f[0], (j + 0.5) / f[1], self.w, self.h]) for i in range(f[0]) for j in range(f[1])]


class CellBox:
	def __init__(self, size_coords=None, abs_coords=None):
		if size_coords is not None:
			cx, cy, w, h = size_coords

			min_x = cx - w * 0.5
			min_y = cy - h * 0.5
			max_x = cx + w * 0.5
			max_y = cy + h * 0.5
		else:
			min_x, min_y, max_x, max_y = abs_coords

			w = max_x - min_x
			h = max_y - min_y
			cx = min_x + w * 0.5
			cy = min_y + h * 0.5

		self.size_coords = (cx, cy, w, h)
		self.abs_coords = (min_x, min_y, max_x, max_y)

	def calculate_iou(self, other_box):
		x1 = max(self.abs_coords[0], other_box.abs_coords[0])
		y1 = max(self.abs_coords[1], other_box.abs_coords[1])
		x2 = min(self.abs_coords[2], other_box.abs_coords[2])
		y2 = min(self.abs_coords[3], other_box.abs_coords[3])

		intersection_box = CellBox(abs_coords=(x1, y1, x2, y2))
		intersection_area = max(intersection_box.size_coords[2], 0) * max(intersection_box.size_coords[3], 0)

		box_area = abs((self.size_coords[2]) * (self.size_coords[3]))
		other_box_area = abs((other_box.size_coords[2]) * (other_box.size_coords[3]))

		union_area = box_area + other_box_area - intersection_area

		iou = intersection_area / union_area
		
		return iou

	def plot_iou(self, other_box, img, name="boxes.png", scale_coords=False):
		_, ax = plt.subplots()

		plt.imshow(img)

		w, h = img.size
		for box, color in [(self, "g"), (other_box, "r")]:
			if scale_coords:
				box = box.scale_box((w, h))
			ax.add_artist(Rectangle((box.abs_coords[0], box.abs_coords[1]), box.size_coords[2], box.size_coords[3], linewidth=1, edgecolor=color, facecolor="none"))

		font = {"color": "green"}
		left, top = self.size_coords[0], self.abs_coords[1]
		if scale_coords:
			left *= w
			top *= h
		plt.text(left, top, f"IOU: {self.calculate_iou(other_box):.5f}", horizontalalignment="center", fontdict=font)

		plt.savefig(f"{config.SAVE_FOLDER_PATH}/{name}")
		plt.close()

	def calculate_offset(self, other_box):
		cx = (self.size_coords[0] - other_box.size_coords[0]) / other_box.size_coords[2]
		cy = (self.size_coords[1] - other_box.size_coords[1]) / other_box.size_coords[3]
		w = np.log(self.size_coords[2] / other_box.size_coords[2])
		h = np.log(self.size_coords[3] / other_box.size_coords[3])

		return (cx, cy, w, h)

	def apply_offset(self, offset):
		cx = self.size_coords[0] + offset[0] * self.size_coords[2]
		cy = self.size_coords[1] + offset[1] * self.size_coords[3]
		w = self.size_coords[2] * np.exp(offset[2])
		h = self.size_coords[3] * np.exp(offset[3])
		other_box = CellBox(size_coords=[cx, cy, w, h])

		return other_box

	def scale_box(self, scalars):
		new_abs_coords = self.abs_coords * np.tile(scalars, (2))
		new_box = CellBox(abs_coords=new_abs_coords)

		return new_box
