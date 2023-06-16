import numpy as np

def default_boxes(k, m, aspect_ratios, f):
	def s(k, m, s_min=0.2, s_max=0.90):
		return s_min + (s_max - s_min) / (m - 1) * (k - 1)

	scale_i = s(k, m)
	extra_box_scale = (scale_i * s(k + 1, m)) ** 0.5

	anchor_boxes = [DefaultBox(scale_i, aspect_ratio, f) for aspect_ratio in aspect_ratios] + [DefaultBox(extra_box_scale, 1, f)]

	return anchor_boxes


class DefaultBox:
	def __init__(self, scale, aspect_ratio, f):
		self.w = scale * np.sqrt(aspect_ratio)
		self.h = scale / np.sqrt(aspect_ratio)

		self.cells = [CellBox(size_coords=[(i + 0.5) / f, (j + 0.5) / f, self.w, self.h]) for i in range(f) for j in range(f)]


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

		if x2 < x1 or y2 < y1:
			return 0

		box_area = (self.abs_coords[2] - self.abs_coords[0]) * (self.abs_coords[3] - self.abs_coords[1])
		gt_box_area = (other_box.abs_coords[2] - other_box.abs_coords[0]) * (other_box.abs_coords[3] - other_box.abs_coords[1])

		intersection_area = (x2 - x1) * (y2 - y1)
		union_area = box_area + gt_box_area - intersection_area

		iou = intersection_area / union_area
		
		return iou

	def calculate_offset(self, other_box):
		cx = self.size_coords[0] - other_box.size_coords[0]
		cy = self.size_coords[1] - other_box.size_coords[1]
		w = np.log(self.size_coords[2] / other_box.size_coords[2])
		h = np.log(self.size_coords[3] / other_box.size_coords[3])

		return (cx, cy, w, h)

	def apply_offset(self, offset):
		cx = self.size_coords[0] + offset[0] * self.size_coords[2]
		cy = self.size_coords[1] + offset[1] * self.size_coords[3]
		w = self.size_coords[2] * np.exp(offset[2])
		h = self.size_coords[3] * np.exp(offset[3])
		other_box = CellBox(size_coords=np.array([cx, cy, w, h]))

		return other_box
