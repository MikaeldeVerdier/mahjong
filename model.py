import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Input, Reshape
from tensorflow.keras.losses import CategoricalCrossentropy, Huber
from tensorflow.keras.metrics import Accuracy, MeanAbsoluteError
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD

from default_box import default_boxes

class SSD_Model:
	def __init__(self, inp_shape, class_amount, max_output=25, lr=1e-4, momentum=0.9, load=False):
		self.inp_shape = inp_shape
		self.class_amount = class_amount
		self.max_output = max_output

		if load is not False:
			self.load_model(load)
			return

		base_network = VGG16(include_top=False, weights="imagenet", input_shape=inp_shape)
		# frozen_layer_amount = 5
		# for layer in base_network.layers[:frozen_layer_amount]:
		# 	layer.trainable = False
		base_network.trainable = False

		# inp = Input(shape=self.inp_shape)

		outputs = []

		# x = base_network(inp)
		x = base_network.get_layer("block4_conv3").output
		outputs.append(x)

		x = base_network.get_layer("block5_conv3").output
		outputs.append(x)

		# Auxiliary layers
		x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation="relu")(x)
		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(x)
		outputs.append(x)

		x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation="relu")(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(x)
		outputs.append(x)

		x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation="relu")(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
		outputs.append(x)

		x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation="relu")(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu")(x)
		outputs.append(x)
		#

		self.default_boxes = []
		aspect_ratios = [0.75, 1, 1.33]

		head_outputs = [[], []]
		for k, output in enumerate(outputs, 1):
			defaults = default_boxes(k, len(outputs), aspect_ratios, output.shape[1])
			for default in defaults:
				self.default_boxes += default.cells

			location_pred = Conv2D(filters=len(defaults) * 4, kernel_size=(3, 3), strides=(1, 1), padding="same")(output)
			location_pred = Reshape((-1, 4))(location_pred)
			head_outputs[0].append(location_pred)

			class_pred = Conv2D(filters=len(defaults) * class_amount, kernel_size=(3, 3), strides=(1, 1), padding="same")(output)
			class_pred = Reshape((-1, class_amount))(class_pred)
			class_pred = Activation("softmax")(class_pred)
			head_outputs[1].append(class_pred)

		location_predictions = Concatenate(axis=1, name="locations")(head_outputs[0])
		class_predictions = Concatenate(axis=1, name="confidences")(head_outputs[1])

		self.model = Model(inputs=[base_network.input], outputs=[location_predictions, class_predictions])
		self.model.compile(loss={"locations": Huber(), "confidences": CategoricalCrossentropy(from_logits=True)}, optimizer=SGD(learning_rate=lr, momentum=momentum), metrics={"locations": MeanAbsoluteError(), "confidences": Accuracy()})

		self.plot_model()
		self.model.summary()

		self.metrics = {"loss": [], "locations_loss": [], "confidences_loss": [], "locations_mean_absolute_error": [], "confidences_accuracy": []}

	def postprocessing(self, boxes, scores, iou_threshold=0.5, score_threshold=0.18):
		boxes = tf.convert_to_tensor(boxes, dtype="float32")
		classes = tf.argmax(scores, axis=1)
		defaults = self.default_boxes
		scores = tf.reduce_max(scores, axis=1)

		selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=self.max_output, iou_threshold=iou_threshold, score_threshold=score_threshold)

		selected_boxes = tf.gather(boxes, selected_indices).numpy()
		selected_classes = tf.gather(classes, selected_indices).numpy()
		selected_defaults = np.take(defaults, selected_indices)
		selected_scores = tf.gather(scores, selected_indices).numpy()

		return selected_boxes, selected_classes, selected_defaults, selected_scores

	def get_preds(self, data):
		inp = tf.expand_dims(data, axis=0)
		bbox_preds, class_preds = self.model.predict_on_batch(inp)

		selected_boxes, selected_classes, selected_defaults, selected_scores = self.postprocessing(bbox_preds[0], class_preds[0])

		bounding_boxes = [box.apply_offset(selected_box).abs_coords for box, selected_box in zip(selected_defaults, selected_boxes)]

		return selected_classes, bounding_boxes, selected_scores

	def match_boxes(self, gt_boxes, threshold=0.5):
		matches = []
		for i, gt_box in enumerate(gt_boxes):
			gt_ious = [box.calculate_iou(gt_box) for box in self.default_boxes]

			matches.append((np.argmax(gt_ious), i))

		for i, box in enumerate(self.default_boxes):
			def_ious = [box.calculate_iou(gt_box) for gt_box in gt_boxes]

			if max(def_ious) > threshold:
				matches.append((i, np.argmax(def_ious)))

		return matches

	def train(self, x, y, epochs):
		fit = self.model.fit(x, y, epochs=epochs)
		for metric in self.metrics:
			self.metrics[metric] += fit.history[metric]

	def save_model(self, name):
		self.model.save(f"save_folder/{name}")

		with open("save_folder/save.json", "w") as f:
			f.write(json.dumps(self.metrics))

		with open("save_folder/default_boxes.pkl", "wb") as f:
			pickle.dump(self.default_boxes, f)

	def load_model(self, name):
		self.model = load_model(f"save_folder/{name}")

		with open("save_folder/save.json", "r") as f:
			self.metrics = json.loads(f.read())
		
		with open("save_folder/default_boxes.pkl", "rb") as f:
			self.default_boxes = pickle.load(f)

	def plot_model(self):
		try:
			plot_model(self.model, to_file="save_folder/model_architecture.png", show_shapes=True, show_layer_names=True)
		except ImportError:
			print("You need to install pydot and graphviz to plot model architecture.")

	def plot_metrics(self):
		_, axs = plt.subplots(len(self.metrics), figsize=(15, 4 * len(self.metrics)))

		for ax, metric in zip(axs, self.metrics):
			ax.plot(self.metrics[metric], label=metric)
			ax.set_xscale("linear")
			ax.legend()

		plt.savefig("save_folder/metrics.png", dpi=200)
