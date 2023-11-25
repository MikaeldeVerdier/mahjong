import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import coremltools as ct

from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Reshape
from tensorflow.keras.losses import CategoricalCrossentropy, Huber
from tensorflow.keras.metrics import MeanSquaredError, Accuracy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD

import config
from default_box import default_boxes, CellBox

class SSD_Model:
	def __init__(self, input_shape, class_amount, lr=1e-5, momentum=0.9, hard_neg_ratio=3, load=False):
		self.input_shape = input_shape
		self.class_amount = class_amount
		self.hard_neg_ratio = hard_neg_ratio

		if load is not False:
			self.load_model(load)
			return

		base_network = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
		# frozen_layer_amount = 5
		# for layer in base_network.layers[:frozen_layer_amount]:
		# 	layer.trainable = False
		base_network.trainable = False

		# inp = Input(shape=self.inp_shape)

		outputs = []

		x = base_network.get_layer("block4_conv3").output
		x = tf.math.l2_normalize(x)
		outputs.append(x)

		x = base_network.get_layer("block5_conv3").output

		# Auxiliary layers
		x = Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation="relu")(x)
		x = Conv2D(filters=1024, kernel_size=(1, 1), activation="relu")(x)
		outputs.append(x)

		x = Conv2D(filters=256, kernel_size=(1, 1), activation="relu")(x)
		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(x)
		outputs.append(x)

		x = Conv2D(filters=128, kernel_size=(1, 1), activation="relu")(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(x)
		outputs.append(x)

		x = Conv2D(filters=128, kernel_size=(1, 1), activation="relu")(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu")(x)
		outputs.append(x)

		x = Conv2D(filters=128, kernel_size=(1, 1), activation="relu")(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu")(x)
		outputs.append(x)
		#

		self.default_boxes = []
		aspect_ratios = [0.75, 1, 1.33]

		head_outputs = [[], []]
		for k, output in enumerate(outputs, 1):
			defaults = default_boxes(k, len(outputs), aspect_ratios, output.shape[1:3])
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

		self.model = Model(inputs=[base_network.input], outputs=[class_predictions, location_predictions])
		self.model.compile(loss={"locations": self.huber_with_mask, "confidences": self.categorical_crossentropy_with_mask}, optimizer=SGD(learning_rate=lr, momentum=momentum))  # , metrics={"locations": MeanSquaredError(), "confidences": Accuracy()})

		self.plot_model()
		self.model.summary()

		self.metrics = {"loss": [], "locations_loss": [], "confidences_loss": []}  # , "locations_mean_squared_error": [], "confidences_accuracy": []}

	def huber_with_mask(self, y_true, y_pred):
		pos_losses = Huber(reduction="none")(y_true, y_pred)
		neg_losses = pos_losses

		pos_mask = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)
		pos_multiplier = tf.cast(pos_mask, tf.float32)
		pos_losses *= pos_multiplier

		neg_mask = tf.logical_not(pos_mask)
		neg_multiplier = tf.cast(neg_mask, tf.float32)
		neg_losses *= neg_multiplier

		sorted_neg_losses = tf.sort(neg_losses, direction="DESCENDING")
		ks = tf.expand_dims(tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1), axis=-1)

		indices = tf.range(tf.shape(sorted_neg_losses)[-1])
		indices_expanded = tf.expand_dims(indices, axis=0)
		mask_multiplier = tf.where(indices_expanded <= ks, tf.ones_like(indices_expanded, dtype=tf.float32), tf.zeros_like(indices_expanded, dtype=tf.float32))
		top_neg_losses = mask_multiplier * sorted_neg_losses

		loss = tf.reduce_sum(tf.concat([pos_losses, top_neg_losses], axis=-1), axis=-1)

		return loss

	def categorical_crossentropy_with_mask(self, y_true, y_pred):
		pos_losses = CategoricalCrossentropy(reduction="none")(y_true, y_pred)
		neg_losses = pos_losses

		pos_mask = tf.not_equal(y_true[:, :, 0], 1)
		pos_multiplier = tf.cast(pos_mask, tf.float32)
		pos_losses *= pos_multiplier

		neg_mask = tf.logical_not(pos_mask)
		neg_multiplier = tf.cast(neg_mask, tf.float32)
		neg_losses *= neg_multiplier

		sorted_neg_losses = tf.sort(neg_losses, direction="DESCENDING")
		ks = tf.expand_dims(tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1), axis=-1)

		indices = tf.range(tf.shape(sorted_neg_losses)[-1])
		indices_expanded = tf.expand_dims(indices, axis=0)
		mask_tensor = tf.where(indices_expanded <= ks, tf.ones_like(indices_expanded, dtype=tf.float32), tf.zeros_like(indices_expanded, dtype=tf.float32))
		top_neg_losses = mask_tensor * sorted_neg_losses

		loss = tf.reduce_sum(tf.concat([pos_losses, top_neg_losses], axis=-1), axis=-1)

		return loss

	def postprocessing(self, boxes, scores, max_output_size=50, iou_threshold=0.5, score_threshold=0.1):
		classes = tf.argmax(scores, axis=1)
		non_backgrounds = classes != 0

		boxes = tf.convert_to_tensor(np.array(boxes)[non_backgrounds], dtype="float32")
		classes = classes[non_backgrounds]
		scores = tf.reduce_max(scores[non_backgrounds], axis=1)

		selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=max_output_size, iou_threshold=iou_threshold, score_threshold=score_threshold)

		selected_boxes = tf.gather(boxes, selected_indices).numpy()
		selected_classes = tf.gather(classes, selected_indices).numpy()
		selected_scores = tf.gather(scores, selected_indices).numpy()

		return selected_boxes, selected_classes, selected_scores

	def get_preds(self, data, conf_threshold=0.1):
		inp = np.expand_dims(data, axis=0)
		offset_preds, class_preds = self.model.predict_on_batch(inp)

		bounding_boxes = [default_box.apply_offset(offset).abs_coords for default_box, offset in zip(self.default_boxes, offset_preds[0])]

		selected_boxes, selected_classes, selected_scores = self.postprocessing(bounding_boxes, class_preds[0], score_threshold=conf_threshold)

		return selected_classes, selected_boxes, selected_scores

	def match_boxes(self, gt_boxes, threshold=0.5):
		matches = []
		for i, gt_box in enumerate(gt_boxes):
			gt_ious = [box.calculate_iou(gt_box) for box in self.default_boxes]

			matches.append((np.argmax(gt_ious), i))

		if len(gt_boxes):
			for i, box in enumerate(self.default_boxes):
				def_ious = [box.calculate_iou(gt_box) for gt_box in gt_boxes]

				if max(def_ious) > threshold:
					matches.append((i, np.argmax(def_ious)))

		return matches
	
	"""
	def hard_negative_mining(self, x, y_true_conf, mask):
		x = np.expand_dims(x, axis=0)
		y_pred_conf = self.model.predict_on_batch(x)[1][0]

		conf_loss = CategoricalCrossentropy(reduction="none")(y_true_conf, y_pred_conf)

		pos_indices = np.where(np.logical_not(mask))[0]

		neg_losses = conf_loss[mask]
		sorted_neg_indices = np.argsort(neg_losses)[::-1]
		k = len(pos_indices) * self.hard_neg_ratio
		top_neg_indices = sorted_neg_indices[:k]

		chosen_indices = np.concatenate([pos_indices, top_neg_indices])

		return chosen_indices
	"""

	def train(self, x, y, epochs):
		# y_true = y["locations"]
		# y_pred = self.model.predict(x)[0]

		# loss = self.huber_with_mask(y_true, y_pred)

		fit = self.model.fit(x, y, epochs=epochs)

		for metric in self.metrics:
			self.metrics[metric] += fit.history[metric]

	def save_model(self, name):
		self.model.save(f"{config.SAVE_FOLDER_PATH}/{name}")

		with open(f"{config.SAVE_FOLDER_PATH}/save.json", "w") as f:
			f.write(json.dumps(self.metrics))

		with open(f"{config.SAVE_FOLDER_PATH}/default_boxes.json", "w") as f:
			boxes = [box.abs_coords for box in self.default_boxes]
			f.write(json.dumps(boxes))

	def load_model(self, name):
		self.model = load_model(f"{config.SAVE_FOLDER_PATH}/{name}", custom_objects={"huber_with_mask": self.huber_with_mask, "categorical_crossentropy_with_mask": self.categorical_crossentropy_with_mask})

		with open(f"{config.SAVE_FOLDER_PATH}/save.json", "r") as f:
			self.metrics = json.loads(f.read())

		with open(f"{config.SAVE_FOLDER_PATH}/default_boxes.json", "r") as f:
			reading = json.loads(f.read())
			self.default_boxes = [CellBox(abs_coords=coords) for coords in reading]

	def plot_model(self):
		try:
			plot_model(self.model, to_file=f"{config.SAVE_FOLDER_PATH}/model_architecture.png", show_shapes=True, show_layer_names=True)
		except ImportError:
			print("You need to install pydot and graphviz to plot model architecture.")

	def plot_metrics(self):
		_, axs = plt.subplots(len(self.metrics), figsize=(15, 4 * len(self.metrics)))

		for ax, metric in zip(axs, self.metrics):
			ax.plot(self.metrics[metric], label=metric)
			ax.set_xscale("linear")
			ax.legend()

		plt.savefig(f"{config.SAVE_FOLDER_PATH}/metrics.png", dpi=200)

	def convert(self, labels, iou_threshold=0.45, conf_threshold=0.25, nbits=16, metadata_changes={}):
		num_classes = len(labels)
		num_boxes = len(self.default_boxes)
		
		metadata = {
			"image_input_description": "Input image",
			"iou_threshold_input_description": f"(optional) IOU threshold override (default: {str(iou_threshold)})",
			"conf_threshold_input_description": f"(optional) Confidence threshold override (default: {str(conf_threshold)})",
			"confidences_output_description": f"Found boxes × [conf_label1, conf_label2, ..., conf_label{num_classes}]",
			"locations_output_description": "Found boxes × [x, y, width, height] (relative to image size)",
			"general_description": "Object detector for Mahjong tiles",
			"author": "Mikael de Verdier",
			"additional": {}
		}
		metadata.update(metadata_changes)

		mlmodel = ct.convert(
			self.model,
			inputs=[ct.ImageType("input_1", shape=(1,) + self.input_shape[-2::-1] + self.input_shape[-1:])]
		)

		spec = mlmodel.get_spec()

		new_names = ["raw_confidence", "raw_coordinates"]
		output_sizes = [num_classes, 4]
		for i in range(2):
			old_name = spec.description.output[i].name
			ct.utils.rename_feature(spec, old_name, new_names[i])
			spec.description.output[i].type.multiArrayType.shape.extend([num_boxes, output_sizes[i]])
			spec.description.output[i].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE

		mlmodel = ct.models.MLModel(spec)

		nms_spec = ct.proto.Model_pb2.Model()
		nms_spec.specificationVersion = 5

		for i in range(2):
			decoder_output = spec.description.output[i].SerializeToString()

			nms_spec.description.input.add()
			nms_spec.description.input[i].ParseFromString(decoder_output)

			nms_spec.description.output.add()
			nms_spec.description.output[i].ParseFromString(decoder_output)

		nms_spec.description.output[0].name = "confidence"
		nms_spec.description.output[1].name = "coordinates"
		
		for i in range(2):
			ma_type = nms_spec.description.output[i].type.multiArrayType
			ma_type.shapeRange.sizeRanges.add()
			ma_type.shapeRange.sizeRanges[0].lowerBound = 0
			ma_type.shapeRange.sizeRanges[0].upperBound = -1
			ma_type.shapeRange.sizeRanges.add()
			ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
			ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
			del ma_type.shape[:]

		nms = nms_spec.nonMaximumSuppression
		nms.confidenceInputFeatureName = "raw_confidence"
		nms.coordinatesInputFeatureName = "raw_coordinates"
		nms.confidenceOutputFeatureName = "confidence"
		nms.coordinatesOutputFeatureName = "coordinates"
		nms.iouThresholdInputFeatureName = "iouThreshold"
		nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

		nms.iouThreshold = iou_threshold
		nms.confidenceThreshold = conf_threshold
		nms.pickTop.perClass = False
		nms.stringClassLabels.vector.extend(labels)

		nms_model = ct.models.MLModel(nms_spec)

		input_features = [
			("input_1", ct.models.datatypes.Array(0)),  # Doesn't matter, is changed later anyway
			("iouThreshold", ct.models.datatypes.Double()),
			("confidenceThreshold", ct.models.datatypes.Double())
		]

		output_features = ["confidence", "coordinates"]

		pipeline = ct.models.pipeline.Pipeline(input_features, output_features)

		pipeline.add_model(mlmodel)
		pipeline.add_model(nms_model)

		pipeline.spec.description.input[1].type.isOptional = True
		pipeline.spec.description.input[2].type.isOptional = True

		pipeline.spec.description.input[0].ParseFromString(spec.description.input[0].SerializeToString())
		pipeline.spec.description.output[0].ParseFromString(nms_spec.description.output[0].SerializeToString())
		pipeline.spec.description.output[1].ParseFromString(nms_spec.description.output[1].SerializeToString())

		pipeline.spec.description.input[0].shortDescription = metadata["image_input_description"]
		pipeline.spec.description.input[1].shortDescription = metadata["iou_threshold_input_description"]
		pipeline.spec.description.input[2].shortDescription = metadata["conf_threshold_input_description"]

		pipeline.spec.description.output[0].shortDescription = metadata["confidences_output_description"]
		pipeline.spec.description.output[1].shortDescription = metadata["locations_output_description"]

		pipeline.spec.description.metadata.shortDescription = metadata["general_description"]
		pipeline.spec.description.metadata.author = metadata["author"]

		pipeline.spec.description.metadata.userDefined.update(metadata["additional"])
		pipeline.spec.specificationVersion = 5

		ct_model = ct.models.MLModel(pipeline.spec)
		quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, nbits)

		quantized_model.save(f"{config.SAVE_FOLDER_PATH}/output_model.mlpackage")
