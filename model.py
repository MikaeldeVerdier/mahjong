import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import coremltools as ct

from keras.utils import plot_model
from keras.applications import VGG16
# from keras.callbacks import TensorBoard
from keras.layers import Activation, Concatenate, Conv2D, Reshape
from keras.losses import CategoricalCrossentropy, Huber
# from keras.metrics import MeanSquaredError, Accuracy
from keras.models import Model, load_model
from keras.optimizers import SGD

import config
from default_box import default_boxes, CellBox

class SSD_Model:  # Consider instead saving weights, and using a seperate training and inference model (to decode in model)
	def __init__(self, input_shape, class_amount, lr=config.LEARNING_RATE, momentum=config.MOMENTUM, hard_neg_ratio=config.HARD_NEGATIVE_RATIO, load=False):
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
		im_aspect_ratio = input_shape[0] / input_shape[1]
		aspect_ratios = [ar / im_aspect_ratio for ar in [0.67, 1, 1.33]]

		head_outputs = [[], []]
		for k, output in enumerate(outputs, 1):
			defaults = default_boxes(k, len(outputs), aspect_ratios, output.shape[1:3])
			for default in defaults:
				self.default_boxes += default.cells

			location_pred = Conv2D(filters=len(defaults) * 4, kernel_size=(3, 3), padding="same")(output)
			location_pred = Reshape((-1, 4))(location_pred)
			head_outputs[0].append(location_pred)

			class_pred = Conv2D(filters=len(defaults) * (class_amount + 1), kernel_size=(3, 3), padding="same")(output)
			class_pred = Reshape((-1, class_amount + 1))(class_pred)
			class_pred = Activation("softmax")(class_pred)
			head_outputs[1].append(class_pred)

		location_predictions = Concatenate(axis=1, name="locations")(head_outputs[0])
		class_predictions = Concatenate(axis=1, name="confidences")(head_outputs[1])

		self.model = Model(inputs=[base_network.input], outputs=[class_predictions, location_predictions])
		self.model.compile(loss={"locations": self.huber_with_mask, "confidences": self.categorical_crossentropy_with_mask}, optimizer=SGD(learning_rate=lr, momentum=momentum), loss_weights={"confidences": 1, "locations": 100})  # , metrics={"locations": MeanSquaredError(), "confidences": Accuracy()})

		self.plot_model()
		self.model.summary()

		self.metrics = {"loss": [], "locations_loss": [], "confidences_loss": [], "val_locations_loss": [], "val_confidences_loss": []}  # , "locations_mean_squared_error": [], "confidences_accuracy": []}

	def huber_with_mask(self, y_true, y_pred):
		pos_losses = Huber(reduction="none")(y_true, y_pred)

		pos_mask = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)
		pos_multiplier = tf.cast(pos_mask, tf.float32)
		pos_losses *= pos_multiplier

		# neg_mask = tf.logical_not(pos_mask)
		# neg_multiplier = tf.cast(neg_mask, tf.float32)
		# neg_losses *= neg_multiplier

		# sorted_neg_losses = tf.sort(neg_losses, direction="DESCENDING")  # In SSD, this is traditionally sorted by CONFIDENCE loss, not locations.
		# ks = tf.expand_dims(tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1), axis=-1)

		# indices = tf.range(tf.shape(sorted_neg_losses)[-1])
		# indices_expanded = tf.expand_dims(indices, axis=0)
		# mask_multiplier = tf.where(indices_expanded <= ks, tf.ones_like(indices_expanded, dtype=tf.float32), tf.zeros_like(indices_expanded, dtype=tf.float32))
		# top_neg_losses = mask_multiplier * sorted_neg_losses

		# loss = tf.reduce_sum(tf.concat([pos_losses, top_neg_losses], axis=-1), axis=-1)

		loss = tf.reduce_sum(pos_losses, axis=-1)

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
		ks = tf.expand_dims(tf.reduce_sum(tf.cast(pos_mask, tf.int32), axis=-1) * tf.constant(self.hard_neg_ratio), axis=-1)

		indices = tf.range(tf.shape(sorted_neg_losses)[-1])
		indices_expanded = tf.expand_dims(indices, axis=0)
		mask_tensor = tf.where(indices_expanded <= ks, tf.ones_like(indices_expanded, dtype=tf.float32), tf.zeros_like(indices_expanded, dtype=tf.float32))
		top_neg_losses = mask_tensor * sorted_neg_losses

		loss = tf.reduce_sum(tf.concat([pos_losses, top_neg_losses], axis=-1), axis=-1)

		return loss

	"""
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
	"""

	def match_boxes(self, gt_boxes, threshold=0.5):  # Consider moving this to default_box.py
		if not len(gt_boxes):
			return []
		
		matches = []
		for i, gt_box in enumerate(gt_boxes):
			gt_ious = [box.calculate_iou(gt_box) for box in self.default_boxes]

			matches.append((np.argmax(gt_ious), i))

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

		callbacks = []  # [TensorBoard(log_dir=f"{config.SAVE_FOLDER_PATH}/logs", histogram_freq=1, write_graph=True, write_images=True, update_freq="epoch", profile_batch=2, embeddings_freq=1)]
		fit = self.model.fit(x, y, epochs=epochs, validation_split=config.VALIDATION_SPLIT, callbacks=callbacks)

		for metric in fit.history:
			if metric in self.metrics:
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

	def create_base_model(self):
		mlmodel = ct.convert(
			self.model,
			inputs=[ct.ImageType("input_1", shape=(1,) + self.input_shape[-2::-1] + self.input_shape[-1:])]
		)

		spec = mlmodel.get_spec()

		new_names = ["confidences", "locations"]
		output_sizes = [self.class_amount + 1, 4]
		for i in range(2):
			old_name = spec.description.output[i].name
			ct.utils.rename_feature(spec, old_name, new_names[i])
			spec.description.output[i].type.multiArrayType.shape.extend([1, len(self.default_boxes), output_sizes[i]])
			spec.description.output[i].type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE

		mlmodel = ct.models.MLModel(spec)

		return mlmodel

	"""
	def build_decoder(self, num_classes, num_boxes):
		input_features = [
			("confidences", ct.models.datatypes.Array(1, num_boxes, num_classes + 1)),
			("locations", ct.models.datatypes.Array(1, num_boxes, 4))
		]

		output_features = [
			("raw_confidence", ct.models.datatypes.Array(num_boxes, num_classes)),
			("raw_coordinates", ct.models.datatypes.Array(num_boxes, 4))
		]

		builder = ct.models.neural_network.NeuralNetworkBuilder(input_features, output_features)

		# builder.add_permute(name="permute_scores", dim=(0, 1, 2, 3), input_name="scores", output_name="permute_scores_output")
		builder.add_slice(name="slice_scores_layer", input_name="confidences", output_name="raw_confidence", axis="width", start_index=1, end_index=num_classes + 1)

		builder.add_permute(name="permute_locations_layer", dim=(0, 3, 2, 1), input_name="locations", output_name="permute_locations")		
		builder.add_slice(name="slice_wh_layer", input_name="permute_locations", output_name="slice_wh", axis="channel", start_index=2, end_index=4)
		builder.add_slice(name="slice_xy_layer", input_name="permute_locations", output_name="slice_xy", axis="channel", start_index=0, end_index=2)
		
		defaults_xy, defaults_wh = zip(*[(default_box.size_coords[:2], default_box.size_coords[2:]) for default_box in self.default_boxes])

		defaults_xy = np.moveaxis(defaults_xy, 0, 1)
		defaults_wh = np.moveaxis(defaults_wh, 0, 1)
		defaults_xy = np.expand_dims(defaults_xy, axis=-1)
		defaults_wh = np.expand_dims(defaults_wh, axis=-1)

		builder.add_load_constant(name="defaults_xy_layer", output_name="defaults_xy", constant_value=defaults_xy, shape=[2, num_boxes, 1])
		builder.add_load_constant(name="defaults_wh_layer", output_name="defaults_wh", constant_value=defaults_wh, shape=[2, num_boxes, 1])

		builder.add_elementwise(name="xw_times_yh_layer", input_names=["slice_xy", "defaults_wh"], output_name="yw_times_wh", mode="MULTIPLY")
		builder.add_elementwise(name="decoded_xy_layer", input_names=["yw_times_wh", "defaults_xy"], output_name="decoded_xy", mode="ADD")

		builder.add_unary(name="exp_wh_layer", input_name="slice_wh", output_name="exp_wh", mode="exp")
		builder.add_elementwise(name="decoded_wh_layer", input_names=["exp_wh", "defaults_wh"], output_name="decoded_wh", mode="MULTIPLY")

		builder.add_slice(name="slice_x_layer", input_name="decoded_xy", output_name="slice_x", axis="channel", start_index=0, end_index=1)
		builder.add_slice(name="slice_y_layer", input_name="decoded_xy", output_name="slice_y", axis="channel", start_index=1, end_index=2)
		builder.add_slice(name="slice_w_layer", input_name="decoded_wh", output_name="slice_w", axis="channel", start_index=0, end_index=1)
		builder.add_slice(name="slice_h_layer", input_name="decoded_wh", output_name="slice_h", axis="channel", start_index=1, end_index=2)

		builder.add_elementwise(name="concat_layer", input_names=["slice_x", "slice_y", "slice_w", "slice_h"], output_name="concat", mode="CONCAT")

		builder.add_permute(name="permute_concat_layer", dim=(0, 3, 2, 1), input_name="concat", output_name="raw_coordinates")

		decoder_model = ct.models.MLModel(builder.spec)

		return decoder_model
	"""

	def build_decoder(self):
		input_features = [
			("confidences", ct.models.datatypes.Array(1, len(self.default_boxes), self.class_amount + 1)),
			("locations", ct.models.datatypes.Array(1, len(self.default_boxes), 4))
		]

		output_features = [
			("raw_confidence", ct.models.datatypes.Array(len(self.default_boxes), self.class_amount)),
			("raw_coordinates", ct.models.datatypes.Array(len(self.default_boxes), 4))
		]

		builder = ct.models.neural_network.NeuralNetworkBuilder(input_features, output_features)

		builder.add_slice(name="slice_scores_layer", input_name="confidences", output_name="raw_confidence", axis="width", start_index=1, end_index=self.class_amount + 1)

		builder.add_slice(name="slice_wh_layer", input_name="locations", output_name="slice_wh", axis="width", start_index=2, end_index=4)
		builder.add_slice(name="slice_xy_layer", input_name="locations", output_name="slice_xy", axis="width", start_index=0, end_index=2)
		
		defaults_xy, defaults_wh = zip(*[(default_box.size_coords[:2], default_box.size_coords[2:]) for default_box in self.default_boxes])

		defaults_xy = np.expand_dims(defaults_xy, axis=0)
		defaults_wh = np.expand_dims(defaults_wh, axis=0)

		# defaults_xy = []
		# defaults_wh = []
		# for default_box in self.default_boxes:
		# 	defaults_xy.append(default_box.size_coords[:2])
		# 	defaults_wh.append(default_box.size_coords[2:])
		# builder.add_slice(name="slice_scores_layer", input_name="confidences", output_name="raw_confidence", axis="width", start_index=1, end_index=num_classes + 1)

		builder.add_load_constant(name="defaults_xy_layer", output_name="defaults_xy", constant_value=defaults_xy, shape=[1, len(self.default_boxes), 2])
		builder.add_load_constant(name="defaults_wh_layer", output_name="defaults_wh", constant_value=defaults_wh, shape=[1, len(self.default_boxes), 2])

		builder.add_elementwise(name="xw_times_yh_layer", input_names=["slice_xy", "defaults_wh"], output_name="yw_times_wh", mode="MULTIPLY")
		builder.add_elementwise(name="decoded_xy_layer", input_names=["yw_times_wh", "defaults_xy"], output_name="decoded_xy", mode="ADD")

		builder.add_unary(name="exp_wh_layer", input_name="slice_wh", output_name="exp_wh", mode="exp")
		builder.add_elementwise(name="decoded_wh_layer", input_names=["exp_wh", "defaults_wh"], output_name="decoded_wh", mode="MULTIPLY")

		builder.add_slice(name="slice_x_layer", input_name="decoded_xy", output_name="slice_x", axis="width", start_index=0, end_index=1)
		builder.add_slice(name="slice_y_layer", input_name="decoded_xy", output_name="slice_y", axis="width", start_index=1, end_index=2)
		builder.add_slice(name="slice_w_layer", input_name="decoded_wh", output_name="slice_w", axis="width", start_index=0, end_index=1)
		builder.add_slice(name="slice_h_layer", input_name="decoded_wh", output_name="slice_h", axis="width", start_index=1, end_index=2)

		builder.add_elementwise(name="concat_layer", input_names=["slice_x", "slice_y", "slice_w", "slice_h"], output_name="concat", mode="CONCAT")  # Concats along the first axis, even though the slices were along the last, which is why a permutation layer is needed.

		builder.add_permute(name="permute_output", dim=(0, 3, 2, 1), input_name="concat", output_name="raw_coordinates")

		decoder_model = ct.models.MLModel(builder.spec)

		return decoder_model

	def create_nms_model(self, previous_model, iou_threshold, conf_threshold, labels):
		nms_spec = ct.proto.Model_pb2.Model()
		nms_spec.specificationVersion = 5

		for i in range(2):
			decoder_output = previous_model.get_spec().description.output[i].SerializeToString()

			nms_spec.description.input.add()
			nms_spec.description.input[i].ParseFromString(decoder_output)

			nms_spec.description.output.add()
			nms_spec.description.output[i].ParseFromString(decoder_output)

		nms_spec.description.output[0].name = "confidence"
		nms_spec.description.output[1].name = "coordinates"
		
		nms_output_sizes = [self.class_amount, 4]
		for i in range(2):
			ma_type = nms_spec.description.output[i].type.multiArrayType
			ma_type.shapeRange.sizeRanges.add()
			ma_type.shapeRange.sizeRanges[0].lowerBound = 0
			ma_type.shapeRange.sizeRanges[0].upperBound = -1
			ma_type.shapeRange.sizeRanges.add()
			ma_type.shapeRange.sizeRanges[1].lowerBound = nms_output_sizes[i]
			ma_type.shapeRange.sizeRanges[1].upperBound = nms_output_sizes[i]
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

		return nms_model
	
	def create_pipeline(self, models):
		input_features = [
			("input_1", ct.models.datatypes.Array(0)),  # Doesn't matter, is changed later anyway
			("iouThreshold", ct.models.datatypes.Double()),
			("confidenceThreshold", ct.models.datatypes.Double())
		]

		output_features = ["confidence", "coordinates"]

		pipeline = ct.models.pipeline.Pipeline(input_features, output_features)

		for model in models:
			pipeline.add_model(model)

		# pipeline.add_model(mlmodel)
		# pipeline.add_model(decoder_model)
		# pipeline.add_model(nms_model)

		pipeline.spec.description.input[1].type.isOptional = True
		pipeline.spec.description.input[2].type.isOptional = True

		pipeline.spec.description.input[0].ParseFromString(models[0].get_spec().description.input[0].SerializeToString())
		pipeline.spec.description.output[0].ParseFromString(models[-1].get_spec().description.output[0].SerializeToString())
		pipeline.spec.description.output[1].ParseFromString(models[-1].get_spec().description.output[1].SerializeToString())

		pipeline.spec.specificationVersion = 5

		pipeline_model = ct.models.MLModel(pipeline.spec)
		
		return pipeline_model

	def convert_to_mlmodel(self, labels, iou_threshold=0.45, conf_threshold=0.25):
		self.iou_threshold = iou_threshold
		self.conf_threshold = conf_threshold

		mlmodel = self.create_base_model()
		decoder_model = self.build_decoder()
		nms_model = self.create_nms_model(decoder_model, iou_threshold, conf_threshold, labels)
		
		self.mlmodel = self.create_pipeline([mlmodel, decoder_model, nms_model])
	
	def save_mlmodel(self, metadata_changes={}, precision_nbits=16):
		pipeline_spec = self.mlmodel.get_spec()

		num_classes = pipeline_spec.description.output[0].type.multiArrayType.shapeRange.sizeRanges[1].lowerBound

		metadata = {
			"image_input_description": "Input image",
			"iou_threshold_input_description": f"(optional) IOU threshold override (default: {str(self.iou_threshold)})",
			"conf_threshold_input_description": f"(optional) Confidence threshold override (default: {str(self.conf_threshold)})",
			"confidences_output_description": f"Found boxes × [conf_label1, conf_label2, ..., conf_label{num_classes}]",
			"locations_output_description": "Found boxes × [x, y, width, height] (relative to image size)",
			"general_description": "Object detector for Mahjong tiles",
			"author": "Mikael de Verdier",
			"additional": {}
		}
		metadata |= metadata_changes

		pipeline_spec.description.input[0].shortDescription = metadata["image_input_description"]
		pipeline_spec.description.input[1].shortDescription = metadata["iou_threshold_input_description"]
		pipeline_spec.description.input[2].shortDescription = metadata["conf_threshold_input_description"]

		pipeline_spec.description.output[0].shortDescription = metadata["confidences_output_description"]
		pipeline_spec.description.output[1].shortDescription = metadata["locations_output_description"]

		pipeline_spec.description.metadata.shortDescription = metadata["general_description"]
		pipeline_spec.description.metadata.author = metadata["author"]

		pipeline_spec.description.metadata.userDefined.update(metadata["additional"])

		ct_model = ct.models.MLModel(pipeline_spec)

		quantized_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, precision_nbits)
		quantized_model.save(f"{config.SAVE_FOLDER_PATH}/output_model.mlpackage")
