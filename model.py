import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import coremltools as ct
from keras.utils import plot_model
# from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
# from keras.callbacks import TensorBoard
from keras.layers import Input, Activation, Concatenate, Conv2D, Reshape, MaxPooling2D, ZeroPadding2D
from keras.losses import CategoricalCrossentropy, Huber
# from keras.metrics import MeanSquaredError, Accuracy
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.regularizers import L2
from tensorflow.python.keras.utils import data_utils

import box_utils
import config
from l2_norm import L2Normalization

class SSD_Model:  # Consider instead saving weights, and using a seperate training and inference model (to decode in model)
	def __init__(self, input_shape, class_amount, lr=config.LEARNING_RATE, momentum=config.MOMENTUM, hard_neg_ratio=config.HARD_NEGATIVE_RATIO, alpha=config.ALPHA, load=False):
		self.input_shape = input_shape
		self.class_amount = class_amount
		self.hard_neg_ratio = hard_neg_ratio
		self.alpha = alpha

		self.preprocess_function = preprocess_input

		if load is not False:
			self.load_model(load)
		else:
			self.build_model(lr, momentum)

	def build_base(self, kernel_initializer=None):
		""" A truncated version of VGG16 configuration D """
		# Credit: https://github.com/Socret360/object-detection-in-keras/blob/master/networks/base_networks/truncated_vgg16.py

		input_layer = Input(shape=self.input_shape, name="input")

		processed_input = self.preprocess_function(input_layer)

		conv1_1 = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(processed_input)
		conv1_2 = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(conv1_1)
		pool1 = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool", padding="same")(conv1_2)

		conv2_1 = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(pool1)
		conv2_2 = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(conv2_1)
		pool2 = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool", padding="same")(conv2_2)

		conv3_1 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(pool2)
		conv3_2 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(conv3_1)
		conv3_3 = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(conv3_2)
		pool3 = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool", padding="same")(conv3_3)

		conv4_1 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(pool3)
		conv4_2 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(conv4_1)
		conv4_3 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(conv4_2)
		pool4 = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool", padding="same")(conv4_3)

		conv5_1 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(pool4)
		conv5_2 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(conv5_1)
		conv5_3 = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(conv5_2)

		model = Model(inputs=input_layer, outputs=conv5_3)

		weights_path_no_top = ("https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
		weights_path = data_utils.get_file("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", weights_path_no_top, cache_subdir="models", file_hash="6d6bbae143d832006294945121d1f1fc")

		model.load_weights(weights_path, by_name=True)

		return model

	def build_model(self, learning_rate, momentum):
		kernel_initializer = "he_normal"

		# base_network = VGG16(include_top=False, weights="imagenet", input_shape=self.input_shape)
		# base_network.layers[-1].pool_size = (3, 3)
		# base_network.layers[-1].strides = (1, 1)

		base_network = self.build_base(kernel_initializer=kernel_initializer)
		base_network.trainable = False
		# for layer in base_network.layers:
		# 	layer.trainable = False

		outputs = []

		x = base_network.get_layer("block4_conv3").output
		x = L2Normalization(gamma_init=20)(x)
		outputs.append(x)

		x = base_network.get_layer("block5_conv3").output

		x = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)

		# Auxiliary layers
		x = Conv2D(filters=1024, kernel_size=(3, 3), padding="same", dilation_rate=(6, 6), activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(x)  # tf.nn.atrous_conv2d (https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d)?
		x = Conv2D(filters=1024, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(x)
		outputs.append(x)

		x = Conv2D(filters=256, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(x)
		x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
		x = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(x)
		outputs.append(x)

		x = Conv2D(filters=128, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(x)
		x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(x)
		outputs.append(x)

		x = Conv2D(filters=128, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(x)
		outputs.append(x)

		x = Conv2D(filters=128, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(x)
		x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(x)
		outputs.append(x)
		#

		self.default_boxes = np.empty(shape=(0, 4))
		im_aspect_ratio = self.input_shape[1] / self.input_shape[0]

		aspect_ratios = [
			(1, 2, 0.5),
			(1, 2, 0.5, 3, 1 / 3),
			(1, 2, 0.5, 3, 1 / 3),
			(1, 2, 0.5, 3, 1 / 3),
			(1, 2, 0.5),
			(1, 2, 0.5)
		]

		# def calc_scale(min_scale, max_scale):
		# 	return np.append(np.linspace(min_scale, max_scale, len(outputs)), [min_scale + (max_scale - min_scale) / (len(outputs) - 1)])
		scales = [
			[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],  # from luigi (mine: s(0.1, 0.9))
			[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
			[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
			[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
			[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
			[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
		]

		def calc_step(val):
			return [val / self.input_shape[0], val / self.input_shape[1]]
		steps = [
			calc_step(8),  # from luigi (mine: None)
			calc_step(16),
			calc_step(32),
			calc_step(64),
			calc_step(100),
			calc_step(300),
		]

		head_outputs = [[], []]
		for k, (output, aspect_ratio, scale, step) in enumerate(zip(outputs, aspect_ratios, scales, steps), 1):
			defaults = box_utils.default_boxes(k - 1, aspect_ratio, scale, output.shape[1:3], steps=step, im_aspect_ratio=im_aspect_ratio)
			# defaults = default_boxes(k, len(outputs), aspect_ratios, output.shape[1:3])
			self.default_boxes = np.concatenate([self.default_boxes, defaults.reshape(-1, 4)])

			location_pred = Conv2D(filters=defaults.shape[1] * 4, kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(output)
			location_pred = Reshape((-1, 4))(location_pred)
			head_outputs[0].append(location_pred)

			class_pred = Conv2D(filters=defaults.shape[1] * (self.class_amount + 1), kernel_size=(3, 3), padding="same", kernel_initializer=kernel_initializer, kernel_regularizer=L2(config.L2_REG))(output)
			class_pred = Reshape((-1, self.class_amount + 1))(class_pred)
			class_pred = Activation("softmax")(class_pred)
			head_outputs[1].append(class_pred)

		location_predictions = Concatenate(axis=1)(head_outputs[0])
		class_predictions = Concatenate(axis=1)(head_outputs[1])

		output = Concatenate(axis=-1)([class_predictions, location_predictions])

		self.model = Model(inputs=[base_network.input], outputs=[output])
		self.model.compile(loss=self.ssd_loss, optimizer=SGD(learning_rate=learning_rate, momentum=momentum))

		self.plot_model()
		self.model.summary()

		self.metrics = {"loss": [], "val_loss": []}  # , "locations_mean_squared_error": [], "confidences_accuracy": []}

	def log_loss(self, y_true, y_pred):
		# Credit: https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py

		y_pred = tf.maximum(y_pred, 1e-15)
		log_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

		return log_loss

	def smooth_L1_loss(self, y_true, y_pred):
		# Credit: https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_loss_function/keras_ssd_loss.py

		absolute_loss = tf.abs(y_true - y_pred)
		square_loss = 0.5 * (y_true - y_pred) ** 2
		l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)

		return tf.reduce_sum(l1_loss, axis=-1)  # ALTERNATIVELY: tf.reduce_mean(l1_loss, axis=-1)

	def ssd_loss(self, y_true, y_pred):
		y_true = tf.cast(y_true, tf.float32)
		y_pred = tf.cast(y_pred, tf.float32)

		cls_loss = self.log_loss(y_true[:, :, :-4], y_pred[:, :, :-4])
		loc_loss = self.smooth_L1_loss(y_true[:, :, -4:], y_pred[:, :, -4:])

		batch_size = tf.shape(y_true)[0]
		num_boxes = tf.shape(y_true)[1]

		pos_mask = tf.reduce_max(y_true[:, :, 1:-4], axis=-1)
		cls_pos_losses = cls_loss * pos_mask

		neg_mask = y_true[:, :, 0]
		cls_neg_losses = cls_loss * neg_mask

		cls_neg_losses_1d = tf.reshape(cls_neg_losses, [-1])
		cls_sorted_neg_indices = tf.argsort(cls_neg_losses_1d, direction="DESCENDING")

		pos_amount = tf.reduce_sum(pos_mask)
		k = tf.cast(pos_amount, tf.int32) * tf.constant(self.hard_neg_ratio)

		cls_top_neg_indices = cls_sorted_neg_indices[:k]
		cls_top_neg_mask = tf.scatter_nd(cls_top_neg_indices[:, None], tf.ones_like(cls_top_neg_indices, tf.float32), tf.shape(cls_neg_losses_1d))

		cls_top_neg_mask_2d = tf.reshape(cls_top_neg_mask, [batch_size, num_boxes])
		cls_top_neg_losses_2d = cls_neg_losses * cls_top_neg_mask_2d

		cls_pos_loss = tf.reduce_sum(cls_pos_losses, axis=-1)
		cls_neg_loss = tf.reduce_sum(cls_top_neg_losses_2d, axis=-1)

		loc_pos_losses = loc_loss * pos_mask
		loc_loss = tf.reduce_sum(loc_pos_losses, axis=-1)

		loss = (cls_pos_loss + cls_neg_loss + tf.constant(self.alpha, dtype=tf.float32) * loc_loss) / tf.maximum(pos_amount, 1.0)
		# loss *= tf.cast(pos_amount != 0, tf.float32)  # Shouldn't loss be 0 when there are no positives (according to paper)
		loss *= tf.cast(batch_size, tf.float32)  # Counteracts Keras' batch loss average, only pos_amount matters (which is already divided by). Should still almost only matter when varying batch size.

		return loss

	"""
	def ssd_loss(self, y_true, y_pred):
		y_true = tf.cast(y_true, tf.float32)
		y_pred = tf.cast(y_pred, tf.float32)

		cls_loss = self.log_loss(y_true[:, :, :-4], y_pred[:, :, :-4])  # CategoricalCrossentropy(reduction="none")(y_true[:, :, :-4], y_pred[:, :, :-4])  #
		loc_loss = self.smooth_L1_loss(y_true[:, :, -4:], y_pred[:, :, -4:])  # Huber(reduction="none")(y_true[:, :, -4:], y_pred[:, :, -4:])  # 

		batch_size = tf.shape(y_true)[0]
		num_boxes = tf.shape(y_true)[1]

		pos_mask = tf.reduce_max(y_true[:, :, 1:-4], axis=-1)  # is not ~neg_mask because of neutral boxes
		cls_pos_losses = cls_loss * pos_mask

		neg_mask = y_true[:, :, 0]
		cls_neg_losses = cls_loss * neg_mask

		flat_confs = tf.reshape(y_pred[:, :, :-4], (-1, self.class_amount + 1))
		labels = tf.argmax(y_true[:, :, :-4], axis=-1)

		max_conf = tf.reduce_max(flat_confs)
		log_sum_exp = tf.math.log(tf.reduce_sum(tf.exp(flat_confs - max_conf), axis=1, keepdims=True)) + max_conf

		confs_for_labels = tf.gather(flat_confs, tf.reshape(labels, (-1, 1)), batch_dims=1)
		flat_loss_est = log_sum_exp - confs_for_labels

		loss_est = tf.reshape(flat_loss_est, (batch_size, num_boxes))
		neg_loss_est = loss_est * neg_mask

		top_indices = tf.argsort(neg_loss_est, direction="DESCENDING")  # _, ind = tf.nn.top_k(loss_c, k=num_boxes)
		top_negs = tf.gather(cls_neg_losses, top_indices, batch_dims=1)

		pos_amount = tf.reduce_sum(pos_mask, axis=-1)
		neg_amount = tf.minimum(tf.reduce_sum(neg_mask, axis=-1), pos_amount * self.hard_neg_ratio)

		range_indices = tf.range(0, num_boxes)
		top_neg_mask = range_indices[None] < tf.cast(neg_amount, tf.int32)[:, None]
		top_neg_losses = top_negs * tf.cast(top_neg_mask, tf.float32)

		cls_pos_loss = tf.reduce_sum(cls_pos_losses)  # , axis=-1)
		cls_neg_loss = tf.reduce_sum(top_neg_losses)  # , axis=-1)

		loc_pos_losses = loc_loss * pos_mask
		loc_pos_loss = tf.reduce_sum(loc_pos_losses)  # , axis=-1)

		tot_pos_amount = tf.reduce_sum(pos_amount)
		loss = (cls_pos_loss + cls_neg_loss + tf.cast(self.alpha, tf.float32) * loc_pos_loss) / tf.maximum(tot_pos_amount, 1.0)  # Not really needed, pos_amount can't be 0 in this implementation
		# loss *= tf.cast(pos_amount != 0, tf.float32)  # Shouldn't loss be 0 when there are no positives (according to paper)
		# loss *= tf.cast(batch_size, tf.float32)  # Counteracts Keras' batch loss average, only pos_amount matters (which is already divided by). Should still almost only matter when varying batch size.

		# loss = tf.reduce_mean(loss)

		return loss
	"""

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
				self.metrics[metric] += fit.history[metric]  # Could be changed to list comprehension

	def save_model(self, name):
		self.model.save(f"{config.SAVE_FOLDER_PATH}/{name}")

		with open(f"{config.SAVE_FOLDER_PATH}/save.json", "w") as f:
			f.write(json.dumps(self.metrics))

		with open(f"{config.SAVE_FOLDER_PATH}/default_boxes.json", "w") as f:
			f.write(json.dumps(self.default_boxes.tolist()))

	def load_model(self, name):
		self.model = load_model(f"{config.SAVE_FOLDER_PATH}/{name}", custom_objects={"ssd_loss": self.ssd_loss})

		with open(f"{config.SAVE_FOLDER_PATH}/save.json", "r") as f:
			self.metrics = json.loads(f.read())

		with open(f"{config.SAVE_FOLDER_PATH}/default_boxes.json", "r") as f:
			reading = json.loads(f.read())
			self.default_boxes = np.array(reading)

	def plot_model(self):
		try:
			plot_model(self.model, to_file=f"{config.SAVE_FOLDER_PATH}/model_architecture.png", show_shapes=True, show_layer_names=True)
		except ImportError:
			print("You need to install pydot and graphviz to plot model architecture.")

	def plot_metrics(self):  # Consider adding labels for axis and such
		_, axs = plt.subplots(len(self.metrics), figsize=(15, 4 * len(self.metrics)))

		for ax, metric in zip(axs, self.metrics):
			ax.plot(self.metrics[metric], label=metric)
			ax.set_xscale("linear")
			ax.legend()

		plt.savefig(f"{config.SAVE_FOLDER_PATH}/metrics.png", dpi=200)
		plt.close()

	def create_decoder_model(self, variances):
		conf_inp = Input(shape=(len(self.default_boxes), self.class_amount + 1), name="confidencesInput")
		offset_inp = Input(shape=(len(self.default_boxes), 4), name="locationsInput")

		confs = conf_inp[:, :, 1:]
		xy = offset_inp[:, :, :2]
		wh = offset_inp[:, :, 2:]

		defaults_xy = self.default_boxes[:, :2][None]
		defaults_wh = self.default_boxes[:, 2:][None]
		tensor_def_xy = tf.constant(defaults_xy, dtype="float32")
		tensor_def_wh = tf.constant(defaults_wh, dtype="float32")

		variances_xy = tf.constant(variances[:2])
		variances_wh = tf.constant(variances[2:])

		decoded_xy = xy * tensor_def_wh * variances_xy + tensor_def_xy
		decoded_wh = tf.exp(wh * variances_wh) * tensor_def_wh

		# locs = Concatenate(axis=-1)([decoded_yx, decoded_hw])  # Convert back to (x, y, w, h)
		locs = Concatenate(axis=-1)([decoded_xy, decoded_wh])

		decoder_model = Model(inputs=[conf_inp, offset_inp], outputs=[confs, locs], name="decoderModel")
		decoder_model.compile()

		return decoder_model

	def create_decoded_tfmodel(self, variances):  # Not actually necessary, could just add them together in pipeline
		inp = Input(shape=self.input_shape, name="image")
		# x = self.preprocess_function(inp)

		preds = self.model(inp)
		class_predictions = preds[:, :, :-4]
		location_predictions = preds[:, :, -4:]

		decoder_model = self.create_decoder_model(variances)
		confs, locs = decoder_model([class_predictions, location_predictions])

		decoded_model = Model(inputs=[inp], outputs=[confs, locs])
		decoded_model.compile()

		return decoded_model

	def create_base_model(self, variances):  # A bit weird to have partial generality. Should really probably be a staticmethod (same for all of these)
		decoded_model = self.create_decoded_tfmodel(variances)

		mlmodel = ct.convert(decoded_model, inputs=[ct.ImageType("image", shape=(1,) + self.input_shape)])

		spec = mlmodel.get_spec()

		new_names = ["raw_confidence", "raw_coordinates"]
		output_sizes = [self.class_amount, 4]
		for i in range(2):
			old_name = spec.description.output[i].name
			ct.utils.rename_feature(spec, old_name, new_names[i])
			spec.description.output[i].type.multiArrayType.shape.extend([len(self.default_boxes), output_sizes[i]])
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

	""" # No longer needed after creating decoding in tensorflow (create_decoder_tfmodel)
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
	"""

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
	
	def create_pipeline(self, models):  # Very partially general (weird)
		input_features = [
			("image", ct.models.datatypes.Array(0)),  # Doesn't matter, is changed later anyway
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

	def convert_to_mlmodel(self, labels, iou_threshold=0.45, conf_threshold=0.25, variances=config.VARIANCES):
		self.iou_threshold = iou_threshold
		self.conf_threshold = conf_threshold

		# mlmodel = self.create_base_model()
		# decoder_model = self.build_decoder()
		mlmodel = self.create_base_model(variances)
		nms_model = self.create_nms_model(mlmodel, iou_threshold, conf_threshold, labels)
		
		self.mlmodel = self.create_pipeline([mlmodel, nms_model])
	
	def save_mlmodel(self, metadata_changes={}, precision_nbits=16, name="output_model"):
		pipeline_spec = self.mlmodel.get_spec()

		# num_classes = pipeline_spec.description.output[0].type.multiArrayType.shapeRange.sizeRanges[1].lowerBound

		metadata = {
			"image_input_description": "Input image",
			"iou_threshold_input_description": f"(optional) IOU threshold override (default: {str(self.iou_threshold)})",
			"conf_threshold_input_description": f"(optional) Confidence threshold override (default: {str(self.conf_threshold)})",
			"confidences_output_description": f"Found boxes × [class_label1, class_label2, ..., class_label{self.class_amount}]",
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
		quantized_model.save(f"{config.SAVE_FOLDER_PATH}/{name}.mlpackage")

	def inference(self, image, labels, iou_threshold=0.45, confidence_threshold=0.25):  # PIL Image
		locations, confidences = self.mlmodel.predict({"image": image, "iouThreshold": iou_threshold, "confidenceThreshold": confidence_threshold}).values()
		predicted_labels = np.array(labels)[np.argmax(confidences, axis=-1)]
		# scaled_boxes = box_utils.scale_box(locations, input_shape[:-1])
		label_confs = np.max(confidences, axis=-1)

		label_infos = [predicted_labels, locations, label_confs]

		return label_infos
