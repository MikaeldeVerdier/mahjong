from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Dropout, MaxPooling2D, Flatten, Dense, Softmax
from tensorflow.keras.regularizers import l2

def conv2D(filters, kernel_size, pooling_size, dropout_factor=0.5, uses_bias=True, l2_reg=1e-4):
	conv = Conv2D(filters=filters, kernel_size=kernel_size, padding="same", data_format="channels_last", use_bias=uses_bias, kernel_regularizer=l2(l2_reg))
	batch_norm = BatchNormalization()
	activation = ReLU()
	dropout = Dropout(dropout_factor)
	pooling = MaxPooling2D(pooling_size)

	def convlutional_layer(x):
		x = conv(x)
		x = batch_norm(x)
		x = activation(x)
		x = dropout(x)
		x = pooling(x)

		return x
	
	return convlutional_layer


def flatten():
	flat = Flatten()

	def flatten_layer(x):
		x = flat(x)

		return x

	return flatten_layer


def dense1D(neuron_amount, dropout_factor=0.5, uses_bias=True, l2_reg=1e-4):
	dense = Dense(neuron_amount, activation="linear", use_bias=uses_bias, kernel_regularizer=l2(l2_reg))
	batch_norm = BatchNormalization()
	activation = ReLU()
	dropout = Dropout(dropout_factor)

	def dense_layer(x):
		x =	dense(x)
		x = batch_norm(x)
		x = activation(x)
		x = dropout(x)

		return x

	return dense_layer


def dense_only(neuron_amount, activation_func="linear", uses_bias=True, l2_reg=1e-4):
	dense = Dense(neuron_amount, activation=activation_func, use_bias=uses_bias,  kernel_regularizer=l2(l2_reg))

	def dense_layer(x):
		x = dense(x)

		return x
	
	return dense_layer
