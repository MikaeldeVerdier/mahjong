import numpy as np
import tensorflow as tf
from keras.layers import Layer

class L2Normalization(Layer):
    def __init__(self, gamma_init=1, axis=-1):
        self.axis = axis
        self.gamma_init = gamma_init

        super(L2Normalization, self).__init__()

    def build(self, input_shape):
        gamma = self.gamma_init * np.ones((input_shape[self.axis],), dtype=np.float32)
        self.gamma = tf.Variable(gamma, trainable=True)

        super(L2Normalization, self).build(input_shape)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, self.axis) * self.gamma
