import tensorflow as tf

"""
Code from: https://github.com/Hakuyume/chainer-ssd/blob/master/lib/multibox_loss.py
Rewritten with the help of ChatGPT
"""

def _elementwise_softmax_cross_entropy(x, t):
		assert x.shape[:-1] == t.shape
		p = tf.reshape(
			tf.gather_nd(tf.reshape(x, [-1, x.shape[-1]]), tf.reshape(t, [-1, 1]), batch_dims=1),
			t.shape
		)
		return tf.reduce_logsumexp(x, axis=-1) - p

def _mine_hard_negative(loss, pos, k):
    pos = tf.cast(pos, tf.bool)
    rank = tf.argsort(tf.argsort(loss * tf.cast(~pos, loss.dtype), axis=1), axis=1)
    hard_neg = rank < (tf.reduce_sum(tf.cast(pos, tf.int32), axis=1, keepdims=True) * k)
    return tf.cast(hard_neg, tf.bool)

def multibox_loss(x_loc, x_conf, t_loc, t_conf, k):
    pos = t_conf > 0
    if tf.reduce_all(tf.logical_not(pos)):
        return 0.0, 0.0

    x_loc = tf.reshape(x_loc, [-1, 4])
    t_loc = tf.reshape(t_loc, [-1, 4])
    loss_loc = tf.keras.losses.Huber()(t_loc, x_loc)
    loss_loc = tf.reduce_sum(loss_loc * tf.cast(tf.reshape(pos, [-1]), loss_loc.dtype)) / tf.reduce_sum(tf.cast(pos, loss_loc.dtype))

    loss_conf = _elementwise_softmax_cross_entropy(x_conf, t_conf)
    hard_neg = _mine_hard_negative(loss_conf, pos, k)
    mask = tf.logical_or(pos, hard_neg)
    loss_conf = tf.reduce_sum(loss_conf * tf.cast(mask, loss_conf.dtype)) / tf.reduce_sum(tf.cast(pos, loss_conf.dtype))

    return loss_loc, loss_conf
