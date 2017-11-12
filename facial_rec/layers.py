import uuid

import numpy as np
import tensorflow as tf

from util import init_w_b


class Dense:
    """Fully-connected Layer object"""
    def __init__(self, M1, M2, f=None, layer_id=None):
        self.M1 = M1
        self.M2 = M2
        self.f = f or tf.nn.relu
        self.id = layer_id or str(uuid.uuid4())[:4]
        W, b = init_w_b(M1, M2)

        self.W = tf.Variable(W, name='W_{}'.format(self.id))
        self.b = tf.Variable(b, name='b_{}'.format(self.id))
        self.params = [self.W, self.b]

    def forward(self, X, train):
        return self.f(tf.matmul(X, self.W) + self.b)


class ConvMaxPool2D:
    """2-dimensional Convolution Layer with Max-Pooling"""
    def __init__(self, shape, pool_size, layer_id=None):
        self.id = layer_id or str(uuid.uuid4())[:4]
        W = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(pool_size)))
        b = np.zeros(shape[-1], dtype=np.float32)

        self.W = tf.Variable(W.astype(np.float32), name='W_{}'.format(self.id))
        self.b = tf.Variable(b.astype(np.float32), name='b_{}'.format(self.id))
        self.params = [self.W, self.b]

    def forward(self, X, train):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        return tf.nn.max_pool(
            conv_out,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )


class Flatten:
    """Flattens result of conv layer(s) to feed into dense layers"""
    def forward(self, X, train):
        x_shape = X.get_shape().as_list()
        return tf.reshape(X, shape=[-1, np.prod(x_shape[1:])])
