import uuid

import numpy as np
import tensorflow as tf

from util import init_w_b


class Dense:
    """Fully-connected Layer object"""
    def __init__(self, M1, M2, f=None, dropout=True, layer_id=None):
        self.M1 = M1
        self.M2 = M2
        self.f = f or tf.nn.relu
        self.id = layer_id or str(uuid.uuid4())[:4]
        W, b = init_w_b(M1, M2)

        self.W = tf.Variable(W, name='W_{}'.format(self.id))
        self.b = tf.Variable(b, name='b_{}'.format(self.id))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.W)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.b)

        self.dropout = dropout

    def forward(self, X, train):
        out =  self.f(tf.matmul(X, self.W) + self.b)
        return tf.nn.dropout(out, 0.5) if self.dropout else out


class ConvMaxPool2D:
    """2-dimensional Convolution Layer with Max-Pooling"""
    def __init__(self, shape, pool_size, layer_id=None):
        self.id = layer_id or str(uuid.uuid4())[:4]
        W = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(pool_size)))
        b = np.zeros(shape[-1], dtype=np.float32)

        self.W = tf.Variable(W.astype(np.float32), name='W_{}'.format(self.id))
        self.b = tf.Variable(b.astype(np.float32), name='b_{}'.format(self.id))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.W)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.b)

    def forward(self, X, train):
        out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        out = tf.nn.relu(tf.nn.bias_add(out, self.b))
        out = tf.nn.dropout(out, 0.5)
        return tf.nn.max_pool(
            out,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME'
        )
        


class Flatten:
    """Flattens result of conv layer(s) to feed into dense layers"""
    def forward(self, X, train):
        x_shape = X.get_shape().as_list()
        return tf.reshape(X, shape=[-1, np.prod(x_shape[1:])])
