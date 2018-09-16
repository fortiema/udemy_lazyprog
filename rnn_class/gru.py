import numpy as np
import theano
import theano.tensor as T

from util import init_weight


class GRU:
    """ Gated Recurrent Unit
    """

    def __init__(self, mi, mo, activation):
        """GRU Layer

        Args:
            mi (int): Input dimension
            mo (int): Output dimension
            activation (theano.tensor.nnet.*): Activation function used to compute h_hat

        """
        self.mi = mi
        self.mo = mo
        self.f = activation

        Wxr = init_weight(mi, mo)
        Whr = init_weight(mo, mo)
        br = np.zeros(mo)
        Wxz = init_weight(mi, mo)
        Whz = init_weight(mo, mo)
        bz = np.zeros(mo)
        Wxh = init_weight(mi, mo)
        Whh = init_weight(mo, mo)
        bh = np.zeros(mo)
        h0 = np.zeros(mo)

        self.Wxr = theano.shared(Wxr)
        self.Whr = theano.shared(Whr)
        self.br = theano.shared(br)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bz = theano.shared(bz)
        self.Wxh = theano.shared(Wxh)
        self.Whh = theano.shared(Whh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)

        self.params = [self.Wxr, self.Whr, self.br, self.Wxz, self.Whz, self.bz, self.Wxh, self.Whh, self.bh, self.h0]

    def recurrence(self, x_t, h_t1):
        r = T.nnet.sigmoid(x_t.dot(self.Wxr) + h_t1.dot(self.Whr) + self.br)
        z = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
        h_hat = self.f(x_t.dot(self.Wxh) + (r * h_t1).dot(self.Whh) + self.bh)
        h = (1 - z) * h_t1 + (z * h_hat)
        return h

    def output(self, x):
        h, _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.h0],
            n_steps=x.shape[0]
        )
        return h
