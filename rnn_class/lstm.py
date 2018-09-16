import numpy as np
import theano
import theano.tensor as T

from util import init_weight


class LSTM:
    """Long Short-Term Memory"""

    def __init__(self, mi, mo):
        self.mi = mi
        self.mo = mo

        # Input
        Wxi = init_weight(mi, mo)
        Whi = init_weight(mo, mo)
        Wci = init_weight(mo, mo)
        bi = np.zeros(mo)
        self.Wxi = theano.shared(Wxi)
        self.Whi = theano.shared(Whi)
        self.Wci = theano.shared(Wci)
        self.bi = theano.shared(bi)

        # Output
        Wxo = init_weight(mi, mo)
        Who = init_weight(mo, mo)
        Wco = init_weight(mo, mo)
        bo = np.zeros(mo)
        self.Wxo = theano.shared(Wxo)
        self.Who = theano.shared(Who)
        self.Wco = theano.shared(Wco)
        self.bo = theano.shared(bo)

        # Forget
        Wxf = init_weight(mi, mo)
        Whf = init_weight(mo, mo)
        Wcf = init_weight(mo, mo)
        bf = np.zeros(mo)
        self.Wxf = theano.shared(Wxf)
        self.Whf = theano.shared(Whf)
        self.Wcf = theano.shared(Wcf)
        self.bf = theano.shared(bf)

        # Memory Cell
        Wxc = init_weight(mi, mo)
        Whc = init_weight(mo, mo)
        bc = np.zeros(mo)
        self.Wxc = theano.shared(Wxc)
        self.Whc = theano.shared(Whc)
        self.bc = theano.shared(bc)

        h0 = np.zeros(mo)
        c0 = np.zeros(mo)
        self.h0 = theano.shared(h0)
        self.c0 = theano.shared(c0)

        self.params = [
            self.Wxi, self.Whi, self.Wci, self.bi,
            self.Wxo, self.Who, self.Wco, self.bo,
            self.Wxf, self.Whf, self.Wcf, self.bf,
            self.Wxc, self.Whc, self.bc,
            self.h0, self.c0
        ]

    def recurrence(self, x_t, h_t1, c_t1):
        i_t = T.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
        f_t = T.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
        c_t = f_t * c_t1 + i_t * T.tanh(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc)
        o_t = T.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
        h_t = o_t * T.tanh(c_t)
        return h_t, c_t

    def output(self, x):
        [h, c], _ = theano.scan(
            fn=self.recurrence,
            sequences=x,
            outputs_info=[self.h0, self.c0],
            n_steps=x.shape[0]
        )
        return h, c
