import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getData, sigmoid, sigmoid_cost, error_rate, relu


class ANN(object):
    """Simple Single Hidden Layer Artificial Neural Network
    """
    def __init__(self, M):
        self.M = M
    
    def fit(self, X, Y, learning_rate=5*10e-7, reg=1.0, epochs=10000, show_figs=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        self.W1 = np.random.randn(D, self.M) / np.sqrt(D + self.M)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M) / np.sqrt(self.M)
        self.b2 = 0

        costs = []
        best_valid_error = 1
        for i in range(epochs):
            # Forward prop.
            pY, Z = self.forward(X)

            # Gradient desc.
            pY_Y = pY - Y
            self.W2 -= learning_rate * (Z.T.dot(pY_Y) + reg*self.W2)
            self.b2 -= learning_rate * ((pY_Y).sum() + reg*self.b2)

            # dZ = np.outer(pY_Y, self.W2) * (Z > 0)
            dZ = np.outer(pY_Y, self.W2) * (1 - Z * Z)
            self.W1 -= learning_rate * (X.T.dot(dZ) + reg*self.W1)
            self.b1 -= learning_rate * (np.sum(dZ, axis=0) + reg*self.b1)

            if i % 50 == 0:
                pYvalid, _ = self.forward(Xvalid)
                c = sigmoid_cost(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.round(pYvalid))
                print("{0: <6}> Cost: {1:}, Error: {2:}".format(i, c, e))
                if e < best_valid_error:
                    best_valid_error = e
        print("Best Error: {}".format(best_valid_error))

        if show_figs:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        # Z = relu(X.dot(self.W1) + self.b1)
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return sigmoid(Z.dot(self.W2) + self.b2), Z

    def predict(self, X):
        pY, _ = self.forward(X)
        return np.round(pY)

    def score(self, X, Y):
        pred = self.predict(X)
        return 1 - error_rate(Y, pred)


def main():
    X, Y = getData("/media/iceman/SSD/datasets/image/fer2013/fer2013.csv", binary=True)

    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    # debug
    # print(X)
    # print(Y)

    model = ANN(100)
    model.fit(X, Y, reg=0.0, show_figs=True)


if __name__ == '__main__':
    main()