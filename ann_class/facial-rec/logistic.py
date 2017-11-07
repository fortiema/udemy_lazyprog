import numpy as np
import matplotlib.pyplot as plt

from util import getData, softmax, cost, y2indicator, error_rate
from sklearn.utils import shuffle


class LogisticModel(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, learning_rate=10e-8, reg=10e-12, epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        Tvalid = y2indicator(Yvalid)
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)
        self.W = np.random.randn(D, K) / np.sqrt(D + K)
        self.b = np.zeros(K)

        costs = []
        best_valid_error = 1
        for i in range(epochs):
            pY = self.forward(X)

            self.W -= learning_rate * (X.T.dot(pY - T) + reg*self.W)
            self.b -= learning_rate * ((pY - T).sum(axis=0) + reg*self.b)

            if i % 100 == 0:
                pYvalid = self.forward(Xvalid)
                c = cost(Tvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                print("{0: <5}> Cost: {1: .3f} Error: {2: .3f}".format(i, c, e))
                if e < best_valid_error:
                    best_valid_error = e
                
        print("Best Validation Error: {}".format(best_valid_error))

        if show_fig:
            plt.plot(costs)
            plt.show()
        

    def forward(self, X):
        return softmax(X.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return np.argmax(pY, axis=1)

    def score(self, X, Y):
        pred = self.predict(X)
        return 1 - error_rate(Y, pred)


def main():
    X, Y = getData("/media/iceman/SSD/datasets/image/fer2013/fer2013.csv")

    model = LogisticModel()
    model.fit(X, Y, show_fig=True)

    print(model.score(X, Y))


if __name__ == '__main__':
    main()