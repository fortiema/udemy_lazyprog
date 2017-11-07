import numpy as np
from process import get_data

X, Y = get_data()

M = 5
D = X.shape[1]
K = len(set(Y))

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

def softmax(a):
    exp_a = np.exp(a)
    return exp_a / exp_a.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)

PYX = forward(X, W1, b1, W2, b2)
P = np.argmax(PYX, axis=1)

def classif_rate(Y, P):
    return np.mean(Y == P)


print('Score: {}'.format(classif_rate(Y, P)))