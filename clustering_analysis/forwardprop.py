import numpy as np
import matplotlib.pyplot as plt

Nclass = 500

X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, -2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

D = 2
M = 3
K = 3

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    exp_A = np.exp(A)
    Y = exp_A / exp_A.sum(axis=1, keepdims=True)
    return Y

def classif_rate(Y, P):
    nb_correct = 0
    nb_total = 0
    for i in range(len(Y)):
        nb_total += 1
        if Y[i] == P[i]:
            nb_correct += 1
    return float(nb_correct) / nb_total

PYX = forward(X, W1, b1, W2, b2)
P = np.argmax(PYX, axis=1)

assert(len(P) == len(Y))

print('Classification: {}'.format(classif_rate(Y, P)))