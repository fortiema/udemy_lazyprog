import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage


def main():
    D = 2
    s = 4
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 900
    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3

    Z = linkage(X, 'ward')
    print("Z: {}".format(Z.shape))
    plt.title('Ward')
    dendrogram(Z)
    plt.show()

    Z = linkage(X, 'single')
    plt.title('Single')
    dendrogram(Z)
    plt.show()

    Z = linkage(X, 'complete')
    plt.title('Complete')
    dendrogram(Z)
    plt.show()


if __name__ == '__main__':
    main()
