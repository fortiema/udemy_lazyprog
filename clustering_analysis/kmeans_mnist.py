from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kmeans import plot_k_means


def get_data(limit=None):
    print('Ingesting data...')
    df = pd.read_csv('mnist_train.csv')
    data = df.as_matrix()
    np.random.shuffle(data)

    X = data[:, 1:] / 255.0
    Y = data[:, 0]

    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


def purity(Y, R):
    N, K = R.shape
    p = 0
    for k in range(K):
        best_target = -1
        max_intersection = 0
        for j in range(K):
            intersection = R[Y==j, k].sum()
            if intersection > max_intersection:
                max_intersection = intersection
                best_target = j
        p += max_intersection
    return p / N


def DBI(X, M, R):
    K, D = M.shape

    sigma = np.zeros(K)
    for k in range(K):
        diffs = X - M[k]
        sq_dist = (diffs * diffs).sum(axis=1)
        w_sq_dist = R[:,k] * sq_dist
        sigma[k] = np.sqrt(w_sq_dist).mean()

    dbi = 0
    for k in range(K):
        max_ratio = 0
        for j in range(K):
            if k != j:
                num = sigma[k] + sigma[j]
                denom = np.linalg.norm(M[k] - M[j])
                ratio = num / denom
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio
    return dbi / K


def main():
    X, Y = get_data(1000)
    print('Number of data points: {}'.format(len(Y)))
    M, R = plot_k_means(X, len(set(Y)))
    print('Purity: {}'.format(purity(Y, R)))
    print('DBI: {}'.format(DBI(X, M, R)))


if __name__ == '__main__':
    main()
