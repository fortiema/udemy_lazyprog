import numpy as np


def conv1D(w, x):
    n = len(x)
    m = len(w)
    y = np.zeros(n + m - 1)

    for i in range(n + m -1):
        y[i] = np.sum([w[j] * x[i-j] for j in range(m) if (i-j)])

    return y


def conv2D(w, x, keep_size=False):
    n1, n2 = x.shape
    m1, m2 = w.shape
    y = np.zeros((n1 + m1 - 1, n2 + m2 - 1))

    for i in range(n1):
        for j in range(n2):
            y[i:i+m1,j:j+m2] = w * x[i,j]  # Faster way to compute conv
            # y[i,j] = np.sum([w[ii,jj] * x[i-ii, j-jj] for ii in range(m1) for jj in range(m2) 
                            # if all([i-ii,j-jj,i-ii<n1,j-jj<n2])])

    if keep_size:
        y = y[m1/2:-m1/2+1,m2/2:-m2/2+1]

    return y
