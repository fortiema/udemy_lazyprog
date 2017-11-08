import sys

import numpy as np
import matplotlib.pyplot as plt

from util import get_data


label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main(fname):
    X, Y = get_data(fname, limit=10, balance_ones=False)

    print(X, Y)

    while True:
        for i in range(7):
            x, y = X[Y==i], Y[Y==i]
            print(x, y)
            N = len(y)
            if N:
                j = np.random.choice(N)
                plt.imshow(x[j].reshape(48, 48), cmap='gray')
                plt.title(label_map[y[j]])
                plt.show()
        prompt = input('Continue? (y|N):\n')
        if not prompt.lower().startswith('y'):
            break


if __name__ == '__main__':
    main(sys.argv[1])
