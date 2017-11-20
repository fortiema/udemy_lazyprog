import numpy as np
import matplotlib.pyplot as plt

def random_normalized(d1, d2):
    """Initialises a valid Markov Matrix (each row sums to 1)"""
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)

class HMM:
    """Hidden Markov Model

    A discrete Hidden Markov Model implementation.
    Training uses Baum-Welch algorithmn.

    """
    def __init__(self, M):
        self.M = M

    def fit(self, X, max_iter=30):
        np.random.seed(42)

        V = max(max(x) for x in X) + 1
        N = len(X)

        # Pi is the initial state dist.
        self.pi = np.ones(self.M) / self.M
        # A is the transition matrix (prob of going from any state X(t-1) to any state X(t))
        self.A = random_normalized(self.M, self.M)
        # B is the probability matrix of making observation v (in vocab V) given the current state
        self.B = random_normalized(self.M, V)

        # Notes:
        # ---
        # alpha -> probability of observing a certain sequence (from 1 to t) and be on state i at time t
        # beta -> probability of observinga certain sequence given initial state

        costs = []
        for epoch in range(max_iter):
            if epoch % 10 == 0:
                print('Iteration {}:'.format(epoch))
            alphas = []
            betas = []
            P = np.zeros(N)
            for n in range(N):
                x = X[n]
                T = len(x)
                alpha = np.zeros((T, self.M))
                alpha[0] = self.pi * self.B[:, x[0]]
                for t in range(1, T):
                    alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
                P[n] = alpha[-1].sum()
                alphas.append(alpha)

                beta = np.zeros((T, self.M))
                beta[-1] = 1
                for t in range(T-2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t+1]] * beta[t+1])
                betas.append(beta)

            assert(np.all(P > 0))
            cost = np.sum(np.log(P))
            costs.append(cost)

            # Update pi, A, B
            self.pi = np.sum((alphas[n][0] * betas[n][0])/P[n] for n in range(N)) / N

            d1 = np.zeros((self.M, 1))
            d2 = np.zeros((self.M, 1))
            a_num = 0
            b_num = 0
            for n in range(N):
                x = X[n]
                T = len(x)

                d1 += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]
                d2 += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / P[n]

                a_num_n = np.zeros((self.M, self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num_n[i,j] += alphas[n][t,i] * self.A[i,j] * self.B[j, x[t+1]] * betas[n][t+1,j]
                a_num += a_num_n / P[n]

                b_num_n = np.zeros((self.M, V))
                for i in range (self.M):
                    for t in range(T):
                        b_num_n[i,x[t]] += alphas[n][t,i] * betas[n][t,i]
                b_num += b_num_n / P[n]

            self.A = a_num / d1
            self.B = b_num / d2
        print('\tpi: {}'.format(self.pi))
        print('\tA: {}'.format(self.A))
        print('\tB: {}'.format(self.B))

        plt.plot(costs)
        plt.show()

    def likelihood(self, x):
        T = len(x)
        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi * self.B[:, x[0]]
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]
        return alpha[-1].sum()

    def likelihood_all(self, X):
        return np.array([self.likelihood(x) for x in X])

    def log_likelihood_all(self, X):
        return np.log(self.likelihood_all(X))

    def get_state_sequence(self, x):
        """Finds most likely state sequence - Viterbi algorithm"""
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = self.pi * self.B[:, x[0]]
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * self.B[j,x[t]]
                psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])

        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

def fit_coin():
    X = []
    for line in open('large_files/coin_data.txt'):
        x = [1 if e == 'H' else 0 for e in line.rstrip() if e]
        X.append(x)

    model = HMM(2)
    model.fit(X)

    L = model.log_likelihood_all(X).sum()
    print('Fit results: {}'.format(L))

    model.pi = np.array([0.5, 0.5])
    model.A = np.array([[0.1, 0.9], [0.8, 0.2]])
    model.B = np.array([[0.6, 0.4], [0.3, 0.7]])
    L = model.log_likelihood_all(X).sum()
    print('True results: {}'.format(L))

    print('Most likely state sequence for {}:'.format(X[0]))
    print(model.get_state_sequence(X[0]))


if __name__ == '__main__':
    fit_coin()