import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

from generate_continuous import get_signals, big_init, simple_init


def random_normalized(d1, d2):
    x = np.random.random((d1, d2))
    return x / x.sum(axis=1, keepdims=True)


class HMM:
    """Gaussian Mixture Hidden Markov Model for continuous domains

    Train using Baum-Welch algorithm

    """
    def __init__(self, M, K):
        self.M = M  # States
        self.K = K  # Gaussians

    def fit(self, X, epochs=30, epsilon=10e-1):
        N = len(X)
        D = X[0].shape[1]

        self.pi = np.ones(self.M) / self.M
        self.A = random_normalized(self.M, self.M)
        self.R = np.ones((self.M, self.K)) / self.K
        self.mu = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
                idx = np.random.choice(N)
                x = X[idx]
                time = np.random.choice(len(x))
                self.mu[i,k] = x[time]

        self.sigma = np.zeros((self.M, self.K, D, D))
        for j in range(self.M):
            for k in range(self.K):
                self.sigma[j, k] = np.eye(D)

        costs = []
        for e in range(epochs):
            if e % 1 == 0:
                print('Iteration {}'.format(e))
            alphas = []
            betas = []
            gammas = []
            Bs = []
            P = np.zeros(N)

            for n in range(N):
                x = X[n]
                T = len(x)

                B = np.zeros((self.M, T))
                component = np.zeros((self.M, self.K, T))
                for j in range(self.M):
                    for t in range(T):
                        for k in range(self.K):
                            p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                            component[j,k,t] = p
                            B[j,t] += p
                Bs.append(B)

                alpha = np.zeros((T, self.M))
                alpha[0] = self.pi * B[:,0]
                for t in range(1, t):
                    alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
                P[n] = alpha[-1].sum()
                assert(P[n] <= 1)
                alphas.append(alpha)

                beta = np.zeros((T, self.M))
                beta[-1] = 1
                for t in range(T-2, -1, -1):
                    beta[t] = self.A.dot(B[:,t+1] * beta[t+1])
                betas.append(beta)

                gamma = np.zeros((T, self.M, self.K))
                for t in range(T):
                    alphabeta = (alphas[n][t,:] * betas[n][t,:]).sum()
                    for j in range(self.M):
                        factor = alphas[n][t,j] * betas[n][t,j] / alphabeta
                        for k in range(self.K):
                            gamma[t,j,k] = factor * component[j,k,t] / B[j,k]
                gammas.append(gamma)

            cost = np.log(P).sum()
            costs.append(cost)

            # M-Step
            self.pi = np.sum((alphas[n][0] * betas[n][0]) / P[n] for n in range(N)) / N
            a_den = np.zeros((self.M, 1))
            a_num = 0
            r_den = np.zeros(self.M)
            r_num = np.zeros((self.M, self.K))
            mu_num = np.zeros((self.M, self.K, D))
            sigma_num = np.zeros((self.M, self.K, D, D))
            for n in range(N):
                x = X[n]
                T = len(x)
                B = Bs[n]
                gamma = gammas[n]

                a_den += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]
                a_num_n = np.zeros((self.M, self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num_n[i,j] += alphas[n][t,i] * self.A[i,j] * B[j,t+1] * betas[n][t+1,j]
                a_num += a_num_n / P[n]

                r_num_n = np.zeros((self.M, self.K))
                r_den_n = np.zeros(self.M)
                for j in range(self.M):
                    for k in range(self.K):
                        for t in range(T):
                            r_num_n[j,k] += gamma[t,j,k]
                            r_den_n[j] += gamma[t,j,k]
                r_num += r_num_n / P[n]
                r_den += r_den_n / P[n]

                mu_num_n = np.zeros((self.M, self.K, D))
                sigma_num_n = np.zeros((self.M, self.K, D, D))
                for j in range(self.M):
                    for k in range(self.K):
                        for t in range(T):
                            mu_num_n[j,k] += gamma[t,j,k] * x[t]
                            sigma_num_n[j,k] += gamma[t,j,k] * np.outer(x[t] - self.mu[j,k], x[t] - self.mu[j,k])
                mu_num += mu_num_n / P[n]
                sigma_num += sigma_num_n / P[n]

            self.A = a_num / a_den

            for j in range(self.M):
                for k in range(self.K):
                    self.R[j,k] = r_num[j,k] / r_den[j]
                    self.mu[j,k] = mu_num[j,k] / r_num[j,k]
                    self.sigma[j,k] = sigma_num[j,k] / r_num[j,k]

        print('A: {}'.format(self.A))
        print('mu: {}'.format(self.mu))
        print('sigma: {}'.format(self.sigma))
        print('R: {}'.format(self.R))
        print('pi: {}'.format(self.pi))

        plt.plot(costs)
        plt.show()

    def likelihood(self, x):
        T = len(x)
        alpha.zeros((T, self.M))

        B = np.zeros((self.M, T))
        for j in range(self.M):
            for t in range(T):
                for k in range(self.K):
                    p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                    B[j,t] += p

        alpha[0] = self.pi * B[:,0]
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
        return alpha[-1].sum()

    def likelihood_all(self, X):
        return np.array([self.likelihood(x) for x in X])

    def log_likelihood_all(self, X):
        return np.log(self.likelihood_all(X))

    def set(self, pi, A, R, mu, sigma):
        self.pi = pi
        self.A = A
        self.R = R
        self.mu = mu
        self.sigma = sigma

def fake_signal(init=simple_init):
    signals = get_signals(N=10, T=10, init=init)
    hmm = HMM(2, 2)
    hmm.fit(signals)
    L = hmm.log_likelihood_all(signals).sum()
    print('Fitted LL: {}'.format(L))

    _,_,_, pi, A, R, mu, sigma = init()
    hmm.set(pi, A, R, mu, sigma)
    L = hmm.log_likelihood_all(signals).sum()
    print('Fitted LL: {}'.format(L))


if __name__ == '__main__':
    fake_signal()