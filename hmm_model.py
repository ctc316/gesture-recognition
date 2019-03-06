import numpy as np
from scipy.special import logsumexp


class HMM:
    def __init__(self, N, M, PI=None, A=None, B=None, max_iters=1000, tol=0.001):
        self.N = N
        self.M = M

        self.PI = PI
        if PI is None:
            self.PI = np.zeros((N))
            self.PI[0] = 1

        self.A = A
        self.B = B
        self.max_iters = max_iters
        self.tol = tol


    def fit(self, data_list):
        N = self.N
        M = self.M

        A = np.random.uniform(low=0.05, high=1, size=(N, N))
        A /= np.sum(A, axis=1)[:, np.newaxis]

        B = np.random.uniform(low=0.05, high=1, size=(M, N))
        B /= np.sum(B, axis=0)[np.newaxis, :]

        iters = 0
        while iters < self.max_iters:
            iters += 1

            sum_gamma = np.zeros((1, N))
            sum_xi = np.zeros((N, N))
            accumu_gamma = np.zeros((M, N))
            likelihood = np.zeros(len(data_list))

            # E-step
            for i, data in enumerate(data_list):
                data = data.flatten()

                # forward, backward
                alpha, beta, prob = self.__forward_backward(data, A, B, self.PI)
                likelihood[i] = prob
                T = alpha.shape[0]

                # optimal state
                gamma = np.zeros((T, N))
                for t in range(T):
                    gamma[t] = alpha[t] * beta[t]
                sum_gamma += np.sum(gamma, axis=0)

                # pair of states
                xi = np.zeros((N, N, T - 1))
                for t in range(T-1):
                    xi[:, :, t] = alpha[t][:, np.newaxis] * A * B[int(data[t + 1])][np.newaxis, :] * beta[t + 1][np.newaxis, :]
                sum_xi += np.sum(xi, axis=2)

                # update B
                for obs_row in range(M):
                    selected = (data == obs_row)
                    accumu_gamma[obs_row] += np.sum(gamma[selected], axis=0)

            sum_likelihood = np.sum(likelihood)
            print('Iterate {0}: log-likelihood = {1}'.format(iters, sum_likelihood))
            if iters > 1 and abs(sum_likelihood - prev_sum_likelihood) < self.tol:
                break
            prev_sum_likelihood = sum_likelihood

            # M-step: average A' and B' over time
            new_A = sum_xi / T
            new_A /= np.sum(new_A, axis=1)[:, np.newaxis]

            new_B = accumu_gamma / T
            new_B /= np.sum(new_B, axis=0)[np.newaxis, :]

            # Update A, B
            A = new_A
            B = new_B

        self.A = A
        self.B = B


    def score(self, data):
        _, _, prob = self.__forward_backward(data, self.A, self.B, self.PI)
        return prob


    def __forward_backward(self, data, A, B, PI):
        T = data.shape[0]
        N = A.shape[0]

        # Forward
        alpha = np.zeros((T, N))
        alpha[0] = PI + B[data[0]]

        for t in range(0, T - 1):
            alpha[t + 1] = logsumexp(alpha[t][:, np.newaxis] + A, axis=0) + B[data[t + 1]]

        # Backward
        beta = np.zeros((T, N))
        beta[-1] = 0

        for t in range(T - 2, -1, -1):
            beta[t] = logsumexp(A + B[data[t + 1]][np.newaxis, :] + beta[t + 1][np.newaxis, :], axis=1)

        # log-likelihood
        prob = logsumexp(alpha[T - 1])

        return alpha, beta, prob