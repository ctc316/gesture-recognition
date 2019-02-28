import numpy as np

class HMM:
    def __init__(self, N, M, pi=None, A=None, B=None, max_iters=1000, diff=0.001):
        self.N = N
        self.M = M
        self.pi = pi
        self.A = A
        self.B = B
        self.max_iters = max_iters
        self.diff = diff
        
    def __forward_backward(self, features, A, B, pi):
    
        T = features.shape[0]
        N = A.shape[0]

        coeff = np.zeros((T, 1))

        alpha = np.zeros((T, N))
        alpha[0] = pi[:, 0] * B[features[0]]
        coeff[0] = 1. / np.maximum(1E-10, np.sum(alpha[0]))
        alpha[0] *= coeff[0]

        # Forward
        for t in range(0, T-1):
            alpha[t+1] = np.sum(alpha[t][:, np.newaxis] * A, axis=0) * B[features[t+1]]

            coeff[t+1] = 1. / np.maximum(1E-10, np.sum(alpha[t+1]))
            alpha[t+1] *= coeff[t+1]
            alpha[t+1] = np.clip(alpha[t+1], a_min=1E-100, a_max=1)

        beta = np.zeros((T, N))
        beta[-1] = 1
        beta[-1] *= coeff[-1]

        # Backward
        for t in range(T-2, -1, -1):
            beta[t] = np.sum(A * B[features[t+1]][np.newaxis, :] * beta[t+1][np.newaxis, :], axis=1)
            beta[t] *= coeff[t]

        P_0 = -np.sum(np.log(coeff))
#         P_0 = -np.log(coeff)[T-1]
#         print(P_0)

        return alpha, beta, P_0, coeff

    def fit(self, gesture_data):
        N = self.N
        M = self.M
        
        # initial state ditribution
        self.pi = np.zeros((N, 1))
        self.pi[0] = 1

        # transition matrix
        A = np.random.uniform(low=0.05, high=1, size=(N, N))
        A = np.triu(np.sort(A)[:, ::-1])
        A /= np.sum(A, axis=1)[:, np.newaxis]

        # emission matrix
        B = np.random.uniform(low=0.1, high=1, size=(M, N))
        B /= np.sum(B, axis=0)[np.newaxis, :]

        c = 0
        while c < self.max_iters:
            c += 1

            gamma_sum = np.zeros((1, N))
            xi_sum = np.zeros((N, N))
            feat_count = np.zeros((M, N))
            likelihood = np.zeros(len(gesture_data))

            for idx, features in enumerate(gesture_data):
                features = features.flatten()
                
                # E-step
                alpha, beta, P_0, coeff = self.__forward_backward(features, A, B, self.pi)
                likelihood[idx] = P_0

                T, N = alpha.shape

                gamma = np.zeros((T, N))
                for t in range(T):
                    gamma[t] = alpha[t] * beta[t] * (1. / coeff[t])
                gamma_sum += np.sum(gamma, axis=0)

                xi = np.zeros((N, N, T-1))
                for t in range(T-1):
                    xi[:, :, t] = alpha[t][:, np.newaxis] * A * B[int(features[t + 1])][np.newaxis, :] * beta[t + 1][np.newaxis, :]
                xi_sum += np.sum(xi, axis=2)

                for l in range(M):
                    feat_i = (features == l)
                    feat_count[l] += np.sum(gamma[feat_i], axis=0)

            newmeanll = np.mean(likelihood)
            print('Iteration: {0},  Mean Likelihood: {1}'.format(c,newmeanll))

            # M-step
            newA = xi_sum / gamma_sum
            newA /= np.sum(newA, axis=1)[:, np.newaxis]
            newB = feat_count / gamma_sum
            newB /= np.sum(newB, axis=0)[np.newaxis, :]

            if c > 1 and abs(newmeanll - meanll) < self.diff: break

            A = newA
            B = newB

            meanll = np.mean(likelihood)
        
        self.A = A
        self.B = B
        

    def score(self, data):
        _, _, P_0, _ = self.__forward_backward(data, self.A, self.B, self.pi)
        return P_0