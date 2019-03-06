import numpy as np

class HMM:
    def __init__(self, N, M, PI=None, A=None, B=None):
        self.N = N
        self.M = M
        
        self.PI = PI if PI is not None else None
        self.A = A if A is not None else None
        self.B = B if B is not None else None
    
        self.max_iters = 1000
        self.diff = 0.001
        
    def __forward_backward(self, sequence, A, B, pi):
    
        T = sequence.shape[0]
        N = A.shape[0]

        coeff = np.zeros((T, 1))

        alpha = np.zeros((T, N))
        alpha[0] = pi * B[sequence[0]]
        coeff[0] = 1. / np.maximum(1E-10, np.sum(alpha[0]))
        alpha[0] *= coeff[0]

        # Forward
        for t in range(0, T-1):
            alpha[t+1] = np.sum(alpha[t][:, np.newaxis] * A, axis=0) * B[sequence[t+1]]

            coeff[t+1] = 1. / np.maximum(1E-10, np.sum(alpha[t+1]))
            alpha[t+1] *= coeff[t+1]
            alpha[t+1] = np.clip(alpha[t+1], a_min=1E-100, a_max=1)

        beta = np.zeros((T, N))
        beta[-1] = 1
        beta[-1] *= coeff[-1]

        # Backward
        for t in range(T-2, -1, -1):
            beta[t] = np.sum(A * B[sequence[t+1]][np.newaxis, :] * beta[t+1][np.newaxis, :], axis=1)
            beta[t] *= coeff[t]

        P_0 = -np.sum(np.log(coeff))
    
        return alpha, beta, P_0, coeff

    def fit(self, gesture_data):
        N = self.N
        M = self.M
        
        # initial state ditribution
        self.PI = np.zeros((N))
        self.PI[0] = 1

        # transition matrix
        A = np.random.uniform(low=0.1, high=1, size=(N, N))
        A /= np.sum(A, axis=1)[:, np.newaxis]

        # emission matrix
        B = np.random.uniform(low=0.1, high=1, size=(M, N))
        B /= np.sum(B, axis=0)[np.newaxis, :]

        count = 0
        while count < self.max_iters:
            count += 1

            gamma_sum = np.zeros((1, N))
            xi_sum = np.zeros((N, N))
            gamma_accumulation = np.zeros((M, N))
            
            likelihood = np.zeros(len(gesture_data))

            for i, sequence in enumerate(gesture_data):
                sequence = sequence.flatten()
                
                # E-step
                alpha, beta, P, coeff = self.__forward_backward(sequence, A, B, self.PI)
                likelihood[i] = P

                T, N = alpha.shape

                # Optimal State: gamma at each time
                gamma = np.zeros((T, N))
                for t in range(T):
                    gamma[t] = alpha[t] * beta[t] * (1. / coeff[t])
                gamma_sum += np.sum(gamma, axis=0)

                # Pair of States: xi
                xi = np.zeros((N, N, T-1))
                for t in range(T-1):
                    xi[:, :, t] = alpha[t][:, np.newaxis] * A * B[int(sequence[t + 1])][np.newaxis, :] * beta[t + 1][np.newaxis, :]
                xi_sum += np.sum(xi, axis=2)

                # accumulating gamma to update B
                for obs_row in range(M):
                    selected_sequence = (sequence == obs_row)
                    gamma_accumulation[obs_row] += np.sum(gamma[selected_sequence], axis=0)

            sum_likelihood = np.sum(likelihood)
            print('Iteration: {0},  Mean Likelihood: {1}'.format(count, sum_likelihood))

            if count > 1 and abs(sum_likelihood - prev_sum_likelihood) < self.diff: 
                break
                
            prev_sum_likelihood = np.sum(likelihood)
            
            # M-step: average A' and B' over time
            A_prime = xi_sum / T
            A_prime /= np.sum(A_prime, axis=1)[:, np.newaxis]
        
            B_prime = gamma_accumulation / T
            B_prime /= np.sum(B_prime, axis=0)[np.newaxis, :]

            # Update A and B
            A = A_prime
            B = B_prime

        self.A = A
        self.B = B
        
    def score(self, sequence):
        _, _, P, _ = self.__forward_backward(sequence, self.A, self.B, self.PI)
        return P