import numpy as np

class UCB_SI():
    def __init__(self, mab, side_info, budget):
        # side_info: K0 x 2
        self.mab = mab
        self.side_info = side_info
        self.budget = budget
        self.K0 = self.mab.nArms
        self.K = None
        self.map_K_K0 = []

        self.U = []
        self.N = []
        self.c = []
        self.mu = []

        assert self.side_info.shape[0] == self.K0

        self.prune(self.side_info)
        self.initialize()

    def prune(self):
        l_max = self.side_info[:,0].max()
        for i in range(self.K0):
            if self.side_info[i,1] >= l_max:
                self.map_K_K0.append(i)
        self.K = len(self.map_K_K0)
        self.map_K_K0 = np.array(self.map_K_K0).astype('int')

    def initialize(self):
        for i in range(self.K):
            l_k = self.side_info[self.map_K_K0[i], 0]
            u_k = self.side_info[self.map_K_K0[i], 1]
            self.U.append(u_k)
            self.N.append(0)
            if l_k >= 0.5:
                self.c.append(2*l_k*(1-l_k))
            else if u_k <= 0.25:
                self.c.append(3*u_k*(1-u_k))
            else:
                self.c.append(0.5)
            self.mu.append(0)

            self.U = np.array(self.U)
            self.N = np.array(self.N)
            self.c = np.array(self.c)
            self.mu = np.array(self.mu)

    def run(self):
        self.A = []
        self.Y = []
        for t in range(self.budget):
            A_t = self.U.argmax()
            Y_t = self.mab.play(self.map_K_K0[A_t])
            self.A.append(A_t)
            self.Y.append(Y_t)
            self.mu[A_t] = (self.N[A_t] * self.mu[A_t] + Y_t) / (1 + self.N[A_t])
            self.N[A_t] += 1
            for k in range(self.K):
                u_k = self.side_info[self.map_K_K0[k], 1]
                self.U[k] = min(u_k, self.mu[k] + np.sqrt(3*self.c[k]*np.log(t+1)/self.N[k]))

        return self.U, self.A, self.Y
