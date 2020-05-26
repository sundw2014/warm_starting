import numpy as np

def extract_bounds(delta_sub, delta_sup, log, K):
    # log: [A, Y]
    A, Y = log
    A = A.astype('int')
    l = []
    u = []
    mu = Y.mean()
    for k in range(K):
        p_k = (A==k).sum()/len(A)
        nu_k = Y[np.logical_not(A==k)].mean()
        l.append(mu - min(delta_sup, nu_k)*(1-p_k))
        u.append(mu - delta_sub*(1-p_k))
    return np.array([l, u]).T
