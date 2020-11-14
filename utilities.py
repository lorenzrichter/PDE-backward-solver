import numpy as np


def get_X_process(problem, K, delta_t, seed=42):
    
    np.random.seed(seed)
    
    N = int(np.ceil(problem.T / delta_t))
    sq_delta_t = np.sqrt(delta_t)
    
    X = np.zeros([N + 1, K, problem.d])
    X[0, :, :] = np.repeat(problem.X_0[np.newaxis, :], K, axis=0)
    xi = np.random.randn(N + 1, K, problem.d)

    for n in range(N):
        X[n + 1, :, :] = (X[n, :, :] + problem.b(X[n, :, :]) * delta_t 
                          + problem.sigma(X[n, :, :]).dot(xi[n + 1, :, :].T).T * sq_delta_t)
        
    return X, xi