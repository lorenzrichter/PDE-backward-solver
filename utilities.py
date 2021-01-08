import numpy as np
import torch as pt


def get_X_process(problem, K, delta_t, seed=42, sigma='constant'):
    '''
    :param problem: problem object that specifies the PDE problem
    :param K: batch size, i.e. number of samples
    :param delta_t: step size
    :param seed: seed for the random number generator
    :sigma: specifies whether the diffusion coefficient is constant or can depend on x.
            In the former case sigma is d x d, in the latter case it is K x d x d.
    '''
    
    np.random.seed(seed)
    
    N = int(np.ceil(problem.T / delta_t))
    sq_delta_t = np.sqrt(delta_t)
    
    X = np.zeros([N + 1, K, problem.d])
    X[0, :, :] = np.repeat(problem.X_0[np.newaxis, :], K, axis=0)
    xi = np.random.randn(N + 1, K, problem.d)

    if sigma == 'constant':
        for n in range(N):
            X[n + 1, :, :] = (X[n, :, :] + problem.b(X[n, :, :]) * delta_t
                              + problem.sigma(X[n, :, :]).dot(xi[n + 1, :, :].T).T * sq_delta_t)

    else:
        problem.modus == 'pytorch'
        X = pt.tensor(X)
        xi = pt.tensor(xi)
        for n in range(N):
            X[n + 1, :, :] = (X[n, :, :] + problem.b(X[n, :, :]) * delta_t
                              + pt.bmm(pt.tensor(problem.sigma(X[n, :, :])), pt.tensor(xi[n + 1, :, :]).unsqueeze(2)).squeeze(2) * sq_delta_t)

    return X, xi