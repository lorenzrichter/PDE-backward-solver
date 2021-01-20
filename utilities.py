#pylint: disable=invalid-name

import numpy as np
import torch as pt

from pytorch_future import hessian


def get_X_process(problem, K, delta_t, seed=42):
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

    if problem.sigma_modus == 'constant':
        for n in range(N):
            X[n + 1, :, :] = (X[n, :, :] + problem.b(X[n, :, :]) * delta_t
                              + problem.sigma(X[n, :, :]).dot(xi[n + 1, :, :].T).T * sq_delta_t)
    else:
#         problem.modus == 'pytorch'
#         for n in range(N):
#            X[n + 1, :, :] = (X[n, :, :] + problem.b(X[n, :, :]) * delta_t
#                               + pt.bmm(pt.tensor(problem.sigma(X[n, :, :])), pt.tensor(xi[n + 1, :, :]).unsqueeze(2)).squeeze(2) * sq_delta_t)
        for n in range(N):
            X[n + 1, :, :] = (X[n, :, :] + problem.b(X[n, :, :]) * delta_t
                              + np.einsum('ijl,il->ij', problem.sigma(X[n, :, :]), xi[n + 1, :, :]) * sq_delta_t)
            # print('noise', np.einsum('ijl,il->ij', problem.sigma(X[n, :, :]), xi[n + 1, :, :]) * sq_delta_t)
            # print('noise parts:', problem.sigma(X[n, :, :]), 'xi',  xi[n + 1, :, :], 'sq_delta_t', sq_delta_t)

    return X, xi


def compute_PDE_loss(problem, delta_t=0.01, K=1000, Y_n=None, vfun=None, testOde=None, seed=42):

    N = int(np.ceil(problem.T / delta_t))

    problem_modus_temp = problem.modus
    problem.modus = 'numpy'

    X, xi = get_X_process(problem, K, delta_t, seed)

    problem.modus = problem_modus_temp

    if problem.modus == 'pt':
        X = pt.autograd.Variable(pt.tensor(X).float(), requires_grad=True)
    else:
        X = X.transpose((2, 1, 0))

    avg_loss = []

    for n in range(N):

        t_n = n * delta_t

        if problem.modus == 'pt':
            X_n = X[n, :, :]
        else:
            X_n = X[:, :, n]

        if problem.modus == 'pt':

            v_of_x = Y_n[n](X_n) # K x 1

            Y_eval = v_of_x.squeeze().sum()
            Y_eval.backward(retain_graph=True)
            v_x, = pt.autograd.grad(Y_eval, X_n, create_graph=True) # K x d
            v_t = (Y_n[n + 1](X_n) - v_of_x) / delta_t

            v_xx = pt.zeros(K, problem.d, problem.d)
            for i, x in enumerate(X_n):
                v_xx[i, :, :] = hessian(Y_n[n], x.unsqueeze(0), create_graph=True).squeeze()

            v_of_x = v_of_x.squeeze().detach().numpy()
            v_x = v_x.detach().numpy()
            v_t = v_t.detach().numpy()
            v_xx = v_xx.detach().numpy()
            X_n = X_n.detach().numpy()

        else:
            v_of_x = vfun.eval_V(t_n, X_n).T
            if n < len(vfun.t_vec_p) - 1:
                v_t = (vfun.eval_V(vfun.t_vec_p[n + 1], X_n).T - v_of_x) / delta_t
            else:
                v_t = (testOde.calc_end_reward(X_n).T - v_of_x)/tau # testOde.calc_end_reward ist aequivalent zu
            v_x = vfun.calc_grad(t_n, X_n).T
            v_xx = vfun.calc_hessian(t_n, X_n).transpose((2, 0, 1))
            X_n = X_n.T


        problem.modus = 'np'

        loss = problem.pde_loss(t_n, X_n, v_of_x, v_t, v_x, v_xx)
        avg_loss.append(np.mean(np.abs(loss)))

        problem.modus = problem_modus_temp

    return avg_loss
