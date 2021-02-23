#pylint: disable=invalid-name

import matplotlib.pyplot as plt
import numpy as np
import torch as pt

try:
    from pytorch_future import hessian
except ImportError:
    print('0')


device = pt.device('cpu')


def get_X_process(problem, K, delta_t, seed=42, x=None, t=0):
    '''
    :param problem: problem object that specifies the PDE problem
    :param K: batch size, i.e. number of samples
    :param delta_t: step size
    :param seed: seed for the random number generator
    :sigma: specifies whether the diffusion coefficient is constant or can depend on x.
            In the former case sigma is d x d, in the latter case it is K x d x d.
    '''

    np.random.seed(seed)

    N = int(np.ceil((problem.T - t) / delta_t))
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


def compute_PDE_loss(problem, delta_t=0.01, K=1000, Y_n=None, vfun=None, testOde=None, seed=44, print_every=None):

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
            v_t = v_t.detach().squeeze().numpy()
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

        if print_every is not None:
            if n % print_every == 0:
                print(n)

    return avg_loss


def plot_NN_evaluation(model, n, n_start=0, reference_solution=True, Y_0_true=None):
    
    model.problem.modus = 'np'
    X, xi = get_X_process(model.problem, model.K, model.delta_t, seed=44)
    model.problem.modus = 'pt'
    X = pt.tensor(X).float().to(device)

    if Y_0_true is None:
        Y_0_true = model.problem.v_true(model.problem.X_0[np.newaxis, :], 0)
        ref_loss = np.mean([np.mean((model.Y_n[n](X[n, :, :]).squeeze().detach().cpu().numpy() 
                         - model.problem.v_true(X[n, :, :].detach().cpu().numpy(), n * model.delta_t))**2) for n in range(model.N + 1)])

    Y_0_est = model.Y_n[0](pt.tensor(model.problem.X_0).to(device).float().unsqueeze(0)).detach().cpu().numpy()
    print('d = %d' % model.problem.d)
    print('Y_0 true:   %.5f' % Y_0_true)
    print('Y_0 est:    %.5f' % Y_0_est)
    print('rel error:  %.5f' % np.abs((Y_0_true - Y_0_est) / Y_0_true))
    if Y_0_true is None:
        print('ref loss:   %.5f' % ref_loss)
    print('time:       %d' % int(np.round(np.sum(model.runtimes))))



    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    ax[0, 0].plot([item for sublist in model.loss_log for item in sublist])
    ax[0, 0].set_yscale('log')
    ax[0, 1].set_title('n = %d/%d' % (n, model.N))
    ax[0, 1].plot(model.loss_log[model.N - n - 1])
    ax[0, 1].set_yscale('log')

    1.1 * pt.min(X[n, :, 0])

    X_val = pt.linspace(1.1 * pt.min(X[n, :, 0]), 0.9 * pt.max(X[n, :, 0]), 500).unsqueeze(1).repeat(1, model.problem.d).to(device)
    ax[1, 0].set_title('n = %d/%d' % (n, model.N))
    ax[1, 0].plot(X_val.cpu().numpy()[:, 0], model.Y_n[n](X_val).detach().cpu().numpy())
    if reference_solution:
        ax[1, 0].plot(X_val.cpu().numpy()[:, 0], model.problem.v_true(X_val.cpu().numpy(), n * model.delta_t).squeeze())

    ax[1, 1].scatter(pt.sum(X[n, :, :]**2, 1).detach().cpu().numpy(), model.Y_n[n](X[n, :, :]).detach().cpu().numpy(), s=0.5)
    if reference_solution:
        ax[1, 1].scatter(pt.sum(X[n, :, :]**2, 1).detach().cpu().numpy(), model.problem.v_true(X[n, :, :].detach().cpu().numpy(), n * model.delta_t).squeeze(), s=0.5);

    x = pt.tensor(model.problem.X_0).unsqueeze(0).float().to(device)
    t_val = np.linspace(0, model.problem.T, model.N + 1)
    ax[1, 2].set_title('x = %.2f' % x[0, 0])
    ax[1, 2].plot(t_val[n_start:], [y_n(x).item() for y_n in model.Y_n[n_start:]])
    if reference_solution:
        ax[1, 2].plot(t_val[n_start:], [model.problem.v_true(x.cpu().numpy(), t) for t in t_val[n_start:]])

    return fig
