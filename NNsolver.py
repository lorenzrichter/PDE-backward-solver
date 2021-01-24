#pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments

import numpy as np
import time
import torch as pt

from utilities import get_X_process


device = pt.device('cpu')


class NNSolver():

    def __init__(self, problem, name, learning_rates, gradient_steps, NN_class, K=1000,
                 K_batch=1000, delta_t=0.01, print_every=500, method='implicit', seed=42):
        self.problem = problem
        self.name = name
        self.method = method
        self.K = K
        self.K_batch = K_batch
        self.delta_t = delta_t
        self.sq_delta_t = pt.sqrt(pt.tensor(delta_t))
        self.N = int(np.ceil(problem.T / delta_t))
        self.learning_rates = learning_rates
        self.gradient_steps = gradient_steps
        self.NN_class = NN_class
        self.Y_n = [NN_class(problem.d, 1, lr=self.learning_rates[n], seed=seed) for n in range(self.N)] + [problem.g]

        # logging
        self.runtimes = []
        self.loss_log = []
        self.print_every = print_every

    def train(self):

        X, xi = get_X_process(self.problem, self.K, self.delta_t)

        self.problem.modus = 'pt'

        X = pt.autograd.Variable(pt.tensor(X).float(), requires_grad=True)
        xi = pt.tensor(xi).float()

        for n in range(self.N - 1, -1, -1):

            self.loss_log.append([])

            if n < self.N - 1:
                self.Y_n[n].load_state_dict(self.Y_n[n + 1].state_dict())

            for l in range(self.gradient_steps[n]):

                t_0 = time.time()

                batch = np.random.choice(np.arange(self.K), size=self.K_batch, replace=False)

                X_n = X[n, batch, :]
                X_n_1 = X[n + 1, batch, :]
                Y_eval = self.Y_n[n](X_n).squeeze().sum()
                Y_eval.backward(retain_graph=True)
                Z_n, = pt.autograd.grad(Y_eval, X_n, create_graph=True)
                sigma_Z_n = pt.mm(self.problem.sigma(X_n).t(), Z_n.t()).t()

                self.Y_n[n].zero_grad()
                #loss = pt.mean((Y_n[n + 1](X_n_1).squeeze() - Y_n[n](X_n).squeeze() 
                #                + problem.h(n * delta_t, X_n, Y_n[n + 1](X_n_1).squeeze(), sigma_Z_n) * delta_t
                #                - pt.sum(sigma_Z_n * xi[n + 1, batch, :], 1) * sq_delta_t)**2)
                loss = pt.mean((self.Y_n[n + 1](X_n_1).squeeze() - self.Y_n[n](X_n).squeeze() 
                                + self.problem.h(n * self.delta_t, X_n, self.Y_n[n](X_n).squeeze(), sigma_Z_n) * self.delta_t
                                - pt.sum(sigma_Z_n * xi[n + 1, batch, :], 1) * self.sq_delta_t)**2)

                #loss = pt.mean((Y_n[n + 1](X[n + 1, :, :]).squeeze() - Y_n[n](X[n, :, :]).squeeze() 
                #               + problem.h(n * delta_t, X[n, :, :], Y_n[n + 1](X[n + 1, :, :]).squeeze(), sigma_Z_n.detach()) * delta_t)**2)

                loss.backward(retain_graph=True)
                self.Y_n[n].optim.step()

                self.loss_log[-1].append(loss.item())

                t_1 = time.time()

                self.runtimes.append(t_1 - t_0)

                if l % self.print_every == 0:
                    print('%d - %d - %.4e - %.2f - est: %.2fmin' % (n, l, self.loss_log[-1][-1], self.runtimes[-1], ((sum(self.gradient_steps[:n+1]) - l) * np.mean(self.runtimes[-100:])) / 60))