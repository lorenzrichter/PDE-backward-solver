import matplotlib.pyplot as plt
import numpy as np
import torch as pt
import time
import sys

sys.path.insert(0, '..')

from NNsolver import NNSolver
from NNarchitectures import DenseNet_g
from problems import DoubleWell
from utilities import get_X_process, plot_NN_evaluation, compute_PDE_loss


t0 = time.time()
device = pt.device('cpu')
T = 0.5
problem = DoubleWell(d=50, d_1=50, d_2=0, T=0.5, eta=.05, kappa=.1)
problem.compute_reference_solution()
problem.compute_reference_solution_2()
K = 2000
K_batch = 2000
print_every = 500
delta_t = 0.01
sq_delta_t = pt.sqrt(pt.tensor(delta_t))
N = int(np.ceil(T / delta_t))
gradient_steps = (N - 1) * [2000] + [25000]
learning_rates = [0.0002] * (N - 1) + [0.0005]

model = NNSolver(problem, 'DoubleWell', learning_rates=learning_rates, gradient_steps=gradient_steps, NN_class=DenseNet_g, K=K, 
                 K_batch=K_batch, delta_t=delta_t, print_every=print_every, method='implicit')

model.Y_n = [DenseNet_g(problem.d, 1, lr=learning_rates[n], problem=problem).to(device) for n in range(N)] + [problem.g]
print('run train')
model.train()
t1 = time.time()
print('it took', t1 - t0)

