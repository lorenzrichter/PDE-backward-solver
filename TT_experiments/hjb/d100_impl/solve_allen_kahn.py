#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg as la
import valuefunction_TT, ode, pol_it
import time
import matplotlib.pyplot as plt



d = 200
T = 0.3
tpoints = np.linspace(0, 0.3, 3001)
tau = tpoints[1] - tpoints[0]

a, b = -10, 10 # interval of the PDE
s = np.linspace(a, b, d) # gridpoints
nu = np.sqrt(2) # diffusion constant
boundary = 'Dirichlet' # use 'Neumann' or "Dirichlet
# boundary = 'Dirichlet' # use 'Neumann' or "Dirichlet
lambd = .1
if boundary == 'Dirichlet':
    print('Dirichlet boundary')
    h = (b - a) / (d + 1)
    A = -2 * np.diag(np.ones(d), 0) + np.diag(np.ones(d - 1), 1) + np.diag(np.ones(d - 1), -1)
    A = nu / h**2 * A
    Q = h * np.eye(d)
elif boundary== 'Neumann':
    print('Neumann boundary')
    h = (b - a) / (d - 1)             # step size in space
    A = -2 * np.diag(np.ones(d), 0) + np.diag(np.ones(d - 1), 1) + np.diag(np.ones(d - 1), -1)
    A[0, 1] = 2; A[d - 1, d - 2] = 2
    A = nu / h**2 * A
    Q = h * np.eye(d)
    Q[0, 0] /=2; Q[d - 1,d - 1] /= 2  # for neumann boundary
else:
    print('Wrong boundary!')
# _B = (np.bitwise_and(s > -0.4, s < 0.4))*1.0
# B = np.zeros(shape=(d, 1))   
# B[:, 0] = _B
# A = -np.eye(d)
B = 0.25 * np.eye(d)
control_dim = B.shape[1]

def initial_condition(t, x):
    return 1/(2+2/5*x**2)


x0 = initial_condition(0, s)

def step_euler(t, x):
    return x + tau * rhs(t, x)


def rhs(t, x):
    return A @ x + x - x**3



x = np.zeros((d, len(tpoints)))
x[:, 0] = x0
for i0 in range(len(tpoints) - 1):
    x[:, i0+1] = step_euler(tpoints[i0], x[:, i0])
    print(x[:, i0+1])
plt.figure()
plt.contourf(np.flip(x[90:109], axis=1))
plt.ylabel('space')
plt.xlabel('time')
plt.colorbar()
# plt.contourf([t_vec, x_vec] , values)
plt.show()

