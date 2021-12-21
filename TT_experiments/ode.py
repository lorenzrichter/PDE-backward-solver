#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Here the SDE is handeled

import xerus as xe
import numpy as np
from scipy import linalg as la
import scipy.integrate as integrate
import sys
sys.path.insert(0, '..')
import problems


class Ode:
    def __init__(self, problem):
        load_me = np.load('save_me.npy')
        self.t_vec_s = np.load('t_vec_s.npy')
        self.tau = load_me[4]    
        self.n = int(load_me[5])
        load_me = np.load('save_me.npy')
        self.T = self.t_vec_s[-1]
        # problems = ['LLGC', 'CosExp', 'AllenCahn', 'UnboundedSin', 'CIR', 'HJB', 'HJBcos', 'Heat', 'Double well']
        if problem == 'LLGC':
            self.problem = problems.LLGC(d=self.n, T = self.T)
        elif problem == 'CosExp':
            self.problem = problems.CosExp(d=self.n, T = self.T)
        elif problem == 'AllenCahn':
            self.problem = problems.AllenCahn(d=self.n, T = self.T)
        elif problem == 'UnboundedSin':
            self.problem = problems.UnboundedSin(d=self.n, T = self.T)
        elif problem == 'CIR':
            self.problem = problems.BondpriceMultidim(d=self.n, T = self.T)
        elif problem == 'HJB':
            self.problem = problems.HJB(d=self.n, T = self.T)
        elif problem == 'HJBcos':
            self.problem = problems.HJBcos(d=self.n, T = self.T)
        elif problem == 'Heat':
            self.problem = problems.Heat(d=self.n, T = self.T)
        elif problem == 'DoubleWell_diag':
            self.problem = problems.DoubleWell(d=self.n, d_1=self.n, d_2=0, T = self.T, eta=.05, kappa=.1, diagonal = True)
            self.problem.compute_reference_solution()
            self.problem.compute_reference_solution_2()
        elif problem == 'DoubleWell_nondiag':
            self.problem = problems.DoubleWell(d=self.n, T = self.T, diagonal = False, kappa=0.5, eta=0.5)
        else:
            print('no known problem was inserted')




    def calc_end_reward(self, t, x):
        return self.problem.g(x.T)


    def calc_end_reward(self, t, x):
        return self.problem.g(x.T)


    def calc_end_reward_grad(self, t, x):
        return self.problem.g_grad(x)

    
    def calc_end_reward_hess(self, t, x):
        return self.problem.g_hess(x)

    def calc_end_reward_laplace(self, t, x):
        pass
    

    def h(self, t, x, y, z):
        return self.problem.h(t,x.T,y,z.T)


    def t_to_ind(self, t):
        # print('t, t/self.tau, int(t/self.tau', t, t/self.tau, int(t/self.tau))
        return int(np.round(t/self.tau, 12))

    def compute_reference(self, t, x):
        return self.problem.v_true(x.T, t)


    def test(self):
        t = 0.
        n = self.n
        m = self.n
        start = np.ones((n,2))
        print('start', start)
        control = np.zeros((m, 2))
        end = self.step(0, start, control)
        print('end', end)
        print('rewards', self.calc_reward(t, start, control), self.calc_reward(t, end, control))
        print('control', self.calc_u(t, start, start))
        return 0



# testOde = Ode()
# testOde.test()
