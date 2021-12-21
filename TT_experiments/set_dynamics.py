#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:02:03 2019

@author: sallandt

Builds system matrices and saves them. Also calculates an initial control.
"""
import xerus as xe
import numpy as np
from scipy import linalg as la
import pickle

def set_dynamics(pol_deg = None, num_valuefunctions = None, n=None, maxrank= 1, T = 1, interval_bounds=[-6,6], x0 = None):
    if n is None:
        n = 5
    if pol_deg is None:
        pol_deg = 2
    if num_valuefunctions is None:
        num_valuefunctions = 101
    if x0 is None:
        x0 = np.zeros(n)
    t_vec_p = np.linspace(0, T, num_valuefunctions)
    tau = np.round(T/(num_valuefunctions-1), 8)
    num_timesteps = int(np.round((T - 0) / tau))
    t_vec_s = np.linspace(0, T, num_timesteps+1)
    
    interval_min = interval_bounds[0] # integration area of HJB equation is [interval_min, interval_max]**n
    interval_max = interval_bounds[1]

    # print('x0', x0)
    
    rank = min(2, maxrank)
    
    # maxrank = 3
    
    
    b = 1 # left end of Domain
    a = -1 # right end of Domain
    lambd = 0.1 # cost parameter
    gamma = 0 # discount factor, 0 for no discount
    boundary = 'Neumann' # use 'Neumann' or "Dirichlet
    use_full_model = True # if False, model is reduced to r Dimensions
    r = n # Model is reduced to r dimensions, only if use_full_model == False
    load = np.zeros([9])
    alpha = 3
    load[0] = alpha; load[1] = gamma; load[2] = interval_min; load[3] = interval_max; load[4] = tau; load[5] = n; load[6] = maxrank
    std_deriv = 0.1
    gamma_vec = np.ones(n)
    
    np.random.seed(1)
    # print('L', L)
    np.save("x0", x0)
    np.save("save_me", load)
    np.save("save_me", load)
    np.save('t_vec_p', t_vec_p)
    np.save('t_vec_s', t_vec_s)
    #
    
    desired_ranks = [rank]*(n-1)
    V_new = 0*xe.TTTensor.random([pol_deg + 1]*n, desired_ranks)
    pickle.dump(V_new, open("V_new", 'wb'))
