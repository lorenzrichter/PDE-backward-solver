#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:58:33 2020

@author: sallandt

"""

import xerus as xe
import numpy as np
from scipy import linalg as la
import valuefunction_TT, ode, pol_it
import time
import pickle

run_set_dynamics = False
if run_set_dynamics == True:
    import set_dynamics
# nos = number of samples
# method either impl, expl, or semi
def main_p(nos = 2000, method='impl', problem='HJB'):    
    vfun = valuefunction_TT.Valuefunction_TT()
    # vfun.test()
    testOde = ode.Ode(problem)
    n_sweep = 1000 # sweeps in ALS
    rel_val_tol = 1e-4 # stop when relative residual in ALS is lower
    rel_tol = 1e-4 # stop when relative difference in iterative scheme is lower
    iterations = 100 # maximum iterations in the implicit scheme
    # method = 'impl'
    # method = 'nik'
    # method = 'expl'
    if method == 'expl' or method == 'nik':
        iterations = 1 # if not implicit set iterations to 1
    nos_test_set = nos  # number of samples of the test set
    # print('number of samples', nos)
    
    polit_params = [nos, nos_test_set, n_sweep, rel_val_tol, rel_tol, iterations]
    
    # testOde.test()
    testpolit = pol_it.Pol_it(vfun, testOde, polit_params)
    testpolit.method = method
    
    
    t00 = time.time()
    t01 = time.perf_counter()
    # solve HJB
    testpolit.solve_HJB()
    t10 = time.time()
    t11 = time.perf_counter()
    print('Solving the BSDE took:, perf_counter()', t11 - t01 )
    print('ref at x', testOde.compute_reference(0, np.load('x0.npy')))
    for i0 in range(len(vfun.V)):
        savestr = str(vfun.pol_deg - 1)
        # savestr = str(vfun.maxrank))
        pickle.dump(testpolit.v.V[i0], open('V_p'+savestr+'_'+method+'_{}'.format(str(i0)), 'wb'))
        np.save('c_add_fun_list_p'+savestr+'_'+method, testpolit.v.c_add_fun_list)
