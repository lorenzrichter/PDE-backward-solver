#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xerus as xe
import numpy as np
import set_dynamics, main_p, main_t

# Main program that runs the experiments for different
# parameters. Note that the solutions are saved
# for later usage.
# If you do not know what you are doing, only alter the 'setup' parameter to
# recreate the tests from the paper.
# 
# method_vec = list of number of time-steps
# pol_deg_vec: list of polynomial degrees
# num_samples:  list of number of samples
# d_vec: list of spatial dimension of the PDE
# maxrankvec = list of (uniformly) maximal ranks
# num_valuefunctions_vec = list of number of time-steps
# final_time = final time
# bounds = [min, max] of orthonormality of H^2 polynomials
# x0_enty = initial value x0 = [x0_entry] * dimension


# List of possible setups, do not change
setups = ['custom', 'HJB100', 'HJBdiffdim', 'DoubleWell50', 'DoubleWell20', \
        'CIR', 'UnboundedSin']

 # Alter the setup parameter to choose a test from the list above
setup = 'custom'


assert setup in setups, "Setup must be premade or 'custom', choose from list\
        'setups'"
print('run setup', setup)




# if setup is custom, vary the different parameters below to get different
# results
if setup == 'custom':
    method_vec = ['impl', 'expl']
    problems = ['LLGC', 'CosExp', 'AllenCahn', 'UnboundedSin', 'CIR', 'HJB', \
            'HJBcos', 'Heat', 'DoubleWell_diag', 'DoubleWell_nondiag']
    problem = 'DoubleWell_diag'
    # pol_deg_vec = np.arange(5)
    pol_deg_vec = [3]
    num_valuefunctions_vec = [51]
    d_vec = [50]
    maxrankvec = [2]
    num_samples = [2000]
    method_vec = ['semi', 'impl', 'expl']
    final_time = 0.5
    bounds = [-3, 3]
    x0_entry = -1
    seed_vec = np.arange(50, 150)
    assert problem in problems, "problem must be premade, choose from list \
            of problems"
elif setup == 'HJB100':
    problem = 'HJB'
    pol_deg_vec = [0]
    num_valuefunctions_vec = [101]
    d_vec = [100]
    maxrankvec = [1]
    num_samples = [2000]
    method_vec = ['semi', 'impl', 'expl']
    final_time = 1
    bounds = [-6, 6]
    x0_entry = 0
    seed_vec = [42]
elif setup == 'HJBdiffdim':
    problem = 'HJB'
    pol_deg_vec = np.arange(5)
    num_valuefunctions_vec = [101]
    d_vec = [1, 2, 5, 10, 50, 100]
    maxrankvec = [1]
    num_samples = [2000]
    method_vec = ['semi', 'impl']
    final_time = 1
    bounds = [-6, 6]
    x0_entry = 0
    seed_vec = [42]
elif setup == 'DoubleWell50':
    problem = 'DoubleWell_diag'
    # pol_deg_vec = np.arange(5)
    pol_deg_vec = [3]
    num_valuefunctions_vec = [51]
    d_vec = [50]
    maxrankvec = [2]
    num_samples = [2000]
    method_vec = ['semi', 'impl', 'expl']
    final_time = 0.5
    bounds = [-3, 3]
    x0_entry = -1
    seed_vec = [42]
elif setup == 'DoubleWell20':
    problem = 'DoubleWell_nondiag'
    pol_deg_vec = [7]
    num_valuefunctions_vec = [31]
    d_vec = [20]
    maxrankvec = [6]
    num_samples = [2000]
    # method_vec = ['expl']
    method_vec = ['semi', 'impl', 'expl']
    final_time = 0.3
    bounds = [-8, 2]
    x0_entry = -1
    seed_vec = [42]
elif setup == 'CIR':
    problem = 'CIR'
    pol_deg_vec = np.arange(1, 4)
    num_valuefunctions_vec = [101]
    d_vec = [100]
    maxrankvec = [1]
    num_samples = [2000]
    # method_vec = ['semi', 'impl', 'expl']
    method_vec = ['semi', 'impl']
    final_time = 1
    bounds = [-0.2, 6]
    x0_entry = 1
    seed_vec = [42]


# Iterate through different setups
# First set the dynamics, then solve the BSDE, then evaluate the result
errorvec = []
for i2 in num_samples:
    for i1 in num_valuefunctions_vec:
        for i4 in maxrankvec:
            for i3 in d_vec:
                for i0 in pol_deg_vec:
                    for i5 in method_vec:
                        resultvec = []
                        errorvec = []
                        runtimevec = []
                        reference_lossvec = []
                        PDE_lossvec = []
                        for i6, seed in enumerate(seed_vec):
                            print('START with number of samples: {}, number of time steps: {}, maxrank: {}, dimension: {}, polynomial degree: {}, method: {}, seed: {}'.format(i2, i1, i4, i3, i0, i5, seed))
                            set_dynamics.set_dynamics(i0, i1, i3, i4, final_time,\
                                    bounds, x0_entry*np.ones(i3), seed+2000)
                            result, error, runtime = main_p.main_p(i2, i5, problem, seed)
                            resultvec.append(result)
                            errorvec.append(error)
                            runtimevec.append(runtime)
                            np.save('resultvec_'+str(i5)+'_'+str(i4)+'_'+str(i3)+'_'+str(i2)+'_'+str(i1)+'_'+str(i0), resultvec)
                            np.save('errorvec_'+str(i5)+'_'+str(i4)+'_'+str(i3)+'_'+str(i2)+'_'+str(i1)+'_'+str(i0), errorvec)
                            np.save('runtimevec_'+str(i5)+'_'+str(i4)+'_'+str(i3)+'_'+str(i2)+'_'+str(i1)+'_'+str(i0), runtimevec)
                            print('np.mean(errorvec)', np.mean(errorvec))
                            print('np.std(errorvec)', np.std(errorvec))
                            print('np.mean(runtimevec)', np.mean(runtimevec))
                            print('np.std(runtimevec)', np.std(runtimevec))
                            print('test the computed solutions with polynomial degree {} and method {}'.format(i0, i5))
                            reference_loss, PDE_loss = main_t.run([i0], [i5], problem, seed+1000, save=False)
                            reference_lossvec.append(reference_loss)
                            PDE_lossvec.append(PDE_loss)
                            # print('reference_lossvec', reference_lossvec)
                            # print('PDE_lossvec', PDE_lossvec)
                            print('np.mean(reference_loss)', np.mean(reference_lossvec))
                            print('np.std(reference_loss)', np.std(reference_lossvec))
                            print('np.mean(PDE_lossvec)', np.mean(PDE_lossvec))
                            print('np.std(PDE_lossvec)', np.std(PDE_lossvec))
                if len(seed_vec) == 1:
                    if not setup == 'HJBdiffdim':
                        print('test the computed solutions with polynomial degrees {} and methods {}'.format(pol_deg_vec, method_vec))
