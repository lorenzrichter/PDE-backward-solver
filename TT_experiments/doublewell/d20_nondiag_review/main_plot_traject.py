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
import matplotlib.pyplot as plt
import time
from utilities import get_X_process 

run_set_dynamics = True
if run_set_dynamics == True:
    import set_dynamics
    
load_num = 'V_p3_pham_'
x0 = np.load('x0.npy')

vfun = valuefunction_TT.Valuefunction_TT(load_num, True, 'c_add_fun_list_p3_pham.npy')
testOde = ode.Ode()
# vfun.set_add_fun(testOde.calc_end_reward, testOde.calc_end_reward_grad)
vfun.set_add_fun(testOde.calc_end_reward, testOde.calc_end_reward_grad, testOde.calc_end_reward_hess)
vfun.V[-1] = 0 * vfun.V[-1]
vfun.c_add_fun_list[-1] = 1
print('vfun.add_fun_list', vfun.c_add_fun_list)
# testOde.test()


t_vec_p = np.load('t_vec_p.npy')
T = t_vec_p[-1]
print('t_vec_p', t_vec_p, 'T', T)
n = vfun.V[0].order(); a = -1; b = 1
x = np.linspace(a, b, n)
max_num_initial_values = 1000
min_num_initial_values = 0

load_me = np.load('save_me.npy')
tau = load_me[4]

steps = np.linspace(0, T, int(T/tau)+1)
# print('steps', steps)
m = len(steps)


nos = 10
curr_t = 0.001


testOde.problem.X_0 = x0
testOde.problem.compute_reference_solution()
testOde.problem.compute_reference_solution_2()
ind = testOde.t_to_ind(curr_t)
seed = 44
delta_t = tau
K = nos
samples_mat, noise_vec =  get_X_process(testOde.problem, K, delta_t, seed)
samples_mat = samples_mat.transpose((2,1,0))
noise_vec = np.sqrt(testOde.tau)*noise_vec.transpose((2,1,0))
avg_loss_vec = []
avg_loss_true_vec = []


traject = samples_mat[:,0,:]
evaled = np.zeros(traject.shape[-1])
ref = np.zeros(evaled.shape)
traject2 = samples_mat[:,1,:]
evaled2 = np.zeros(traject.shape[-1])
ref2 = np.zeros(evaled.shape)
for i0 in range(evaled.size):
    print('i0', i0)
    evaled[i0] = vfun.eval_V(t_vec_p[i0], traject[:, i0])
    ref[i0] = testOde.compute_reference(t_vec_p[i0], traject[:, i0])
    evaled2[i0] = vfun.eval_V(t_vec_p[i0], traject2[:, i0])
    ref2[i0] = testOde.compute_reference(t_vec_p[i0], traject2[:, i0])

np.save("v_tt_traj1", evaled)
np.save("v_ref_traj1", ref)
np.save("v_tt_traj2", evaled2)
np.save("v_ref_traj2", ref2)
plt.figure()
plt.plot(t_vec_p, evaled, c='b')
plt.plot(t_vec_p, ref, c='orange', linestyle='dashed')
plt.plot(t_vec_p, evaled2, c='b')
plt.plot(t_vec_p, ref2, c='orange', linestyle = 'dashed')
plt.xlabel('t')
plt.legend(['TT', 'ref'])
plt.title('Plot along single trajectory')
plt.show()




