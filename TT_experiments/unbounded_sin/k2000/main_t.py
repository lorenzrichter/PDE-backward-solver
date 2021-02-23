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
    
def main_t(pol_deg, testOde, _type, plot_vec):
    load_num = 'V_p'+str(pol_deg)+'_'+_type+'_'
    x0 = np.load('x0.npy')
    
    vfun = valuefunction_TT.Valuefunction_TT(load_num, True, 'c_add_fun_list_p'+str(pol_deg)+'_'+_type+'.npy')
    # vfun.set_add_fun(testOde.calc_end_reward, testOde.calc_end_reward_grad)
    vfun.set_add_fun(testOde.calc_end_reward, testOde.calc_end_reward_grad, testOde.calc_end_reward_hess)
    vfun.V[-1] = 0 * vfun.V[-1]
    pol_deg = np.max(vfun.V[-1].dimensions)
    vfun.c_add_fun_list[-1] = 1
    print('vfun.add_fun_list', vfun.c_add_fun_list)
    # testOde.test()
    
    
    t_vec_p = np.load('t_vec_p.npy')
    T = t_vec_p[-1]
     #print('t_vec_p', t_vec_p, 'T', T)
    n = vfun.V[0].order(); a = -1; b = 1
    x = np.linspace(a, b, n)
    
    load_me = np.load('save_me.npy')
    tau = load_me[4]
    
    testOde.problem.X_0 = x0
    curr_t = 0.1
    ind = testOde.t_to_ind(curr_t)
    avg_loss_vec = []
    avg_loss_true_vec = []
    
    vec = np.zeros((n, numsteps))
    ref_vec = []
    
    def eval_v_repeat(t, x):
        return vfun.eval_V(t, (x*np.ones((n, x.size))))
        # return vfun.eval_V(t, (x*np.ones((n, x.size))).squeeze())
    
    x_vec = xx
    t_vec = vfun.t_vec_p[2:]
    values = np.zeros((len(t_vec), len(x_vec)))
    for i0 in range(len(t_vec)):
        # print('t_vec[i0]', t_vec[i0], i0)
        values[i0,:] = eval_v_repeat(t_vec[i0], x_vec)
        
    # plt.contourf([t_vec, x_vec] , values)
    # plt.show()
    for i0 in range(numsteps):
        vec[:, i0] = xx[i0]
        # ref_vec.append(testOde.problem.v_true(vec[:, i0], point))
        ref_vec.append(0)
    
    eval_at_xx = vfun.eval_V(point, vec)
    
    
    v_xo = []
    v_xo_ref = []
    
    for ind in range(len(vfun.t_vec_p)-1):
        curr_t = vfun.t_vec_p[ind]
        curr_x = x0 + np.sqrt(curr_t) * testOde.problem.sigma(x0) @ np.ones(testOde.n)
        v_xo.append(vfun.eval_V(curr_t, curr_x))
        # v_xo_ref.append(0)
        v_xo_ref.append(testOde.problem.v_true(curr_x[None, :], curr_t))
    
    
    
    # for ind in range(0, -1, 1):
    for ind in plot_vec:
        # print('ind', ind)
        samples = samples_mat[:,:,ind]
        curr_t = vfun.t_vec_p[ind]
        # print('t',curr_t,'ind', ind)
        # print('np.min, np.max (samples)', np.min(samples), np.max(samples))
        # print('samples', samples)
        
        
        
        
        v_of_x = vfun.eval_V(curr_t, samples).T
        # v_true = np.zeros(v_of_x.shape)
        # for i0 in range(v_true.size):
            # v_true[i0] = testOde.problem.v_true(samples[:, i0], curr_t)
        v_true = testOde.problem.v_true(samples.T, curr_t)
        # if ind == 698 or ind == 699 or ind == 700:
            # print('v_of_x', v_of_x)
            # print('v_true', v_true)
    
        # print('vofx', v_of_x)
        # print('v_next',vfun.eval_V(curr_t+tau, samples).T)
        if ind < len(vfun.t_vec_p) - 1:
            v_t  =  (vfun.eval_V(vfun.t_vec_p[ind+1], samples).T - v_of_x)/tau
        else:
            v_t  =  (testOde.calc_end_reward(samples).T - v_of_x)/tau
        v_x = vfun.calc_grad(curr_t, samples).T
        v_xx = vfun.calc_hessian(curr_t, samples).transpose((2,0,1))
        loss = testOde.problem.pde_loss(curr_t, samples.T, v_of_x, v_t, v_x, v_xx)
        loss_true = v_of_x - v_true
        avg_loss_vec.append(np.abs(loss))
        avg_loss_true_vec.append(np.abs(loss_true / v_true))
        # avg_loss_true_vec.append(np.abs(loss_true))
        # print('rel_loss_true[-1]', avg_loss_true_vec[-1])
        # print('rel_loss_true[-1]', np.mean(avg_loss_true_vec[-1]))
    avg_loss_true_vec = np.vstack(avg_loss_true_vec)
    avg_loss_vec = np.vstack(avg_loss_vec)
    # avg_loss_true_vec = None
    # avg_loss_vec = None
    # print('avg_loss_true_vec', avg_loss_true_vec.shape)
    print('avg loss pde', np.mean(avg_loss_vec[:-1]))
    print('avg loss true ', np.mean(avg_loss_true_vec[:-1]))
    
    return values, eval_at_xx, v_xo, avg_loss_vec, avg_loss_true_vec, v_xo_ref


load_me = np.load('save_me.npy')
tau = load_me[4]
testOde = ode.Ode()
seed = 44
delta_t = tau
# nos = 3
nos = 1000
K = nos
samples_mat, noise_vec =  get_X_process(testOde.problem, K, delta_t, seed)
samples_mat = samples_mat.transpose((2,1,0))
noise_vec = np.sqrt(testOde.tau)*noise_vec.transpose((2,1,0))
min_pol_deg = 7
max_pol_deg = 8
v_x0 = []
# for i0 in range(samples_mat.shape[2]):
    # avg_norm = np.mean(la.norm(samples_mat[:,:,i0], axis=0))
    # print('avg_norm', avg_norm)


values = []
eval_at_x = []
avg_loss = []
v_xo_list = []
avg_loss_true = []

t_vec_p = np.load('t_vec_p.npy')
plot_vec = range(0, len(t_vec_p) - 1, 1)
T = t_vec_p[-1]
steps = np.linspace(0, T, int(T/tau)+1)
# print('steps', steps)
m = len(steps)

numsteps = 100
xx = np.linspace(-1, 1, numsteps)
point = 0.02
_type = 'pham'
# _type = 'nik'
# _type = 'expl'
for i0 in range(min_pol_deg, max_pol_deg):
    print('i0', i0, 'maxpoldeg', max_pol_deg)
    ret = main_t(i0, testOde, _type, plot_vec)
    values.append(ret[0])
    eval_at_x.append(ret[1])
    v_xo_list.append(ret[2])
    avg_loss.append(ret[3])
    avg_loss_true.append(ret[4])
    v_xo_ref = ret[5]


# values_ref = [0.28407, 0.20131, 0.15746, 0.13036, 0.11179, 0.098481, 0.088329, 0.080428, 0.07406, 0.06887, 0.06455, 0.060893, 0.057786, 0.055111, 0.052802]
# print('v_xo', v_xo)
# values_ref.reverse()
# values_ref.append(0.5)

t_points = np.linspace(0, 0.3, 16)


# plt.figure()
# plt.contourf(values.T, 100)
# plt.ylabel('space')
# plt.xlabel('time')
# plt.colorbar()

eval_ref = []
for i0 in range(len(xx)):
    eval_ref.append(testOde.problem.v_true(np.repeat(xx[i0], testOde.n), point))
plt.figure()
plt.plot(xx, eval_ref)
for i0 in range(len(eval_at_x)):
    plt.plot(xx, eval_at_x[i0])
# plt.plot(xx, ref_vec)
plt.legend(['tt'])
plt.legend(['2', '3', '4', '5', '6', '7', '8', ])
plt.title('plot at [x,x,x,x] for t ='+str(point))





plt.figure()
for i0 in range(len(eval_at_x)):
    plt.plot(steps[1:-1], v_xo_list[i0][1:])
plt.plot(steps[1:-1], v_xo_ref[1:])
plt.title('Time plot for fixed x=x_0')
plt.xlabel('t')
plt.legend(['2', '3', '4', '5', '6', '7', '8', ])
# plt.show()


avg_loss_list = []
for i0 in range(len(avg_loss)):
    avg_loss_list.append(np.mean(avg_loss[i0][1:, :]**2))
    # avg_loss_list.append(np.mean(avg_loss[i0][1:]))
    print('avg_loss', avg_loss_list[-1])
# plt.show()
plt.figure()
plt.plot(avg_loss_list)
plt.title('avgloss')

plt.figure()
for i0 in avg_loss_true:
    print('i0', i0.shape)
    plt.plot(plot_vec, np.mean(i0, axis=1))
plt.xlabel('t')
plt.title('mean relative error over time')
plt.show()
