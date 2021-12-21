#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Plot BSDE results along trajectories - not incorporated into the 'script.py' framework
import xerus as xe
import numpy as np
from scipy import linalg as la
import valuefunction_TT, ode, pol_it
import matplotlib.pyplot as plt
import time
from utilities import get_X_process 

# Set min and maximal polynomial degree
min_pol_deg = 1
max_pol_deg = 5

# Set type
_type = 'impl'
# _type = 'expl'

run_set_dynamics = False
if run_set_dynamics == True:
    import set_dynamics
    
x0 = np.load('x0.npy')

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
    
    traject = samples_mat[:,0,:]
    evaled = np.zeros(traject.shape[-1])
    ref = np.zeros(evaled.shape)
    traject2 = samples_mat[:,1,:]
    evaled2 = np.zeros(traject.shape[-1])
    ref2 = np.zeros(evaled.shape)
    for i0 in range(evaled.size):
        evaled[i0] = vfun.eval_V(t_vec_p[i0], traject[:, i0][:,None])
        evaled2[i0] = vfun.eval_V(t_vec_p[i0], traject2[:, i0][:,None])
    return evaled, evaled2


load_me = np.load('save_me.npy')
tau = load_me[4]
testOde = ode.Ode()
seed = 44
delta_t = tau
nos = 1000
K = nos
samples_mat, noise_vec =  get_X_process(testOde.problem, K, delta_t, seed)
samples_mat = samples_mat.transpose((2,1,0))
noise_vec = np.sqrt(testOde.tau)*noise_vec.transpose((2,1,0))
v_x0 = []

evaled1_list = []
evaled2_list = []

t_vec_p = np.load('t_vec_p.npy')
plot_vec = range(0, len(t_vec_p) - 1, 20)
T = t_vec_p[-1]
steps = np.linspace(0, T, int(T/tau)+1)
# print('steps', steps)
m = len(steps)

numsteps = 100
xx = np.linspace(-1, 1, numsteps)
point = 0.02
for i0 in range(min_pol_deg, max_pol_deg):
    ret = main_t(i0, testOde, _type, plot_vec)
    evaled1_list.append(ret[0])
    evaled2_list.append(ret[1])




traject = samples_mat[:,0,:]
evaled = np.zeros(traject.shape[-1])
ref = np.zeros(evaled.shape)
for i0 in range(evaled.size):
    print('i0', i0)
    # evaled[i0] = vfun.eval_V(t_vec_p[i0], traject[:, i0])
    ref[i0] = testOde.compute_reference(t_vec_p[i0], traject[:, i0])

# plt.figure()
# plt.plot(t_vec_p, evaled)
# plt.xlabel('t')
# plt.legend(['TT', 'ref'])
# plt.title('Plot along single trajectory')
# plt.show()


plt.figure()
plt.plot(t_vec_p, ref)
for i0 in range(len(evaled1_list)):
    plt.plot(t_vec_p, evaled1_list[i0])
    # plt.plot(t_vec_p, evaled2_list[i0])
plt.xlabel('t')
plt.legend(['ref', '1', '2', '3', '4', '5', '6'])
plt.title('Plot along single trajectory, different poly. deg.')
plt.show()


