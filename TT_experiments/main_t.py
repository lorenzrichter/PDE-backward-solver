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

run_set_dynamics = False
if run_set_dynamics == True:
    import set_dynamics
    
def run(pol_deg_vec, method_vec, problem):
    def main_t(pol_deg, testOde, _type, plot_vec):
        load_num = 'V_p'+str(pol_deg)+'_'+_type+'_'
        x0 = np.load('x0.npy')
        
        vfun = valuefunction_TT.Valuefunction_TT(load_num, True, 'c_add_fun_list_p'+str(pol_deg)+'_'+_type+'.npy')
        # vfun.set_add_fun(testOde.calc_end_reward, testOde.calc_end_reward_grad)
        vfun.set_add_fun(testOde.calc_end_reward, testOde.calc_end_reward_grad, testOde.calc_end_reward_hess)
        vfun.V[-1] = 0 * vfun.V[-1]
        pol_deg = np.max(vfun.V[-1].dimensions)
        vfun.c_add_fun_list[-1] = 1
        # print('vfun.add_fun_list', vfun.c_add_fun_list)
        # testOde.test()
        
        
        t_vec_p = np.load('t_vec_p.npy')
        T = t_vec_p[-1]
         #print('t_vec_p', t_vec_p, 'T', T)
        n = vfun.V[0].order(); a = -1; b = 1
        x = np.linspace(a, b, n)
        
        load_me = np.load('save_me.npy')
        tau = load_me[4]
        
        testOde.problem.X_0 = x0
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
        for i0 in range(numsteps):
            vec[:, i0] = xx[i0]
            # ref_vec.append(testOde.problem.v_true(vec[:, i0], point))
            ref_vec.append(0)
        
        eval_at_xx = vfun.eval_V(point, vec)
        
        
        v_xo = []
        v_xo_ref = []
        
        x0 = np.ones(x0.shape)
        for ind in range(len(vfun.t_vec_p) - 1):
            # print('vfun.V[ind].ranks()', vfun.V[ind].ranks())
            curr_t = vfun.t_vec_p[ind]
            v_xo.append(vfun.eval_V(curr_t, np.sqrt(2*curr_t) * x0))
            v_xo_ref.append(0)
            # v_xo_ref.append(testOde.problem.v_true(np.sqrt(2*curr_t) * x0, curr_t))
        
        
        
        # for ind in range(0, -1, 1):
        for ind in plot_vec:
            # print('ind', ind)
            samples = samples_mat[:,:,ind]
            curr_t = vfun.t_vec_p[ind]
            v_of_x = vfun.eval_V(curr_t, samples).T
            v_true = np.zeros(v_of_x.shape)
            for i0 in range(v_true.size):
                v_true[i0] = Y_ref[ind, i0]
                # v_true[i0] = testOde.problem.v_true(samples[:, i0], curr_t)
            # v_true = testOde.problem.v_true(samples.T, curr_t)
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
            # print(np.sort(np.abs(loss**2    ))[-20:])
            loss_true = v_of_x - v_true
            avg_loss_vec.append(np.abs(loss))
            avg_loss_true_vec.append(np.abs(loss_true / v_true))
            # print('avglosspde', np.mean(np.abs(loss)**2), 'max', np.max(np.abs(loss)**2))
            # print('rel_loss_true[-1]', avg_loss_true_vec[-1])
            # print('rel_loss_true[-1]', np.mean(avg_loss_true_vec[-1]), 'max', np.max(avg_loss_true_vec[-1]))
        avg_loss_true_vec = np.vstack(avg_loss_true_vec)
        avg_loss_vec = np.vstack(avg_loss_vec)
        # print('avg_loss_true_vec', np.mean(avg_loss_true_vec))
        # print('avg loss pde', np.mean(avg_loss_vec[:-1]))
        # print('avg loss true ', np.mean(avg_loss_true_vec[:-1]))
        
        return values, eval_at_xx, v_xo, avg_loss_vec, avg_loss_true_vec, v_xo_ref

    for method in method_vec:
        _type = method
        print('method:', method)
        load_me = np.load('save_me.npy')
        tau = load_me[4]
        testOde = ode.Ode(problem)
        if testOde.problem.name == 'DoubleWell':
            testOde.problem.compute_reference_solution()
            testOde.problem.compute_reference_solution_2()
        seed = 44
        delta_t = tau
        # nos = 3
        nos = 10
        K = nos
        samples_mat, noise_vec =  get_X_process(testOde.problem, K, delta_t, seed)
        # print('samples_mat', samples_mat)
        
        X = samples_mat
        N = X.shape[0]
        K = X.shape[1]
        
        samples_mat = samples_mat.transpose((2,1,0))
        # print('samples_mat.shape', samples_mat.shape)
        noise_vec = np.sqrt(testOde.tau)*noise_vec.transpose((2,1,0))
        t_vec_p = np.load('t_vec_p.npy')
        print('calculate reference')
        def sample_ref(x0, t):
            if len(x0.shape) == 1:
                X_, xi = get_X_process(testOde.problem, 1000, delta_t, seed=46, t=t, x=x0)
                return -np.log(np.mean(np.exp(-testOde.problem.g(X_[-1, :, :]))))
            else:
                ret = np.zeros(x0.shape[1])
                for i0 in range(x0.shape[1]):
                    X_, xi = get_X_process(testOde.problem, 1000, delta_t, seed=46, t=t, x=x0[:, i0])
                    ret[i0] = -np.log(np.mean(np.exp(-testOde.problem.g(X_[-1, :, :]))))
                return ret

        try:
            print('try loading reference values')
            Y_ref = np.load('Y_ref_d'+str(testOde.n)+'_'+str(problem)+'.npy')
        except:
            print('exception file does not exist (yet), compute reference values instead')
        # if True:
            Y_ref = np.zeros((samples_mat.shape[2], samples_mat.shape[1]))
            for ind in range(samples_mat.shape[2]):
                curr_t = t_vec_p[ind]
                samples = samples_mat[:,:,ind]
                for i0 in range(samples_mat.shape[1]):
                    if testOde.problem.name == 'DoubleWell':
                        if not testOde.problem.diagonal:
                            Y_ref[ind, i0] = sample_ref(samples[:, i0], curr_t)
                        else:
                            Y_ref[ind, i0] = testOde.problem.v_true(samples[:, i0], curr_t)
                    else:
                        Y_ref[ind, i0] = testOde.problem.v_true(samples[:, i0], curr_t)
            np.save('Y_ref_d'+str(testOde.n)+'_'+str(problem), Y_ref)
        print('done with calculating reference')
        
        v_x0 = []
        # for i0 in range(samples_mat.shape[2]):
            # avg_norm = np.mean(la.norm(samples_mat[:,:,i0], axis=0))
            # print('avg_norm', avg_norm)
        
        
        values = []
        eval_at_x = []
        avg_loss = []
        v_xo_list = []
        avg_loss_true = []
        
        plot_vec = range(0, len(t_vec_p) - 1, 1)
        T = t_vec_p[-1]
        steps = np.linspace(0, T, int(T/tau)+1)
        # print('steps', steps)
        m = len(steps)
        
        numsteps = 100
        xx = np.linspace(-1, 1, numsteps)
        point = testOde.T/5
        for i0 in range(len(pol_deg_vec)):
            # print('i0', i0, 'len(pol_deg_vec)', len(pol_deg_vec))
            ret = main_t(pol_deg_vec[i0], testOde, _type, plot_vec)
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
        
        # plt.figure()
        # plt.contourf(values.T, 100)
        # plt.ylabel('space')
        # plt.xlabel('time')
        # plt.colorbar()
        
        eval_ref = []
        for i0 in range(len(xx)):
            if testOde.problem.name == 'DoubleWell':
                if not testOde.problem.diagonal:
                    eval_ref.append(sample_ref(np.repeat(xx[i0], testOde.n), point))
                else:
                    eval_ref.append(testOde.problem.v_true(np.repeat(xx[i0], testOde.n), point))
            else:
                eval_ref.append(testOde.problem.v_true(np.repeat(xx[i0], testOde.n), point))
        plt.figure()
        plt.plot(xx, eval_ref)
        for i0 in range(len(eval_at_x)):
            plt.plot(xx, eval_at_x[i0])
        # plt.plot(xx, ref_vec)
        plt.legend(['tt'])
        plt.legend(['ref', '3', '4', '5', '6', '7', '8', ])
        plt.title('plot at [x,x,x,x] for t ='+str(point))
        
        
        
        
        
        plt.figure()
        for i0 in range(len(eval_at_x)):
            plt.plot(steps[:-1], v_xo_list[i0])
        plt.title('Time plot for fixed x=x_0')
        plt.xlabel('t')
        plt.legend(['2', '3', '4', '5', '6', '7', '8', ])
        
        
        avg_loss_list = []
        print('PDE LOSS')
        for i0 in range(len(avg_loss)):
            avg_loss_list.append(np.mean(avg_loss[i0][1:, :]**2))
            # avg_loss_list.append(np.mean(avg_loss[i0][1:]))
            print('avg PDE loss for polynomial degree = {}:'.format(pol_deg_vec[i0]), avg_loss_list[-1])
        plt.figure()
        plt.plot(avg_loss_list)
        plt.title('avg PDE loss')
        
        plt.figure()
        print('MEAN RELATIVE ERROR')
        for i0 in range(len(avg_loss_true)):
            # print('i0', i0.shape, 'mean error true', 
            print('avg loss for polynomial degree = {}:'.format(pol_deg_vec[i0]), np.mean(avg_loss_true[i0]))
            plt.plot(plot_vec, np.mean(avg_loss_true[i0], axis=1))
        plt.xlabel('t')
        plt.title('mean relative error over time')
        plt.show()
