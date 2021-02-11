
import xerus as xe
import numpy as np
from scipy import linalg as la
import scipy
import matplotlib.pyplot as plt
import pickle

from utilities import get_X_process 
import problems


class Pol_it:
    def __init__(self, initial_valuefun, ode, polit_params):
        self.v = initial_valuefun
        self.ode = ode
        self.v.set_add_fun(self.ode.calc_end_reward, self.ode.calc_end_reward_grad, self.ode.calc_end_reward_hess)
        [self.nos, self.nos_test_set, self.n_sweep, self.rel_val_tol, self.rel_tol, self.max_pol_iter, self.max_iter_Phi, self.horizon] = polit_params
        load_me = np.load('save_me.npy')
        self.interval_half = load_me[2]
        self.t_vec_p = self.v.t_vec_p
        self.t_vec_s = self.ode.t_vec_s
        self.current_time = 0
        self.current_end_time = 0
        self.curr_ind = 0
        self.x0 = np.load('x0.npy')
        self.samples, self.samples_test, self.noise_vec, self.noise_vec_test = self.build_samples(-self.interval_half, self.interval_half)
        self.data_x = self.v.prepare_data_before_opt(self.samples)
        self.constraints_list = self.construct_constraints_list()
        self.c = self.ode.calc_end_reward(self.t_vec_p[-1], self.samples[:,:,-1])
        self.c_test = self.ode.calc_end_reward(self.t_vec_p[-1], self.samples_test[:,:,-1])



#         vec1 = 1*np.ones((self.ode.n, 1))
#         vec1 = np.random.uniform(-2, 2, vec1.shape)
#         vec2 = 1*vec1
#         eps = 1e-4
#         vec2[0,0] += eps
#         res1 = self.v.add_fun(0.3, vec1)
#         res2 = self.v.add_fun(0.3, vec2)
#         res3 = self.v.grad_add_fun(1, vec1)
#         print('vec1', vec1)
#         print('vec2', vec2)
#         print('grad', res3)
#         print('discr', (res2 - res1) / eps)
#         print('analytic', res3[0])
#         input()


    def build_samples(self, samples_min, samples_max):
        self.ode.problem.X_0 = self.x0
        seed = 42
        delta_t = self.ode.tau
        K = self.nos
        samples_mat, noise_vec =  get_X_process(self.ode.problem, K, delta_t, seed)
        samples_mat = samples_mat.transpose((2,1,0))
        noise_vec = np.sqrt(self.ode.tau)*noise_vec.transpose((2,1,0))
        seed = seed+1
        K = self.nos_test_set
        samples_mat_test, noise_vec_test =  get_X_process(self.ode.problem, K, delta_t, seed)
        samples_mat_test = samples_mat_test.transpose((2,1,0))
        noise_vec_test = np.sqrt(self.ode.tau)*noise_vec_test.transpose((2,1,0))
        print(samples_mat.shape, noise_vec.shape)
        print('min, max', np.min(samples_mat), np.max(samples_mat))
        # input()

        return samples_mat, samples_mat_test, noise_vec, noise_vec_test

    def construct_constraints_list(self):
        # return None
        n = self.ode.n
        xvec = np.zeros(shape=(n, 1))
        P_list = self.v.P_batch(xvec)
        # dP_list = self.v.dP_batch(xvec)
        # for i0 in range(n):
            # P_list[i0][:,i0] = dP_list[i0][:,i0]
        return P_list

    
    def solve_HJB(self, start_num = None):
        if type(start_num) is not int:
            start_num = len(self.t_vec_p) -2
        pickle.dump(self.v.V[len(self.t_vec_p) -1], open('V_{}'.format(str(len(self.t_vec_p) -1)), 'wb'))
        
        for i0 in range(start_num, -1, -1):
            self.curr_ind = i0
            if i0 is not start_num:
                print('set V', i0)
                self.v.V[self.curr_ind] = self.v.V[self.curr_ind+1]
                self.v.c_add_fun_list[self.curr_ind] = 1*self.v.c_add_fun_list[self.curr_ind+1] 
            self.current_time = self.v.t_vec_p[i0]
            ind_end = np.minimum(len(self.t_vec_p) - 1, i0+self.horizon)
            self.current_end_time = self.v.t_vec_p[ind_end]
            print('ind_end', ind_end, 't_start, t_end', self.current_time, self.current_end_time)
            self.solve_HJB_fixed_time()


    def solve_HJB_fixed_time(self):
        pol_iter = 0 
        rel_diff = 1
        pol_it_counter = 0
        rel_diff_eval = 1
        while(rel_diff > self.rel_tol and pol_iter < self.max_pol_iter and rel_diff_eval > self.rel_tol/10):
            pol_iter += 1
            V_old = 1*self.v.V[self.curr_ind]
            curr_samples, rew_MC = self.build_rhs_batch()
            curr_samples_test, rew_MC_test = self.build_rhs_batch(True)
            eval_V_before = self.v.eval_V(self.current_time, curr_samples_test) 
            calc_grad_before = self.v.calc_grad(self.current_time, curr_samples_test)
            if self.current_time == 0:
                avg = 1/self.samples.shape[1]*np.sum(rew_MC)
                # rew_MC[:] = avg
                # curr_samples = curr_samples + np.random.randn(curr_samples.shape[0], curr_samples.shape[1])
                # self.v.V[0].round([1]*len(self.v.V[0].ranks()))
                print('RESULT v(x)', avg)
    
            data_y = self.v.prepare_data_while_opt(self.current_time, curr_samples)
    
            # data = [self.data_x, data_y[0], data_y[1], rew_MC, self.constraints_list, self.calc_mean_error, curr_samples_test, rew_MC_test]
            data = [self.data_x, data_y[0], data_y[1], data_y[2], rew_MC, self.constraints_list, self.calc_mean_error, curr_samples_test, rew_MC_test]
            params = [self.n_sweep, self.rel_val_tol]
            
            self.v.solve_linear_HJB(data, params)
            eval_V_after = self.v.eval_V(self.current_time, curr_samples_test) 
            calc_grad_after = self.v.calc_grad(self.current_time, curr_samples_test)
            # postprocessing
            # input()
            pickle.dump(self.v.V[self.curr_ind], open('V_{}'.format(str(self.curr_ind)), 'wb'))
            np.save('c_add_fun_list', self.v.c_add_fun_list)
            try:
                rel_diff_eval = (la.norm(eval_V_before - eval_V_after) + la.norm(calc_grad_before - calc_grad_after)) / la.norm(eval_V_before + la.norm(calc_grad_before - calc_grad_after))
            except:
                rel_diff = 1

            try:
                rel_diff = xe.frob_norm(self.v.V[self.curr_ind] - V_old) / xe.frob_norm(V_old)
            except:
                rel_diff = 1
            pol_it_counter += 1
            print('num', pol_it_counter, "rel_diff", rel_diff, 'frob_norm(V)', xe.frob_norm(self.v.V[self.curr_ind]), 'frob_norm(V_old)', xe.frob_norm(V_old), 'rel_diff_eval', rel_diff_eval)
            if self.current_time < -0.5 and self.ode.n == 1:
                plt.figure()
                plt.scatter(curr_samples.reshape((rew_MC.size)), rew_MC, s=0.1)
                plt.scatter(curr_samples.reshape((rew_MC.size)), self.ode.compute_reference(self.current_time, curr_samples), c='g', s=0.1)
                plt.scatter(curr_samples.reshape((rew_MC.size)), self.v.eval_V(self.current_time, curr_samples), c='r', s=0.1)
                plt.show()
        # if pol_iter == self.max_pol_iter:
            # input('max_pol_iter reached')
        if self.current_time != 0:
            self.c = self.v.eval_V(self.current_time, self.samples[:,:,self.curr_ind])
            self.c_test = self.v.eval_V(self.current_time, self.samples_test[:,:,self.curr_ind])
        # mean_error_test_set = self.calc_mean_error(self.samples_test, y_mat_test, rew_MC_test)
        # print('frob_norm(V)', xe.frob_norm(self.v.V[self.curr_ind]))
    
    def calc_mean_error(self, V, xmat, rew_MC):
        error = (self.v.eval_V(V, xmat) - rew_MC)
        return np.linalg.norm(error)**2 / rew_MC.size


    def calc_u(self, t, x):
        grad = self.v.calc_grad(t, x)
        return self.ode.calc_u(t, x, grad)

    
    def build_rhs_batch(self, validation = False):
        points = np.linspace(self.v.t_to_ind(self.current_time), self.v.t_to_ind(self.current_end_time), 2)
        point = int(np.round(points[0]))
        print('point', point)
        if not validation:
            curr_samples = self.samples[:,:,point]
            curr_samples_next = self.samples[:,:,point+1]
            curr_c = self.c
            noise = self.noise_vec[:,:,point+1]
        else:
            curr_samples = self.samples_test[:,:,point]
            curr_samples_next = self.samples_test[:,:,point+1]
            curr_c = self.c_test
            noise = self.noise_vec_test[:,:,point+1]
        def calc_z(t, x):
            if self.ode.problem.sigma_modus == 'constant':
                return (self.ode.problem.sigma(x.T).T @ self.v.calc_grad(t, x))
            else:
                return np.einsum('ikj,ki->ji', self.ode.problem.sigma(x.T), self.v.calc_grad(t, x))

        if self.method == 'pham':
            y = self.v.eval_V(self.current_time, curr_samples)
            # z = self.ode.problem.sigma(curr_samples).T @ self.v.calc_grad(self.current_time, curr_samples)
            z = calc_z(self.current_time, curr_samples)
            rew_MC = curr_c + self.v.tau * self.ode.h(self.current_time, curr_samples, y, z) - np.sum(z*noise, axis=0)
        elif self.method == 'nik':
            y = self.v.eval_V(self.current_time+self.v.tau, curr_samples_next)
            # z = self.ode.problem.sigma(curr_samples).T @ self.v.calc_grad(self.current_time + self.v.tau, curr_samples)
            z = calc_z(self.current_time + self.v.tau, curr_samples_next)

            # hess = self.ode.problem.sigma(curr_samples)[0,0]*self.v.calc_hessian(self.current_time + self.v.tau, curr_samples)
            # laplace = np.trace(hess)
            laplace = self.ode.problem.sigma(curr_samples_next)[0,0]*self.v.calc_laplace(self.current_time + self.v.tau, curr_samples_next)
            rew_MC = curr_c + self.v.tau * (self.ode.h(self.current_time+self.v.tau, curr_samples_next, y, z) - laplace) - np.sum(z*noise, axis=0)

        else:
            y = self.v.eval_V(self.current_time+self.v.tau, curr_samples_next)
            # z = self.ode.problem.sigma(curr_samples).T @ self.v.calc_grad(self.current_time + self.v.tau, curr_samples)
            z = calc_z(self.current_time + self.v.tau, curr_samples_next)
            rew_MC = curr_c + self.v.tau * self.ode.h(self.current_time, curr_samples_next, y, z)

        # rew_MC = self.ode.compute_reference(self.current_time, curr_samples)
        return curr_samples, rew_MC


