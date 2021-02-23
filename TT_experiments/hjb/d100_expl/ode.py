import xerus as xe
import numpy as np
from scipy import linalg as la
import scipy.integrate as integrate
import problems


class Ode:
    def __init__(self):
        load_me = np.load('save_me.npy')
        self.t_vec_s = np.load('t_vec_s.npy')
        self.tau = load_me[4]    
        self.n = int(load_me[5])
        load_me = np.load('save_me.npy')
        self.T = self.t_vec_s[-1]
        self.problem = problems.HJB(d=self.n, T = self.T)

    def calc_end_reward(self, t, x):
        return self.problem.g(x.T)


    def calc_end_reward(self, t, x):
        return self.problem.g(x.T)


    def calc_end_reward_grad(self, t, x):
        return 2 * x / (np.sum(x**2, axis=0) + 1)
    
    def calc_end_reward_hess(self, t, x):
        sum_x_plus1 = np.sum(x**2, axis=0) + 1
        sum_x_plus1_squared = sum_x_plus1**2
        if len(x.shape) == 1:
            ret =  np.tensordot(x, -4*x/sum_x_plus1_squared, axes=((),()))
            ret[range(self.n), range(self.n)] = 2 * (-2 * x**2 + sum_x_plus1) / sum_x_plus1_squared
            return ret
        else:
            ret = np.einsum('ik,jk,k->ijk', x, x, -4/(sum_x_plus1_squared))
            # print('(1.28*x**2 - 0.8) / denominator', 1.28*x**2 / denominator  - 0.8 / (0.4*la.norm(x, axis = 0)**2 + 2)**2)
            ret[range(self.n), range(self.n),:] = 2 * (-2 * x**2 + sum_x_plus1) / sum_x_plus1_squared
            return ret

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
