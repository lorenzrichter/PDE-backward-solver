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
        self.problem = problems.UnboundedSin(d=self.n, T = self.T)

    def calc_end_reward(self, t, x):
        return self.problem.g(x.T)


    def calc_end_reward(self, t, x):
        return self.problem.g(x.T)


    def calc_end_reward_grad(self, t, x):
        a = 1.0 * np.arange(1, self.n + 1)
        return np.outer(a, np.sin(x.T @ (-1*a)))
    
    def calc_end_reward_hess(self, t, x):
        a = 1.0 * np.arange(1, self.n + 1)
        cos_value = -np.cos(x.T @ (1 * a))
        outer_1 = np.outer(a, cos_value)
        # print('cos_value.shape', cos_value)
        # print('outer_1.shape', outer_1)
        ret = np.tensordot(a, outer_1, axes=((),()))
        # print('ret.shape', ret)
        return np.tensordot(a, outer_1, axes=((),()))

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
