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
        # self.problem = problems.DoubleWell(d=self.n, T = self.T, diagonal = True)
        self.problem = problems.DoubleWell(d=self.n, T = self.T, diagonal = False, kappa=0.5, eta=0.5); print('warning low kappa/eta')

    def calc_end_reward(self, t, x):
        return self.problem.g(x.T)


    def calc_end_reward(self, t, x):
        return self.problem.g(x.T)


    def calc_end_reward_grad(self, t, x):
        if len(x.shape) == 1:
            return self.problem.eta_ * 2 * (x - 1)
        else:
            return self.problem.eta_[:, None] * 2 * (x - 1)
    
    def calc_end_reward_hess(self, t, x):
        ret = np.diag(2 * self.problem.eta_)
        if len(x.shape) == 1:
            return ret
        else:
            return np.repeat(ret[:,:,None], x.shape[1], axis=2)

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
