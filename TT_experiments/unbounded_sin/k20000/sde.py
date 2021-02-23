import xerus as xe
import numpy as np
from scipy import linalg as la
import scipy.integrate as integrate


class Ode:
    def __init__(self):
        load_me = np.load('save_me.npy')
        self.lambd = load_me[0]
        self.interval_half = load_me[2]
        self.tau = load_me[3]    
        self.n = int(load_me[4])
        self.sqrttau = np.sqrt(self.tau)
        self.sigma = load_me[5]
        self.alpha = 3
        stablenum = 5
        self.maxdiff = load_me[6]
        self.x_target = np.load('x_target.npy')
        # number_stabledim = min(000, self.n)
        number_stabledim = min(1000, self.n)
        print('number stabledim', number_stabledim)
        if number_stabledim == self.n:
            self.kappa_vec = np.array([stablenum]*number_stabledim)
        else:
            self.kappa_vec = np.array([stablenum]*number_stabledim+[1]*(self.n-number_stabledim))

        print('kappavec', self.kappa_vec)
        if not np.all(self.kappa_vec == self.kappa_vec[0]):
            print('not every kappavec entry is the same. check grad_doublewell_batch!')

    def grad_pot(self, t, x):
        return self.grad_doublewell(t, x)
        # return grad_rastrigin(x_in)


    def double_well(self, x):
        if len(x.shape) == 1:
            return self.kappa_vec*np.sum(x**2 - 1)**2
        else:
            return self.kappa_vec*np.sum(x**2 - 1, axis = 1)**2

    def grad_doublewell(self, t, x):
        if len(x.shape) == 1:
            print('grad_doublewell not tested')
            return 4*self.kappa_vec*(x**2-1)*x
        else:
            return self.kappa_vec[0]*4*(x**2 -1)*x


    def step(self, t, x, u, check_criterion=None, noise=None):
        return self.step_euler_mayurama(t, x, u, check_criterion, noise)
    
    def step_euler_mayurama(self, t, x, u, check_criterion, noise=None):
        if len(x.shape) == 1:
            if self.check_criterion(x):
                return x
            else:
                if noise is not None:
                    ret = x + self.tau*self.rhs_curr(t, x, u)  + self.sigma*noise
                else:
                    ret = x + self.tau*self.rhs_curr(t, x, u) + self.sigma*np.random.normal(loc=0.0,scale=self.sqrttau)
                ret = np.minimum(ret, self.interval_half)
                ret = np.maximum(ret, -self.interval_half)
                return ret
        else:
            if noise is not None:
                ret = x+(self.tau*self.rhs_curr(t, x, u)  + self.sigma*noise)*self.check_criterion(x)
            else:
                ret = x + (self.tau*self.rhs_curr(t, x, u) +  self.sigma*np.random.normal(loc=0.0,scale=self.sqrttau,size=x.shape))*self.check_criterion(x)
            ret[ret > self.interval_half] = self.interval_half
            ret[ret < -self.interval_half] = -self.interval_half
            return ret
        


    def step_rk4(self, t, x, u, rhs):
        k1 = self.tau * rhs(t, x, u)
        k2 = self.tau * rhs(t+self.tau/2, x + k1/2, u)
        k3 = self.tau * rhs(t+self.tau/2, x + k2/2, u)
        k4 = self.tau * rhs(t+self.tau, x + k3, u)
        return x + 1/6*(k1 + 2*k2 + 2*k3 + k4)


    def rhs_curr(self, t, x, u):
        return self.check_criterion(x)*(self.f(t, x) + self.g(t, x) * u)


    def f(self, t, x):
        return -self.grad_pot(t, x)
        # return np.ones(x.shape)


    def g(self, t, x):
        return 1


    def solver(self, t_points, x, calc_u_fun):
        print('does not work for SDE')
        # xshape = x.shape
        # num_entries = x.size
        # xreshaped = x.reshape(num_entries)
        # def rhs_ode(t,x):
        #     xorig_shape = x.reshape(xshape)
        #     ret =  (self.f(t, xorig_shape) + self.g(t, xorig_shape) @ calc_u_fun(t, xorig_shape))
        #     return ret.reshape(num_entries)
        # y = integrate.solve_ivp(rhs_ode, [t_points[0], t_points[-1]], xreshaped, t_eval=t_points)
        # y_mat = y.y.reshape(xshape+(len(t_points),))
        # return y_mat
        return 0


    def q(self, t, x):
        if len(x.shape) == 1:
            return 1
        else:
            return np.ones(x.shape[1])


    def r(self, t, u):
        if len(u.shape) == 1:
            return 1/2*la.norm(u)**2
        else:
            return 1/2*np.linalg.norm(u, axis = 0)**2


    def calc_reward(self, t, x, u):
        return self.tau*self.check_criterion(x)*(self.q(t, x) + self.r(t, u))

    
    def calc_u(self, t, x, grad):
        if len(x.shape) == 1:
            if check_criterion(x):
                return 0
            else:
                return -self.g(t, x) * grad
        else:
            # return np.sqrt(2)*np.ones(grad.shape)*self.check_criterion(x)
            return -self.g(t, x) *grad*self.check_criterion(x)


    def check_criterion(self, x):
        if len(x.shape) == 1:
            if np.linalg.norm(x - self.x_target) < self.maxdiff:
                return True
            else:
                return False
        else:
            ret = np.linalg.norm(x.T - self.x_target, axis=1)
            ret = (ret > self.maxdiff).astype(int)
            return ret


    def sample_boundary(self, no_dif_samples, samples_dim):
        samples_mat = np.zeros(shape=(samples_dim, no_dif_samples))
        for i0 in range(no_dif_samples):
            sample = np.random.uniform(-1, 1, samples_dim)
            sample /= np.linalg.norm(sample)/self.maxdiff
            samples_mat[:, i0] = self.x_target - sample
        return samples_mat


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
