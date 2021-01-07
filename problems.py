import numpy as np
import torch as pt

from numpy import exp, log
from scipy.linalg import expm


class LLGC():
    '''
        Ornstein-Uhlenbeck with linear terminal costs.
    '''
    def __init__(self, name='LLGC', d=1, off_diag=0, T=1, seed=42, modus='np'):

        np.random.seed(seed)
        self.modus = modus
        self.name = name
        self.d = d
        self.T = T
        self.A = -np.eye(self.d) + off_diag * np.random.randn(self.d, self.d)
        self.A_pt = pt.tensor(self.A).float()
        self.B = np.eye(self.d) + off_diag * np.random.randn(self.d, self.d)
        self.B_pt = pt.tensor(self.B).float()
        self.alpha = np.ones([self.d, 1])
        self.alpha_pt = pt.tensor(self.alpha).float()
        self.X_0 = np.zeros(self.d)
        self.delta_t_v = 0.001

        if ~np.all(np.linalg.eigvals(self.A) < 0):
            print('not all EV of A are negative')

    def b(self, x):
        if self.modus == 'pt':
            return pt.mm(self.A_pt, x.t()).t()
        return self.A.dot(x.T).T

    def sigma(self, x):
        if self.modus == 'pt':
            return pt.tensor(self.B_pt)
        return self.B

    def h(self, t, x, y, z):
        if self.modus == 'pt':
            return -0.5 * pt.sum(z**2, dim=1)
        return -0.5 * np.sum(z**2, dim=1)

    def g(self, x):
        if self.modus == 'pt':
            return pt.mm(x, self.alpha_pt)[:, 0]
        return x.dot(self.alpha)[:, 0]

    def u_true(self, x, t):
        return -self.sigma(x).T.dot(expm(self.A.T * (self.T - t)).dot(
            self.alpha) * np.ones(x.shape).T)

    def v_true(self, x, t):
        N = int(np.floor((self.T - t) / self.delta_t_v)) + 1
        Sigma_n = np.zeros([self.d, self.d])
        for t_n in np.linspace(t, self.T, N):
            Sigma_n += (expm(self.A * (self.T - t_n))
                        .dot(self.sigma(np.zeros([self.d, self.d])))
                        .dot(self.sigma(np.zeros([self.d, self.d])).t())
                        .dot(expm(self.A.T * (self.T - t_n)))) * self.delta_t_v
        return ((expm(self.A * (self.T - t)).dot(x.t()).T).dot(self.alpha)
                - 0.5 * self.alpha.T.dot(Sigma_n.dot(self.alpha)))
    
    

class CosExp():
    def __init__(self, name='CosExp', d=1, T=1, seed=42, modus='np'):

        np.random.seed(seed)
        self.modus = modus
        self.name = name
        self.d = d
        self.T = T
        self.B = np.eye(self.d) / np.sqrt(self.d)
        self.B_pt = pt.tensor(self.B).float()
        self.alpha = np.ones([self.d, 1]) # not needed, can delete?
        self.alpha_pt = pt.tensor(self.alpha).float() # not needed, can delete?
        self.X_0 = np.zeros(self.d)
        self.delta_t_v = 0.001

    def b(self, x):
        if self.modus == 'pt':
            return pt.ones(x.shape) / (5 * self.d)
        return np.ones(x.shape) / (5 * self.d)

    def sigma(self, x):
        if self.modus == 'pt':
            return pt.tensor(self.B_pt)
        return self.B

    def h(self, t, x, y, z):
        if self.modus == 'pt':
            return ((pt.cos(pt.sum(x, 1)) + 0.2 * pt.sin(pt.sum(x, 1))) * pt.exp(pt.tensor(self.T - t) / 2) 
                     - 0.5 * (pt.sin(pt.sum(x, 1)) * pt.cos(pt.sum(x, 1)) * pt.exp(pt.tensor(self.T - t)))**2 
                     + 1 / (2 * self.d) * (y * pt.sum(z, 1))**2)
        return ((np.cos(np.sum(x, 1)) + 0.2 * np.sin(np.sum(x, 1))) * np.exp((self.T - t) / 2) 
                     - 0.5 * (np.sin(np.sum(x, 1)) * np.cos(np.sum(x, 1)) * np.exp(self.T - t))**2 
                     + 1 / (2 * self.d) * (y * np.sum(z, 1))**2)

    def g(self, x):
        if self.modus == 'pt':
            return pt.cos(pt.sum(x, 1))
        return np.cos(np.sum(x, 1))

    def u_true(self, x, t):
        return np.repeat((np.sin(np.sum(x, 1)) * np.exp((self.T - t) / 2) / np.sqrt(self.d))[:, np.newaxis], 2, 1)
    
    def v_true(self, x, t):
        return np.cos(np.sum(x, 1)) * np.exp((self.T - t) / 2)
    
    

class AllenKahn():
    def __init__(self, name='CosExp', d=1, T=1, seed=42, modus='np'):

        np.random.seed(seed)
        self.modus = modus
        self.name = name
        self.d = d
        self.T = T
        self.B = np.eye(self.d) * np.sqrt(2)
        self.B_pt = pt.tensor(self.B).float()
        self.alpha = np.ones([self.d, 1]) # not needed, can delete?
        self.alpha_pt = pt.tensor(self.alpha).float() # not needed, can delete?
        self.X_0 = np.zeros(self.d)
        self.delta_t_v = 0.001

    def b(self, x):
        if self.modus == 'pt':
            return 0
            # return pt.zeros(x.shape)
        return 0
        # return np.zeros(x.shape)

    def sigma(self, x):
        if self.modus == 'pt':
            return pt.tensor(self.B_pt)
        return self.B

    def h(self, t, x, y, z):
        if self.modus == 'pt':
            return y - y**3
        return y - y**3

    def g(self, x):
        if self.modus == 'pt':
            return 1/(2+2/5*pt.norm(x, 1)**2)
        return 1/(2+2/5*np.linalg.norm(x, axis=1)**2)

    def u_true(self, x, t):
        print('no reference solution known')
        return 0
    
    def v_true(self, x, t):
        print('no reference solution known')
        return 0


class UnboundedSin():
    def __init__(self, name='CosExp', d=1, T=1, seed=42, modus='np'):

        np.random.seed(seed)
        self.modus = modus
        self.name = name
        self.d = d
        self.T = T
        self.B = 1 / np.sqrt(self.d) * np.eye(self.d)
        self.B_pt = pt.tensor(self.B).float()
        self.alpha = np.ones([self.d, 1]) # not needed, can delete?
        self.alpha_pt = pt.tensor(self.alpha).float() # not needed, can delete?
        self.X_0 = 0.5 * np.ones(self.d)
        self.rescal = 0.5
        self.Sig = self.B[0, 0]
        self.Sig_pt = self.B_pt[0, 0]
        self.delta_t_v = 0.001

    def b(self, x):
        if self.modus == 'pt':
            return 0
            # return pt.zeros(x.shape)
        return 0
        # return np.zeros(x.shape)

    def sigma(self, x):
        if self.modus == 'pt':
            return pt.tensor(self.B_pt)
        return self.B

    def h(self, t, x, u, Du):
        if self.modus == 'pt':
            Ut = - pt.mean(pt.where(x < 0, pt.sin(x), x), 1)
            xSum = x @ pt.arange(1., self.d + 1.)
            cosU = pt.cos(xSum)
            UVal = -(self.T - t) * Ut + cosU
            DUVal = (self.T - t) * pt.mean(pt.where(x < 0, pt.cos(x), pt.ones(x.shape)), 1) - self.d * (self.d + 1.) / 2. * pt.sin(xSum)
            D2UVal = -cosU * self.d * (self.d + 1) * (2 * self.d + 1) / 6. - (self.T - t) * pt.mean(pt.where(x < 0,  pt.sin(x), pt.zeros(x.shape)), 1)
            ret =  - Ut - 0.5 * self.Sig_pt * self.Sig_pt * D2UVal - self.rescal * (UVal * DUVal / self.d + UVal * UVal) + self.rescal * (u**2 + 1 / pt.sqrt(pt.tensor([self.d]).float()) * u.squeeze() * pt.sum(Du, 1))
        else:
            # u time derivative
            Ut = - np.mean(np.where(x < 0, np.sin(x), x), axis=-1)
            # u value
            xSum= x @ np.arange(1., self.d + 1.)
            cosU= np.cos(xSum)
            UVal = -(self.T - t) * Ut + cosU
            # U X derivarive (sum)
            DUVal = (self.T - t) * np.mean(np.where(x < 0, np.cos(x), np.ones(np.shape(x))), axis=-1) - self.d * (self.d + 1.) / 2. * np.sin(xSum)
            # sum of diag of Hessian
            D2UVal = -cosU * self.d * (self.d + 1) * (2 * self.d + 1) / 6. - (self.T - t) * np.mean(np.where(x < 0,  np.sin(x), np.zeros(np.shape(x))), axis=-1)
            # ret =  - Ut- 0.5*self.Sig*self.Sig*D2UVal - self.rescal*(UVal*DUVal/self.d + UVal*UVal)+ self.rescal*( np.power(u,2.) + np.multiply(u,np.mean(Du,axis=-1)))
            ret =  - Ut - 0.5 * self.Sig * self.Sig * D2UVal - self.rescal * (UVal * DUVal / self.d + UVal * UVal) + self.rescal * ( np.power(u, 2.) + 1 / np.sqrt(self.d) * np.multiply(u, np.sum(Du, axis=-1)))
        return ret.squeeze()

    def g(self, x):
        a = 1.0 * np.arange(1, self.d + 1)
        if self.modus == 'pt':
            a = pt.tensor(a).float()
            return pt.cos(pt.sum(x * a.unsqueeze(0), 1))
        return np.cos(x @ a)

    def u_true(self, x, t):
        print('no reference solution known')
        return 0
    
    def v_true(self, x, t):
        a = 1.0 * np.arange(1, self.d + 1)        
        return (self.T - t) * np.mean(np.where(x < 0,  np.sin(x), x), axis=-1) + np.cos(x @ a)
