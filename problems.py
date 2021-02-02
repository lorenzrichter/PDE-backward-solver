#pylint: disable=invalid-name, too-many-instance-attributes, too-many-arguments

import numpy as np
import torch as pt

from numpy import exp, log
from scipy.linalg import expm, solve_banded
from scipy.linalg import solve_continuous_are


device = 'cpu'  # pt.device('cuda' if pt.cuda.is_available() else 'cpu')


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
        self.sigma_modus = 'constant'

        if ~np.all(np.linalg.eigvals(self.A) < 0):
            print('not all EV of A are negative')

    def b(self, x):
        if self.modus == 'pt':
            return pt.mm(self.A_pt, x.t()).t()
        return self.A.dot(x.T).T

    def sigma(self, x):
        if self.modus == 'pt':
            return self.B_pt
        return self.B

    def h(self, t, x, y, z):
        if self.modus == 'pt':
            return -0.5 * pt.sum(z**2, dim=1)
        BTx = B.T @ x.T
        lx = x.T @ self.Q
        return

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
        self.sigma_modus = 'constant'

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
    def __init__(self, name='CosExp', d=1, T=0.3, seed=42, modus='np'):

        np.random.seed(seed)
        self.modus = modus
        self.name = name
        self.d = d
        self.T = T
        self.B = np.eye(self.d) * np.sqrt(2)
        self.B_pt = pt.tensor(self.B).float().to(device)
        self.alpha = np.ones([self.d, 1]) # not needed, can delete?
        self.alpha_pt = pt.tensor(self.alpha).float().to(device) # not needed, can delete?
        self.X_0 = np.zeros(self.d)
        self.sigma_modus = 'constant'

    def b(self, x):
        if self.modus == 'pt':
            return pt.zeros(x.shape).to(device)
        # return 0
        return np.zeros(x.shape)

    def sigma(self, x):
        if self.modus == 'pt':
            return self.B_pt
        return self.B

    def h(self, t, x, y, z):
        return y - y**3

    def g(self, x):
        if self.modus == 'pt':
            return 1 / (2 + 2 / 5 * pt.sum(x**2, 1))
        return 1 / (2 + 2 / 5 * np.linalg.norm(x, axis=1)**2)

    def u_true(self, x, t):
        print('no reference solution known')
        return 0

    def v_true(self, x, t):
        print('no reference solution known')
        return 0


    # t \in R                   is current time
    # x \in samples x d         is sample matrix
    # v \in samples             is value function evaluated at t, x
    # vt \in samples            is time derivative of v at t, x
    # vx \in samples x d        is gradient of v w.r.t x at t, x
    # vxx \in samples x d x d   is hessian of v w.r.t. x at t, x
    # returns: PDE_loss at every sample point
    # returns is a vector \in samples
    def pde_loss(self, t, x, v, vt, vx, vxx):
        assert x.shape[0] == v.shape[0] == vt.shape[0] == vx.shape[0] == vxx.shape[0]
        assert len(v.shape) == 1
        assert x.shape == vx.shape
        assert len(vxx.shape) == 3
        sigma = self.sigma(x)
        assert sigma.shape == (self.d, self.d)
        sigmaTsigma = sigma.T @ sigma
        # sigmaTsigma = np.einsum('ij,ik->ijk', sigma[:, :, 0], sigma[:, :, 0])
        loss = vt + np.einsum('il,il->i', self.b(x), vx) + 1 / 2 * (np.sum(sigmaTsigma[None, :, :] * vxx, axis = (1, 2))) + self.h(t, x, v, (sigma.T @ vx.T).T)
        return loss


class UnboundedSin():
    def __init__(self, name='UnboundedSin', d=1, T=1, seed=42, modus='np'):

        np.random.seed(seed)
        self.modus = modus
        self.name = name
        self.d = d
        self.T = T
        self.B = 1 / np.sqrt(self.d) * np.eye(self.d)
        self.B_pt = pt.tensor(self.B).float().to(device)
        self.X_0 = 0.5 * np.ones(self.d)
        self.rescal = 0.5
        self.Sig = self.B[0, 0]
        self.Sig_pt = self.B_pt[0, 0]
        self.sigma_modus = 'constant'

    def b(self, x):
        if self.modus == 'pt':
            return pt.zeros(x.shape).to(device)
        # return 0
        return np.zeros(x.shape)

    def sigma(self, x):
        if self.modus == 'pt':
            return self.B_pt
        return self.B

    def h(self, t, x, u, Du):
        if self.modus == 'pt':
            Ut = - pt.mean(pt.where(x < 0, pt.sin(x), x), 1)
            xSum = x @ pt.arange(1., self.d + 1.).to(device)
            cosU = pt.cos(xSum)
            UVal = -(self.T - t) * Ut + cosU
            DUVal = (self.T - t) * pt.mean(pt.where(x < 0, pt.cos(x), pt.ones(x.shape).to(device)), 1) - self.d * (self.d + 1.) / 2. * pt.sin(xSum)
            D2UVal = -cosU * self.d * (self.d + 1) * (2 * self.d + 1) / 6. - (self.T - t) * pt.mean(pt.where(x < 0,  pt.sin(x), pt.zeros(x.shape).to(device)), 1)
            ret = - Ut - 0.5 * self.Sig_pt * self.Sig_pt * D2UVal - self.rescal * (UVal * DUVal / self.d + UVal * UVal) + self.rescal * (u**2 + 1 / pt.sqrt(pt.tensor([self.d]).float().to(device)) * u.squeeze() * pt.sum(Du, 1))
        else:
            # u time derivative
            Ut = - np.mean(np.where(x < 0,  np.sin(x), x), axis=-1)
            # u value
            xSum = x @ np.arange(1., self.d + 1.)
            cosU = np.cos(xSum)
            UVal = -(self.T - t) * Ut + cosU
            # U X derivarive (sum)
            DUVal = (self.T - t) * np.mean(np.where(x < 0, np.cos(x), np.ones(np.shape(x))), axis=-1) - self.d * (self.d + 1.) / 2. * np.sin(xSum)
            # sum of diag of Hessian
            D2UVal = -cosU * self.d * (self.d + 1) * (2 * self.d + 1) / 6. - (self.T - t) * np.mean(np.where(x < 0, np.sin(x), np.zeros(np.shape(x))), axis=-1)

            # ret =  - Ut- 0.5*self.Sig*self.Sig*D2UVal - self.rescal*(UVal*DUVal/self.d + UVal*UVal)+ self.rescal*( np.power(u,2.) + np.multiply(u,np.mean(Du,axis=-1)))
            ret = - Ut - 0.5 * self.Sig * self.Sig * D2UVal - self.rescal * (UVal * DUVal / self.d + UVal * UVal)+ self.rescal * (np.power(u, 2.) + 1 / np.sqrt(self.d) * np.multiply(u, np.sum(Du, axis=-1)))
        return  ret.squeeze()

    def g(self, x):
        if self.modus == 'pt':
            a = pt.arange(1, self.d + 1).float().unsqueeze(1).to(device)
            return pt.cos(pt.mm(x, a).squeeze(1))
        a = 1.0 * np.arange(1, self.d + 1)
        return np.cos(x @ a)

    def u_true(self, x, t):
        print('no reference solution known')
        return 0

    def v_true(self, x, t):
        a = 1.0 * np.arange(1, self.d + 1)
        return (self.T - t) * np.mean(np.where(x < 0, np.sin(x), x), axis=-1) + np.cos(x @ a)


    # t \in R                   is current time
    # x \in samples x d         is sample matrix
    # v \in samples             is value function evaluated at t, x
    # vt \in samples            is time derivative of v at t, x
    # vx \in samples x d        is gradient of v w.r.t x at t, x
    # vxx \in samples x d x d   is hessian of v w.r.t. x at t, x
    # returns: PDE_loss at every sample point
    # returns is a vector \in samples
    def pde_loss(self, t, x, v, vt, vx, vxx):
        assert x.shape[0] == v.shape[0] == vt.shape[0] == vx.shape[0] == vxx.shape[0]
        assert len(v.shape) == 1
        assert x.shape == vx.shape
        assert len(vxx.shape) == 3
        sigma = self.sigma(x)
        assert sigma.shape == (self.d, self.d)
        sigmaTsigma = sigma.T @ sigma
        loss = vt + np.einsum('il,il->i', self.b(x), vx) + 1 / 2 * (np.sum(sigmaTsigma[None, :, :] * vxx, axis = (1, 2))) + self.h(t, x, v, (sigma.T @ vx.T).T)
        return loss


class Schloegl_SPDE():
    def __init__(self, name='CosExp', d=1, T=1, seed=42, modus='np'):
        a, b = -1, 1 # interval of the PDE
        s = np.linspace(a, b, d) # gridpoints
        nu = 1 # diffusion constant
        boundary = 'Neumann' # use 'Neumann' or "Dirichlet
        lambd = .1
        if boundary == 'Dirichlet':
            print('Dirichlet boundary')
            h = (b - a) / (d + 1)
            A = -2 * np.diag(np.ones(d), 0) + np.diag(np.ones(d - 1), 1) + np.diag(np.ones(d - 1), -1)
            A = nu / h**2 * A
            Q = h * np.eye(d)
        elif boundary== 'Neumann':
            print('Neumann boundary')
            h = (b - a) / (d - 1)             # step size in space
            A = -2 * np.diag(np.ones(d), 0) + np.diag(np.ones(d - 1), 1) + np.diag(np.ones(d - 1), -1)
            A[0, 1] = 2; A[d - 1, d - 2] = 2
            A = nu / h**2 * A
            Q = h * np.eye(d)
            Q[0, 0] /=2; Q[d - 1,d - 1] /= 2  # for neumann boundary
        else:
            print('Wrong boundary!')
        # _B = (np.bitwise_and(s > -0.4, s < 0.4))*1.0
        # B = np.zeros(shape=(d, 1))   
        # B[:, 0] = _B
        # A = -np.eye(d)
        B = 0.25 * np.eye(d)
        control_dim = B.shape[1]
        R = lambd * np.identity(control_dim)
        self.R_inv = np.linalg.inv(R)
        self.Pi = solve_continuous_are(A, B, Q, R)
        print('A', A, 'B', B, 'R', R, 'Q', Q)

        np.random.seed(seed)
        self.modus = modus
        self.name = name
        self.d = d
        self.T = T

        self.A = A
        self.A_pt = pt.tensor(self.A).float()
        self.B = B
        self.B_pt = pt.tensor(self.B).float()
        self.Q = Q
        self.Q_pt = pt.tensor(self.Q).float()
        self.R = R
        self.R_pt = pt.tensor(self.R).float()
        self.alpha = np.ones([self.d, 1]) # not needed, can delete?
        self.alpha_pt = pt.tensor(self.alpha).float() # not needed, can delete?
        self.X_0 = 1.2*np.ones(self.d)
        self.sigma_modus = 'constant'

    def calc_u(self, t, x, grad):
        if len(x.shape) == 1:
            return (-self.R_inv @ np.dot(grad, self.B) / 2).T
        else:
            u_mat = np.tensordot(grad, self.B, axes=((0), (0)))
            return (-self.R_inv @ u_mat.T / 2).T

    def calc_grad_fixed_control(self, t, x):
        return 2*self.Pi @ x.T

    def b(self, x, t=0):
        if self.modus == 'pt':
            return pt.mm(self.A_pt, x.t()).t()
        ret = self.A.dot(x.T).T + self.NL(t, x) + (self.B @ self.calc_u(t, x, self.calc_grad_fixed_control(t, x)).T).T
        return ret

    def NL(self, t, x):
        return x**3

    def b_variable_u(self, x, u, t=0):
        if self.modus == 'pt':
            return pt.mm(self.A_pt, x.t()).t()
        ret = self.A.dot(x.T).T + (self.B @ u.T).T
        return ret

    def sigma(self, x):
        if self.modus == 'pt':
            return pt.tensor(self.B_pt)
        return self.B

    def h(self, t, x, u, Du):
        xtqx = np.einsum('li,ik,lk->l', x, self.Q, x)
        Btgradv = self.B.T @ Du.T
        baru = self.calc_u(t, x.T, self.calc_grad_fixed_control(t, x))
        baruRbaru = np.einsum('li,ik,lk->l', baru, self.R, baru)
        BtgradvRinvBtgradv = np.einsum('il,ik,kl->l', Btgradv, self.R_inv, Btgradv)
        baruBtgradv = np.einsum('li,il->l', baru, Btgradv)
        # return -(xtqx - 1/4*BtgradvRinvBtgradv - baruBtgradv + baruRbaru)
        # print('np.linalg.norm(xtqx)', np.linalg.norm(xtqx), 'np.linalg.norm(baruRbaru)', np.linalg.norm(baruRbaru))
        return -xtqx - baruRbaru

    def g(self, x):
        if self.modus == 'pt':
            return 0
        if len(x.shape) == 1:
            return x.T @ self.Q @ x
        else:
            return np.einsum('li,ik,lk->l', x, self.Q, x)
        return 

    def u_true(self, x, t):
        print('no reference solution known')
        return 0

    def v_true(self, x, t):
        a = 1.0 * np.arange(1, self.d + 1)        
        return (self.T - t) * np.mean(np.where(x < 0, np.sin(x), x), axis=-1) + np.cos(x @ a)

    # t \in R                   is current time
    # x \in samples x d         is sample matrix
    # v \in samples             is value function evaluated at t, x
    # vt \in samples            is time derivative of v at t, x
    # vx \in samples x d        is gradient of v w.r.t x at t, x
    # vxx \in samples x d x d   is hessian of v w.r.t. x at t, x
    # returns: PDE_loss at every sample point
    # returns is a vector \in samples
    def pde_loss(self, t, x, v, vt, vx, vxx):
        assert x.shape[0] == v.shape[0] == vt.shape[0] == vx.shape[0] == vxx.shape[0]
        assert len(v.shape) == 1
        assert x.shape == vx.shape
        assert len(vxx.shape) == 3
        sigma = self.sigma(x)
        assert sigma.shape == (self.d, self.d)
        sigmaTsigma = sigma.T @ sigma
        # sigmaTsigma = np.einsum('ij,ik->ijk', sigma[:, :, 0], sigma[:, :, 0])
        loss = vt + np.einsum('il,il->i', self.b(x), vx) + 1 / 2 * (np.sum(sigmaTsigma[None, :, :] * vxx, axis=(1,2))) + self.h(t, x, v, (sigma.T @ vx.T).T)
        return loss


class BondpriceMultidim():
    def __init__(self, name='bondprice_multidim', d=1, T=1, seed=42, modus='np'):

        np.random.seed(seed)
        self.A = np.random.uniform(size=d)
        self.A_pt = pt.tensor(self.A).float().to(device)
        self.B = np.random.uniform(size=d)
        self.B_pt = pt.tensor(self.B).float().to(device)
        self.S_vec = np.random.uniform(size=d)
        self.S_vec_pt = pt.tensor(self.S_vec).float().to(device)
        self.S = np.zeros((d, d))
        self.S[:, 0] = self.S_vec
        self.S_pt = pt.tensor(self.S).float().to(device)
        self.modus = modus
        self.name = name
        self.d = d
        self.T = T
        self.X_0 = np.ones(self.d)
        self.sigma_modus = 'variable'

    def b(self, x):
        if self.modus == 'pt':
            return (self.A_pt * (self.B_pt - x)).reshape(-1, self.d)
        return (self.A * (self.B - x)).reshape(-1, self.d)


    def sigma(self, x):

        if self.modus == 'pt':
            sqrtx = pt.sqrt(pt.abs(x))
            ret = pt.zeros((x.shape[0], self.S_vec.size, self.S_vec.size)).to(device)
            # Sdotsqrtx = np.einsum('i,ji->ji', self.S_vec, sqrtx)
            Sdotsqrtx = self.S_vec_pt * sqrtx
            ret[:, :, 0] = Sdotsqrtx
            return ret

        sqrtx = np.sqrt(np.abs(x))
        if len(x.shape) == 2:
            ret = np.zeros((x.shape[0], self.S_vec.size, self.S_vec.size))
            # Sdotsqrtx = np.einsum('i,ji->ji', self.S_vec, sqrtx)
            Sdotsqrtx = self.S_vec * sqrtx
            ret[:, :, 0] = Sdotsqrtx
        else:
            ret = np.zeros((self.S_vec.size, self.S_vec.size))
            Sdotsqrtx = np.einsum('i,i->i', self.S_vec, sqrtx)
            ret = ret[:, 0] = Sdotsqrtx

        # ret =  (self.S @ np.sqrt(np.abs(x).T)).T
        return ret
        # return (self.S@np.sqrt(np.abs(x))).reshape(-1, self.d)


    def h(self, t, x, u, Du):
        if self.modus == 'pt':
            return -u * pt.max(x, 1)[0]
        return -u * np.max(x, 1)

    def g(self, x):
        if self.modus == 'pt':
            return pt.ones((x.shape[0])).to(device)
        return np.ones((x.shape[0])).squeeze()


    def u_true(self, x, t):
        print('no reference solution known')
        return 0
    
    def v_true(self, x, t):
        print('no reference solution known')
        return 0

    # t \in R                   is current time
    # x \in samples x d         is sample matrix
    # v \in samples             is value function evaluated at t, x
    # vt \in samples            is time derivative of v at t, x
    # vx \in samples x d        is gradient of v w.r.t x at t, x
    # vxx \in samples x d x d   is hessian of v w.r.t. x at t, x
    # returns: PDE_loss at every sample point
    # returns is a vector \in samples
    def pde_loss(self, t, x, v, vt, vx, vxx):
        assert x.shape[0] == v.shape[0] == vt.shape[0] == vx.shape[0] == vxx.shape[0]
        assert len(v.shape) == 1
        assert x.shape == vx.shape
        assert len(vxx.shape) == 3
        sigma = self.sigma(x)
        assert sigma.shape == (x.shape[0], self.d, self.d)
        sigmaTsigma = np.einsum('ij,ik->ijk', sigma[:, :, 0], sigma[:, :, 0])
        loss = vt + np.einsum('il,il->i', self.b(x), vx) + 1 / 2 * (np.sum(sigmaTsigma * vxx, axis = (1, 2))) + self. h(t, x, v, vx)
        return loss


class HJB():
    def __init__(self, name='HJB', d=1, T=1, seed=42, modus='np'):

        np.random.seed(seed)
        self.modus = modus
        self.name = name
        self.d = d
        self.T = T
        self.B = np.sqrt(2.0) * np.eye(self.d)
        self.B_pt = pt.tensor(self.B).float().to(device)
        self.X_0 = np.zeros(self.d)
        self.sigma_modus = 'constant'

    def b(self, x):
        if self.modus == 'pt':
            return  0 * pt.ones(x.shape).to(device) # pt.zeros(x.shape).to(device) # 
        return 0 * np.ones(x.shape) # np.zeros(x.shape) # 

    def sigma(self, x):
        if self.modus == 'pt':
            return self.B_pt
        return self.B

    def h(self, t, x, y, z):
        if self.modus == 'pt':
            return -0.5 * pt.sum(z**2, 1)
        else:
            return -0.5 * np.sum(z**2, 1)

    def g(self, x):
        if self.modus == 'pt':
            return pt.log(0.5 + 0.5 * pt.sum(x**2, 1))
        return log(0.5 + 0.5 * np.sum(x**2, 1))

    def u_true(self, x, t):
        print('no reference solution known')
        return 0

    def v_true(self, x, t):
        print('no reference solution known')
        return 0


    # t \in R                   is current time
    # x \in samples x d         is sample matrix
    # v \in samples             is value function evaluated at t, x
    # vt \in samples            is time derivative of v at t, x
    # vx \in samples x d        is gradient of v w.r.t x at t, x
    # vxx \in samples x d x d   is hessian of v w.r.t. x at t, x
    # returns: PDE_loss at every sample point
    # returns is a vector \in samples
    def pde_loss(self, t, x, v, vt, vx, vxx):
        assert x.shape[0] == v.shape[0] == vt.shape[0] == vx.shape[0] == vxx.shape[0]
        assert len(v.shape) == 1
        assert x.shape == vx.shape
        assert len(vxx.shape) == 3
        sigma = self.sigma(x)
        assert sigma.shape == (self.d, self.d)
        sigmaTsigma = sigma.T @ sigma
        loss = vt + np.einsum('il,il->i', self.b(x), vx) + 1 / 2 * (np.sum(sigmaTsigma[None, :, :] * vxx, axis = (1, 2))) + self.h(t, x, v, (sigma.T @ vx.T).T)
        return loss


class DoubleWell():
    '''
        Multidimensional double well potential
    '''
    def __init__(self, name='Double well', d=1, d_1=1, d_2=0, T=1, eta=1, kappa=1, modus='np'):
        self.name = name
        self.d = d
        self.d_1 = d_1
        self.d_2 = d_2
        self.T = T
        self.eta = eta
        self.eta_ = np.array([eta] * d_1 + [1.0] * d_2)
        self.eta_pt = pt.tensor(self.eta_).to(device).float()
        self.kappa = kappa
        self.kappa_ = np.array([kappa] * d_1 + [1.0] * d_2)
        self.kappa_pt = pt.tensor(self.kappa_).to(device).float()
        self.B = np.eye(self.d)
        self.B_pt = pt.tensor(self.B).to(device).float()
        self.X_0 = -np.ones(self.d)
        self.ref_sol_is_defined = False
        self.sigma_modus = 'constant'
        self.modus = modus

    def V(self, x):
        return self.kappa * (x**2 - 1)**2

    def V_2(self, x):
        return (x**2 - 1)**2

    def grad_V(self, x):
        if self.modus == 'pt':
            return 4.0 * self.kappa_pt * (x * (x**2 - pt.ones(self.d).to(device)))
        return 4.0 * self.kappa_ * (x * (x**2 - np.ones(self.d)))

    def b(self, x):
        return -self.grad_V(x)

    def sigma(self, x):
        if self.modus == 'pt':
            return self.B_pt
        return self.B # self.B.repeat(x.shape[0], 1, 1)

    def h(self, t, x, y, z):
        if self.modus == 'pt':
            return -0.5 * pt.sum(z**2, dim=1)
        return -0.5 * np.sum(z**2, dim=1)

    def g_1(self, x_1):
        if self.modus == 'pt':
            return self.eta_pt * (x_1 - 1)**2
        return self.eta * (x_1 - 1)**2

    def g_2(self, x_1):
        return (x_1 - 1)**2

    def g(self, x):
        if self.modus == 'pt':
            return ((pt.sum(self.eta_pt * (x - pt.ones(self.d).to(device))**2, 1)))
        return ((np.sum(self.eta_ * (x - np.ones(self.d))**2, 1)))

    def compute_reference_solution(self, delta_t=0.005, xb=2.5, nx=1000):

        self.xb = xb # range of x, [-xb, xb]
        self.nx = nx # number of discrete interval
        self.dx = 2.0 * self.xb / self.nx
        self.delta_t = delta_t

        beta = 2

        self.xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=True)

        # A = D^{-1} L D
        # assumes Neumann boundary conditions

        A = np.zeros([self.nx, self.nx])
        for i in range(0, self.nx):

            x = -self.xb + (i + 0.5) * self.dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * self.dx
                x1 = -self.xb + i * self.dx
                A[i, i - 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = exp(beta * (self.V(x) - self.V(x1))) / self.dx**2
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * self.dx
                x1 = -self.xb + (i + 1) * self.dx
                A[i, i + 1] = -exp(beta * 0.5 * (self.V(x0) + self.V(x) - 2 * self.V(x1))) / self.dx**2
                A[i, i] = A[i, i] + exp(beta * (self.V(x) - self.V(x1))) / self.dx**2

        A = -A / beta
        N = int(self.T / self.delta_t)

        D = np.diag(exp(beta * self.V(self.xvec) / 2))
        D_inv = np.diag(exp(-beta * self.V(self.xvec) / 2))

        np.linalg.cond(np.eye(self.nx) - self.delta_t * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.delta_t * A)

        self.psi = np.zeros([N + 1, self.nx])
        self.psi[N, :] = exp(-self.g_1(self.xvec))

        for n in range(N - 1, -1, -1):
            band = - self.delta_t * np.vstack([np.append([0], np.diagonal(A, offset=1)),
                                               np.diagonal(A, offset=0) - N / self.T,
                                               np.append(np.diagonal(A, offset=1), [0])])

            self.psi[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi[n + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - delta_t * A, D_inv.dot(psi[n + 1, :])));


        self.u = np.zeros([N + 1, self.nx - 1])
        for n in range(N + 1):
            for i in range(self.nx - 1):
                self.u[n, i] = -2 / beta * self.B[0, 0] * (- log(self.psi[n, i + 1]) + log(self.psi[n, i])) / self.dx
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)

    def v_true_1(self, x, t):
        i = np.floor((x.squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(- log(self.psi[n, i])).reshape([1, len(i)])

    def u_true_1(self, x, t):
        x = x.unsqueeze(1)
        x = x.t()
        i = np.floor((np.clip(x, -self.xb, self.xb - 2 * self.dx).squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(self.u[n, i]).reshape([1, len(i)])
        #return interpolate.interp1d(self.xvec, self.u)(x)[:, n]
        
    def compute_reference_solution_2(self, delta_t=0.005, xb=2.5, nx=1000):

        self.xb = xb # range of x, [-xb, xb]
        self.nx = nx # number of discrete interval
        self.dx = 2.0 * self.xb / self.nx
        self.delta_t = delta_t

        beta = 2

        self.xvec = np.linspace(-self.xb, self.xb, self.nx, endpoint=True)

        # A = D^{-1} L D
        # assumes Neumann boundary conditions

        A = np.zeros([self.nx, self.nx])
        for i in range(0, self.nx):

            x = -self.xb + (i + 0.5) * self.dx
            if i > 0:
                x0 = -self.xb + (i - 0.5) * self.dx
                x1 = -self.xb + i * self.dx
                A[i, i - 1] = -exp(beta * 0.5 * (self.V_2(x0) + self.V_2(x) - 2 * self.V_2(x1))) / self.dx**2
                A[i, i] = exp(beta * (self.V_2(x) - self.V_2(x1))) / self.dx**2
            if i < self.nx - 1:
                x0 = -self.xb + (i + 1.5) * self.dx
                x1 = -self.xb + (i + 1) * self.dx
                A[i, i + 1] = -exp(beta * 0.5 * (self.V_2(x0) + self.V_2(x) - 2 * self.V_2(x1))) / self.dx**2
                A[i, i] = A[i, i] + exp(beta * (self.V_2(x) - self.V_2(x1))) / self.dx**2

        A = -A / beta
        N = int(self.T / self.delta_t)

        D = np.diag(exp(beta * self.V_2(self.xvec) / 2))
        D_inv = np.diag(exp(-beta * self.V_2(self.xvec) / 2))

        np.linalg.cond(np.eye(self.nx) - self.delta_t * A)
        #w, vv = np.linalg.eigh(np.eye(self.nx) - self.delta_t * A)

        self.psi_2 = np.zeros([N + 1, self.nx])
        self.psi_2[N, :] = exp(-self.g_2(self.xvec))

        for n in range(N - 1, -1, -1):
            band = - self.delta_t * np.vstack([np.append([0], np.diagonal(A, offset=1)),
                                               np.diagonal(A, offset=0) - N / self.T,
                                               np.append(np.diagonal(A, offset=1), [0])])

            self.psi_2[n, :] = D.dot(solve_banded([1, 1], band, D_inv.dot(self.psi_2[n + 1, :])))
            #psi[n, :] = np.dot(D, np.linalg.solve(np.eye(self.nx) - delta_t * A, D_inv.dot(psi[n + 1, :])));


        self.u_2 = np.zeros([N + 1, self.nx - 1])
        for n in range(N + 1):
            for i in range(self.nx - 1):
                self.u_2[n, i] = -2 / beta * self.B[0, 0] * (- log(self.psi_2[n, i + 1]) + log(self.psi_2[n, i])) / self.dx
        #self.u = 2 / beta * np.gradient(np.log(self.psi), self.dx, 1)

    def u_true_2(self, x, t):
        x = x.unsqueeze(1)
        x = x.t()
        i = np.floor((np.clip(x, -self.xb, self.xb - 2 * self.dx).squeeze(0) + self.xb) / self.dx).long()
        i[-1] -= 2
        n = int(np.ceil(t / self.delta_t))
        return np.array(self.u_2[n, i]).reshape([1, len(i)])
        #return interpolate.interp1d(self.xvec, self.u)(x)[:, n]

    def v_true(self, x, t):
        return None

    def u_true(self, x, t):
        return np.concatenate([self.u_true_1(x[:, i], t).T for i in range(self.d_1)] + [self.u_true_2(x[:, i], t).T for i in range(self.d_1, self.d)], 1).T
