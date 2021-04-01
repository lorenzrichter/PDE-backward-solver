import xerus as xe
import numpy as np
from scipy import linalg as la
import pickle
import orth_pol
import matplotlib.pyplot as plt


def polynomials(pol_deg):
    def create_lambd_fun(index):
        if index == 0:
            return lambda x: 1
        elif index == 1:
            return lambda x: np.where(x < 0,  np.sin(x), x)
        elif index == 2:
            return lambda x: x
#         if index == 0:
#             return lambda x: np.sin(x)
#         elif index == 1:
#             return lambda x: np.cos(x)


    def create_dlambd_fun(index):
        if index == 0:
            return lambda x: 0
        elif index == 1:
            return lambda x: np.where(x < 0,  np.cos(x), 1)
        elif index == 2:
            return lambda x: 1
 #        if index == 0:
 #            return lambda x: np.cos(x)
 #        elif index == 1:
 #            return lambda x: -np.sin(x)
 #        # def _(x):
 #            # return pol[index](x)
        # return _
    def create_ddlambd_fun(index):
        if index == 0:
            return lambda x: 0
        elif index == 1:
            return lambda x: np.where(x < 0,  -np.sin(x), 0)
        elif index == 2:
            return lambda x: 0
 #        if index == 0:
 #            return lambda x: np.cos(x)
 #        elif index == 1:
 #            return lambda x: -np.sin(x)
 #        # def _(x):
 #            # return pol[index](x)
        # return _
    pol = []
    for i0 in range(pol_deg):
        pol.append(create_lambd_fun(i0))
    dpol = []
    for i0 in range(pol_deg):
        dpol.append(create_dlambd_fun(i0))
    ddpol = []
    for i0 in range(pol_deg):
        ddpol.append(create_ddlambd_fun(i0))
    return pol, dpol, ddpol
        

class Valuefunction_TT:
    def __init__(self, valuefunction_prename=None, load_existing_list=False, c_name=None, grad_c_name=None):
        self.t_vec_p = np.load('t_vec_p.npy')
        self.V = []
        if valuefunction_prename == None:
            V_load = pickle.load(open('V_new', 'rb'))
            for i0 in range(len(self.t_vec_p)):
                self.V.append(1*V_load)
            self.c_add_fun_list = np.zeros(len(self.V))
            self.c_add_fun_list[-1] = 1
            self.c_add_fun_list[-2] = 1
            # self.V= [V_load for i0 in range(len(self.t_vec_p))]
        else:
            if load_existing_list:
                for i0 in range(len(self.t_vec_p)):
                    self.V.append(pickle.load(open(valuefunction_prename+str(i0), 'rb')))
                self.c_add_fun_list = np.load(c_name)
            else:
                V_load = pickle.load(open(valuefunction_prename, 'rb'))
                for i0 in range(len(self.t_vec_p)):
                    self.V.append(1*V_load)
                self.c_add_fun_list = np.zeros(len(self.V))
                self.c_add_fun_list[-1] = 1
                self.c_add_fun_list[-2] = 1
                # self.V= [V_load for i0 in range(len(self.t_vec_p))]
        self.r = self.V[0].order()
        load_me = np.load('save_me.npy')
        self.tau = self.t_vec_p[1] - self.t_vec_p[0]
        self.integrate_min = load_me[2]
        self.integrate_max = load_me[3]
        self.pol_deg = np.max(self.V[0].dimensions)
        usepol = True
        # usepol = False
        self.ddpol = []
        if not usepol:
            self.pol, self.dpol, self.ddpol = polynomials(self.pol_deg)
        else:
            self.pol, self.dpol = orth_pol.calc_pol(self.integrate_max, self.integrate_min, self.pol_deg-1)
            for i0 in range(self.pol_deg):
                self.ddpol.append(np.polyder(self.dpol[i0]))
        print('self.pol_deg', self.pol_deg)
        self.add_fun = lambda t, x: np.zeros(x.shape[-1])
        self.grad_add_fun = lambda t, x: np.zeros(x.shape[-1])
        self.hess_add_fun = lambda t, x: np.zeros(x.shape[-1])
        self.laplacian_add_fun = lambda t, x: np.zeros(x.shape[-1])


    def set_add_fun(self, fun, dfun, ddfun=None, laplfun=None):
        self.add_fun = fun
        self.grad_add_fun = dfun
        if ddfun is not None:
            self.hess_add_fun = ddfun
        if laplfun is not None:
            self.laplacian_add_fun = laplfun


    def load_valuefun(self, valuefunction_prename):
        self.V = []
        for i0 in range(len(self.t_vec_p)):
            self.V.append(pickle.load(open(valuefunction_prename+str(i0), 'rb')))
        self.pol_deg = np.max(self.V[0].dimensions)


    def eval_V(self, t, x):
        if type(t) == float or type(t) == np.float64:
            # print('t is float')
            V = self.V[self.t_to_ind(t)]
            add_fun_const = self.c_add_fun_list[self.t_to_ind(t)]
        else:
            # print('t is not float. t is', type(t))
            V, add_fun_const = t
        if len(x.shape) == 1:
            ii, jj, kk = xe.indices(3)
            feat = self.P(x)
            temp = xe.Tensor([1])
            comp = xe.Tensor()
            temp[0] = 1
            for iter_1 in range(self.r):
                comp = V.get_component(iter_1)
                temp(kk) << temp(ii)*comp(ii, jj, kk)*xe.Tensor.from_buffer(feat[iter_1])(jj)
            return temp[0] + add_fun_const * self.add_fun(t, x)
        else:
            feat = self.P_batch(x)
            temp = np.ones(shape=(1, x.shape[1]))
            for iter_1 in range(x.shape[0]):
                comp = V.get_component(iter_1).to_ndarray()
                temp = np.einsum('il,ijk,jl->kl', temp, comp, feat[iter_1])
            return temp[0] + add_fun_const * self.add_fun(0, x)

    def calc_grad(self, t, x):
        if type(t) == float or type(t) == np.float64 or type(t) == int:
            # print('t is float')
            V = self.V[self.t_to_ind(t)]
            add_fun_const = self.c_add_fun_list[self.t_to_ind(t)]
        else:
            # print('t is not float. t is', type(t))
            V, add_fun_const = t
            V = t
        # print('V in calc_grad', xe.Tensor(V))
        # print('t, self.t_to_ind(t)', t, self.t_to_ind(t),'frob_norm(v)', xe.frob_norm(V))
        if len(x.shape) == 1:
            c1, c2, c3 = xe.indices(3)
            feat = self.P(x)
            dfeat = self.dP(x)
            dV = np.zeros(shape=self.r)
            temp = xe.Tensor([1])
            comp = xe.Tensor()
            temp_right = xe.Tensor.ones([1])
            temp_left = xe.Tensor.ones([1])
            list_right = [None]*(self.r)
            list_right[self.r-1] = xe.Tensor(temp_right)
            for iter_0 in range(self.r-1, 0, -1):
                comp = V.get_component(iter_0)
                temp_right(c1) << temp_right(c3) * comp(c1, c2, c3) * xe.Tensor.from_buffer(feat[iter_0])(c2)
    #            temp_right = xe.contract(comp, False, temp_right, False, 1)
    #            temp_right = xe.contract(temp_right, False, xe.Tensor.from_buffer(feat[iter_0]), False, 1)
                list_right[iter_0-1] = xe.Tensor(temp_right)
            for iter_0 in range(self.r):
                comp = V.get_component(iter_0)
                temp() << temp_left(c1) * comp(c1, c2, c3) * xe.Tensor.from_buffer(dfeat[iter_0])(c2) * list_right[iter_0](c3)
    #            temp = xe.contract(comp, False, list_right[iter_0], False, 1)
    #            temp = xe.contract(temp, False, xe.Tensor.from_buffer(dfeat[iter_0]), False, 1)
    #            temp = xe.contract(temp, False, temp_left, False, 1)
                temp_left(c3) << temp_left(c1) * comp(c1, c2, c3) * xe.Tensor.from_buffer(feat[iter_0])(c2)
    #            temp_left = xe.contract(temp_left, False, comp, False, 1)
    #            temp_left = xe.contract(xe.Tensor.from_buffer(feat[iter_0]), False, temp_left, False, 1)
    
                dV[iter_0] = temp[0]
            return dV + add_fun_const * self.grad_add_fun(t, x)
        else:
            nos = x.shape[1]
            feat = self.P_batch(x)
            dfeat = self.dP_batch(x)
            dV_mat = np.zeros(shape=x.shape)
            temp = np.zeros(1)
            temp_right = np.ones(shape = (1,nos))
            temp_left = np.ones(shape=(1,nos))
            list_right = [None]*(self.r)
            list_right[self.r-1] = temp_right
            for iter_0 in range(self.r-1, 0, -1):
                comp = V.get_component(iter_0).to_ndarray()
    #            temp_right(c1) << temp_right(c3) * comp(c1, c2, c3) * feat[iter_0](c2)
                list_right[iter_0-1] = np.einsum('kl,ijk,jl->il', list_right[iter_0], comp, feat[iter_0])
            for iter_0 in range(self.r):
                comp = V.get_component(iter_0).to_ndarray()
    #            temp() << temp_left(c1) * comp(c1, c2, c3) * dfeat[iter_0](c2) \
    #                * list_right[iter_0](c3)
                temp = np.einsum('il,ijk,jl,kl->l', temp_left, comp, dfeat[iter_0], list_right[iter_0])
    #            temp(c3) << temp_left(c1) * comp(c1, c2, c3) * feat[iter_0](c2)
                temp_left = np.einsum('il,ijk,jl->kl', temp_left, comp, feat[iter_0])
                dV_mat[iter_0,:] = temp
    #        _u = -gamma/lambd*np.dot(dV, B) - shift_TT
            return dV_mat + add_fun_const * self.grad_add_fun(t, x)




    def calc_laplace(self, t, x):
        if type(t) == float or type(t) == np.float64 or type(t) == int:
            # print('t is float')
            V = self.V[self.t_to_ind(t)]
            add_fun_const = self.c_add_fun_list[self.t_to_ind(t)]
        else:
            # print('t is not float. t is', type(t))
            V, add_fun_const = t
            V = t
        # print('V in calc_grad', xe.Tensor(V))
        # print('t, self.t_to_ind(t)', t, self.t_to_ind(t),'frob_norm(v)', xe.frob_norm(V))
        if len(x.shape) == 1:
            c1, c2, c3 = xe.indices(3)
            feat = self.P(x)
            dfeat = self.ddP(x)
            dV = 0
            temp = xe.Tensor([1])
            comp = xe.Tensor()
            temp_right = xe.Tensor.ones([1])
            temp_left = xe.Tensor.ones([1])
            list_right = [None]*(self.r)
            list_right[self.r-1] = xe.Tensor(temp_right)
            for iter_0 in range(self.r-1, 0, -1):
                comp = V.get_component(iter_0)
                temp_right(c1) << temp_right(c3) * comp(c1, c2, c3) * xe.Tensor.from_buffer(feat[iter_0])(c2)
    #            temp_right = xe.contract(comp, False, temp_right, False, 1)
    #            temp_right = xe.contract(temp_right, False, xe.Tensor.from_buffer(feat[iter_0]), False, 1)
                list_right[iter_0-1] = xe.Tensor(temp_right)
            for iter_0 in range(self.r):
                comp = V.get_component(iter_0)
                temp() << temp_left(c1) * comp(c1, c2, c3) * xe.Tensor.from_buffer(dfeat[iter_0])(c2) * list_right[iter_0](c3)
    #            temp = xe.contract(comp, False, list_right[iter_0], False, 1)
    #            temp = xe.contract(temp, False, xe.Tensor.from_buffer(dfeat[iter_0]), False, 1)
    #            temp = xe.contract(temp, False, temp_left, False, 1)
                temp_left(c3) << temp_left(c1) * comp(c1, c2, c3) * xe.Tensor.from_buffer(feat[iter_0])(c2)
    #            temp_left = xe.contract(temp_left, False, comp, False, 1)
    #            temp_left = xe.contract(xe.Tensor.from_buffer(feat[iter_0]), False, temp_left, False, 1)
    
                dV +=  temp[0]
            return dV + add_fun_const * self.laplacian_add_fun(t, x)
        else:
            nos = x.shape[1]
            feat = self.P_batch(x)
            dfeat = self.ddP_batch(x)
            dV_mat = np.zeros(shape=x.shape[-1])
            temp = np.zeros(1)
            temp_right = np.ones(shape = (1,nos))
            temp_left = np.ones(shape=(1,nos))
            list_right = [None]*(self.r)
            list_right[self.r-1] = temp_right
            for iter_0 in range(self.r-1, 0, -1):
                comp = V.get_component(iter_0).to_ndarray()
    #            temp_right(c1) << temp_right(c3) * comp(c1, c2, c3) * feat[iter_0](c2)
                list_right[iter_0-1] = np.einsum('kl,ijk,jl->il', list_right[iter_0], comp, feat[iter_0])
            for iter_0 in range(self.r):
                comp = V.get_component(iter_0).to_ndarray()
    #            temp() << temp_left(c1) * comp(c1, c2, c3) * dfeat[iter_0](c2) \
    #                * list_right[iter_0](c3)
                temp = np.einsum('il,ijk,jl,kl->l', temp_left, comp, dfeat[iter_0], list_right[iter_0])
    #            temp(c3) << temp_left(c1) * comp(c1, c2, c3) * feat[iter_0](c2)
                temp_left = np.einsum('il,ijk,jl->kl', temp_left, comp, feat[iter_0])
                dV_mat += temp
    #        _u = -gamma/lambd*np.dot(dV, B) - shift_TT
            return dV_mat + add_fun_const * self.laplacian_add_fun(t, x)



    def calc_hessian(self, t, x):
        if type(t) == float or type(t) == np.float64:
            # print('t is float')
            V = self.V[self.t_to_ind(t)]
            add_fun_const = self.c_add_fun_list[self.t_to_ind(t)]
        else:
            # print('t is not float. t is', type(t))
            V, add_fun_const = t
            V = t
        # print('V in calc_grad', xe.Tensor(V))
        # print('t, self.t_to_ind(t)', t, self.t_to_ind(t),'frob_norm(v)', xe.frob_norm(V))
        r = V.order()
        # print('x.shape', x.shape)
        if len(x.shape) == 1:
            feat = self.P(x)
            dfeat = self.dP(x)
            ddfeat = self.ddP(x)
            hess = np.zeros((r,r))
            for i0 in range(r):
                for i1 in range(i0):
                    curr_feat = 1*feat
                    curr_feat[i0] = dfeat[i0]
                    curr_feat[i1] = dfeat[i1]
                    contracted = self.contract_feat(V, curr_feat)
                    hess[i0, i1] = contracted
                    hess[i1, i0] = contracted
                curr_feat = 1*feat
                curr_feat[i0] = ddfeat[i0]
                hess[i0,i0] = self.contract_feat(V, curr_feat)
        else:
            feat = self.P_batch(x)
            dfeat = self.dP_batch(x)
            ddfeat = self.ddP_batch(x)
            hess = np.zeros((r,r,x.shape[1]))
            for i0 in range(r):
                for i1 in range(i0):
                    curr_feat = 1*feat
                    curr_feat[i0] = dfeat[i0]
                    curr_feat[i1] = dfeat[i1]
                    contracted = self.contract_feat_batch(V, curr_feat)
                    hess[i0, i1] = contracted
                    hess[i1, i0] = contracted
                curr_feat = 1*feat
                curr_feat[i0] = ddfeat[i0]
                hess[i0,i0,:] = self.contract_feat_batch(V, curr_feat)
        return hess + add_fun_const * self.hess_add_fun(t, x)



    def contract_feat(self, T, feat_list):
        # y = proj @ x
        ii, jj, kk = xe.indices(3)
        # x_T = xe.Tensor.from_ndarray(y)
        comp = xe.Tensor()
        temp = xe.Tensor([1])
        temp[0] = 1
        r = T.order()
        for iter_1 in range(r):
            comp = T.get_component(iter_1)
            temp(kk) << temp(ii)*comp(ii, jj, kk)*xe.Tensor.from_ndarray(feat_list[iter_1])(jj)
        return temp[0]


    def contract_feat_batch(self, T, feat_list):
        # y = proj @ x
        ii, jj, kk, ll = xe.indices(4)
        # x_T = xe.Tensor.from_ndarray(y)
        comp = xe.Tensor()
        temp = np.zeros([1, feat_list[0].shape[1]])
        temp[0,:] = 1
        r = T.order()
        for iter_1 in range(r):
            comp = T.get_component(iter_1).to_ndarray()
            # temp(kk, ll) << temp(ii, ll)*comp(ii, jj, kk)*xe.Tensor.from_ndarray(feat_list[iter_1])(jj,ll )
            temp = np.einsum('il,ijk,jl->kl', temp, comp, feat_list[iter_1])
        return temp[0,:]


    def P(self, x):
        ret = []
        ret = [np.zeros(shape=(self.pol_deg)) for _ in range(self.r)]
        for i0 in range(self.r):
            for i1 in range(self.pol_deg):
                ret[i0][i1] = self.pol[i1](x[i0])
        return ret
    
    
    def dP(self, x):
        ret = []
        ret = [np.zeros(shape=(self.pol_deg)) for _ in range(self.r)]
        for i0 in range(self.r):
            for i1 in range(1, self.pol_deg):
                ret[i0][i1] = self.dpol[i1](x[i0])
        return ret


    def ddP(_x):
        ret = []
        r = len(_x)
        ret = [np.zeros(shape=(self.pol_deg)) for _ in range(r)]
        for i0 in range(r):
            for i1 in range(1, self.pol_deg):
                ret[i0][i1] = self.ddpol[i1](_x[i0])
        return ret


    'needs a self.r times nos matrix of samples and returns a list of size self.r with '
    'elements of size pol_deg times nos with polynomials evaluated' 
    def P_batch(self, x):
        ret = []
        ret = [np.zeros(shape=(self.pol_deg, x.shape[1])) for _ in range(self.r)]
        for i0 in range(self.r):
            for i1 in range(self.pol_deg):
                ret[i0][i1,:] = self.pol[i1](x[i0,:])
        return ret
    
    'needs a self.r times nos matrix of samples and returns a list of size self.r with '
    'elements of size pol_deg times nos with polynomials evaluated' 
    def dP_batch(self, x):
        ret = []
        ret = [np.zeros(shape=(self.pol_deg, x.shape[1])) for _ in range(self.r)]
        for i0 in range(self.r):
            for i1 in range(0, self.pol_deg):
                ret[i0][i1,:] = self.dpol[i1](x[i0,:])
        return ret


    'needs a self.r times nos matrix of samples and returns a list of size self.r with '
    'elements of size pol_deg times nos with polynomials evaluated' 
    def ddP_batch(self, x):
        ret = []
        ret = [np.zeros(shape=(self.pol_deg, x.shape[1])) for _ in range(self.r)]
        for i0 in range(self.r):
            for i1 in range(0, self.pol_deg):
                ret[i0][i1,:] = self.ddpol[i1](x[i0,:])
        return ret


    def prepare_data_before_opt(self, x):
        return None


    def prepare_data_while_opt(self, t, x):
        return [self.P_batch(x), self.t_to_ind(t), self.add_fun(t, x)]


    def solve_linear_HJB(self, data, params):
        _, mat_list, ind, add_fun_list, rew_MC, P_vec, calc_validation_set_fun, curr_samples_test, rew_MC_test = data
        # print('ind', ind)
        V = 1*self.V[ind]
        n_sweep, rel_val_tol = params

        _n_sweep = 0; rel_val = 1; rel_val_test = 1
        smin = 0.2#/np.sqrt(nos)*np.sqrt(np.power(p,noo)*noo)
        val = 1e-6
        omega = 1.0
        fmin = 0.2
        fomega = 1.05
        omega_list = []
        smin_list = []
        val_list = []
        val_list_test = []
        maxrank = 2
        maxranks = maxrank*np.ones(V.order()-1)
        maxranksiter = V.dimensions[0]
        counter = 0
        while maxranksiter <= maxrank and counter < V.order() - 1:
            maxranks[counter] = maxranksiter
            counter += 1
            maxranks[-counter] = maxranksiter
            maxranksiter *= V.dimensions[counter]
        adapt = False
        kminor = 1
        min_sweep_before_adapt = 2
        old_val_test = calc_validation_set_fun([V, self.c_add_fun_list[ind]], curr_samples_test, rew_MC_test)
        ranks_before = V.ranks()
        ranks_changed = False
        ranks_counter = 0
        # while _n_sweep < n_sweep and rel_val > rel_val_tol:
        while _n_sweep < n_sweep and rel_val > rel_val_tol and rel_val_test > (rel_val_tol-0.0001) or _n_sweep <= 0:
            if ranks_counter >= min_sweep_before_adapt:
                adapt = True
            if ranks_before != V.ranks():
                ranks_before = V.ranks()
                adapt = False
                ranks_counter = 0
            if val < 1e-8:
                adapt = False
                if _n_sweep == 50:
                    adapt = True
            # V = 1e-4*xe.TTTensor.random(V.dimensions, V.ranks())
            V.move_core(0)
            V_old = 1*V
            _n_sweep += 1; ranks_counter += 1
            # print('val', val)
            c_old = self.c_add_fun_list[ind] 
            old_val, val, c_new = self.update_components_np(V, val, mat_list, rew_MC, _n_sweep, P_vec, smin, omega, kminor, adapt, maxranks, add_fun_list, self.c_add_fun_list[ind])
            self.c_add_fun_list[ind] = c_new
            val_test = calc_validation_set_fun([V, self.c_add_fun_list[ind]], curr_samples_test, rew_MC_test)
            rel_val = (old_val - val) / old_val
            rel_val_test = (old_val_test - val_test) / old_val_test
            val_list.append(val)
            val_list_test.append(val_test)
            print("---------- iterate = " + str(_n_sweep) + " omega = " + str(omega) + " smin = " + str(smin) + " val " + str(val), 'val_test', val_test, 'adapt', adapt) 
            print(V.ranks()) 
            print('val, old_val', val, old_val, 'rel_val', rel_val, '_n_sweep', _n_sweep, 'frob_norm(v)', xe.frob_norm(V), 'validation, oldval', val_test, old_val_test, 'rel_val_test', rel_val_test)
            old_val_test = 1*val_test
            omega = np.min([omega/fomega,np.max([val, np.sqrt(val)])])
            omega = np.max([omega,rel_val])
            smin = np.max([np.min([0.2*omega, 0.2*rel_val]), 0]) #/np.sqrt(noo)*np.sqrt(np.power(p,noo)*noo)
            # smin = np.min([0.2*omega, 0.2*rel_val, 0.2*rel_val_test]) #/np.sqrt(noo)*np.sqrt(np.power(p,noo)*noo)
            # print('smin after update', smin, 'omega', omega, 'rel_val', rel_val, 'rel_val_test', rel_val_test)
            if val < 1e-16:
                break
        # if rel_val_test < 0:
            # V = V_old
            # self.c_add_fun_list[ind] = c_old
        # print("---------- iterate = " + str(_n_sweep) + " omega = " + str(omega) + " smin = " + str(smin) + " val " + str(val), 'val_test', val_test, 'adapt', adapt) 
        # print(V.ranks()) 
        # print('val, old_val', val, old_val, 'rel_val', rel_val, '_n_sweep', _n_sweep, 'frob_norm(v)', xe.frob_norm(V), 'validation, oldval', val_test, old_val_test, 'rel_val_test', rel_val_test)
        print('self.c_add_fun_list[ind]', self.c_add_fun_list[ind], 'c_old', c_old, 'c_new', c_new)
        # print('V', xe.Tensor(V), 'curr_c =', self.c_add_fun_list[ind])
        # input()
        # input()
        V.round(np.max([0.01*smin,1e-14]))
        self.V[ind] = V
        # for i0 in range(980, len(self.t_vec_p)):
            # print('after solvlinhjb frobnorms', i0, xe.frob_norm(self.V[i0]))

        
    def adapt_ranks(self, U, S, Vt,smin):
        """ Add a new rank to S
        Parameters
        ----------
        U: xe.Tensor
            left part of SVD
        S: xe.Tensor
            middle part of SVD, diagonal matrix
        Vt: xe.Tensor
            right part of SVD
        smin: float
            Threshold for smalles singluar values
    
        Returns
        -------
        Unew: xe.Tensor
            left part of SVD with one rank increased
        Snew: xe.Tensor
            middle part of SVD, diagonal matrix with one rank increased
        Vtnew: xe.Tensor
            right part of SVD with one rank increased
        """
        i1,i2,i3,i4,i5,i6,j1,j2,j3,j4,k1,k2,k3 = xe.indices(13)
        res = xe.Tensor()
        #S
        Snew = xe.Tensor([S.dimensions[0]+1,S.dimensions[1]+1])
        Snew.offset_add(S, [0,0])
        Snew[S.dimensions[0],S.dimensions[1]] = 0.01 * smin
    
        #U
        onesU = xe.Tensor.ones([U.dimensions[0],U.dimensions[1]])
        Unew = xe.Tensor([U.dimensions[0],U.dimensions[1],U.dimensions[2]+1])
        Unew.offset_add(U, [0,0,0])
        res(i1,i2) << U(i1,i2,k1) * U(j1,j2,k1) * onesU(j1,j2)
        onesU = onesU - res
        res(i1,i2) << U(i1,i2,k1) * U(j1,j2,k1) * onesU(j1,j2)
        onesU = onesU - res
        onesU.reinterpret_dimensions([U.dimensions[0],U.dimensions[1],1])
        if xe.frob_norm(onesU) != 0:
            onesU = onesU / xe.frob_norm(onesU)
            Unew.offset_add(onesU, [0,0,U.dimensions[2]])
    
        #Vt
        onesVt = xe.Tensor.ones([Vt.dimensions[1],Vt.dimensions[2]])
        Vtnew = xe.Tensor([Vt.dimensions[0]+1,Vt.dimensions[1],Vt.dimensions[2]])
        Vtnew.offset_add(Vt, [0,0,0])
        res(i1,i2) << Vt(k1,i1,i2) * Vt(k1,j1,j2) * onesVt(j1,j2)
        onesVt = onesVt - res
        res(i1,i2) << Vt(k1,i1,i2) * Vt(k1,j1,j2) * onesVt(j1,j2)
        onesVt = onesVt - res
        onesVt.reinterpret_dimensions([1,Vt.dimensions[1],Vt.dimensions[2]])
        if xe.frob_norm(onesVt) != 0:
            onesVt = onesVt / xe.frob_norm(onesVt)
            Vtnew.offset_add(onesVt, [Vt.dimensions[0],0,0])
    
        return Unew, Snew, Vtnew


    def update_components_np(self, G, w, mat_list, rew_MC, n_sweep, P_constraints_vec, smin, omega, kminor,adapt, maxranks, add_fun_list, current_fun_c):
        noo = G.order()
        Smu_left,Gamma, Smu_right, Theta, U_left, U_right, Vt_left, Vt_right = (xe.Tensor() for i in range(8))
        p = mat_list[0].shape[0]
        i1,i2,i3,i4,i5,i6,j1,j2,j3,j4,k1,k2,k3 = xe.indices(13)
        constraints_constant = 0
        num_constraints = P_constraints_vec[0].shape[1]
        d = G.order()
          # building Stacks for operators   
        lStack_x = [np.ones(shape=[1,rew_MC.size])]
        rStack_x = [np.ones(shape=[1,rew_MC.size])]
        G0_lStack = [np.ones(shape=(1, num_constraints))]
        G0_rStack = [np.ones(shape=(1, num_constraints))]
        if G.order() > 1:
            G.move_core(1)
            G.move_core(0)
        for i0 in range(d-1,0,-1): 
            G_tmp = G.get_component(i0).to_ndarray()
            A_tmp_x = mat_list[i0]
            rStack_xnp = rStack_x[-1]
            G_tmp_np_x = np.tensordot(G_tmp, A_tmp_x, axes=((1),(0)))
            rStack_xnpres = np.einsum('jkm,km->jm',G_tmp_np_x, rStack_xnp)
            rStack_x.append(rStack_xnpres)
            
            rStack_G0_tmp = G0_rStack[-1]            
            G0_tmp_np = np.tensordot(G_tmp, P_constraints_vec[i0], axes=((1),(0)))
            G0_tmp = np.einsum('jkm,km->jm',G0_tmp_np, rStack_G0_tmp)
            # G0_tmp = np.einsum('ijk,jl,kl->il',G_tmp, P_constraints_vec[i0], rStack_G0_tmp)
            G0_rStack.append(G0_tmp)
        #loop over each component from left to right
        for i0 in range(0, d):
            # get singular values and orthogonalize wrt the next core mu
            if i0 > 0:
                # get left and middle component
                Gmu_left = G.get_component(i0-1)
                Gmu_middle = G.get_component(i0)
                (U_left(i1,i2,k1), Smu_left(k1,k2), Vt_left(k2,i3)) << xe.SVD(Gmu_left(i1,i2,i3))
                Gmu_middle(i1,i2,i3) << Vt_left(i1,k2) *Gmu_middle(k2,i2,i3)
                if G.ranks()[i0-1] < maxranks[i0-1] and adapt \
                    and Smu_left[int(np.max([Smu_left.dimensions[0] - kminor,0])),int(np.max([int(Smu_left.dimensions[1] - kminor),0]))] > smin:
                    U_left, Smu_left, Gmu_middle  = self.adapt_ranks(U_left, Smu_left, Gmu_middle,smin)
                sing = [Smu_left[i,i] for i in range(Smu_left.dimensions[0])]
                # print('left', 'smin', smin, 'sing', sing)
                
                Gmu_middle(i1,i2,i3) << Smu_left(i1,k1)*Gmu_middle(k1,i2,i3)
                G.set_component(i0-1, U_left)
                G.set_component(i0, Gmu_middle)
                Gamma = np.zeros(Smu_left.dimensions) # build cut-off sing value matrix Gamma
                for j in range(Smu_left.dimensions[0]):
                    Gamma[j,j] = 1 / np.max([smin,Smu_left[j,j]])
                # print('Gamma', Gamma)
            if i0 < d - 1:
                # get middle and rightcomponent
                Gmu_middle = G.get_component(i0)
                Gmu_right = G.get_component(i0+1)
                (U_right(i1,i2,k1), Smu_right(k1,k2), Vt_right(k2,i3)) << xe.SVD(Gmu_middle(i1,i2,i3))


                sing = [Smu_right[i,i] for i in range(Smu_right.dimensions[0])]
                # print('right', 'smin', smin, 'sing', sing)
                Gmu_right(i1,i2,i3) << Vt_right(i1,k1) *Gmu_right(k1,i2,i3)
                #if mu == d-2 and G.ranks()[mu] < maxranks[mu] and adapt and Smu_right[Smu_right.dimensions[0] - kminor,Smu_right.dimensions[1] - kminor] > smin:
                #    U_right, Smu_right, Gmu_right  = adapt_ranks(U_right, Smu_right, Gmu_right,smin)
                Gmu_middle(i1,i2,i3) << U_right(i1,i2,k1) * Smu_right(k1,i3)
                # G.set_component(i0, Gmu_middle)
                # G.set_component(i0+1, Gmu_right)
                Theta = np.zeros([G.ranks()[i0], G.ranks()[i0]]) # build cut-off sing value matrix Theta
                # Theta = np.zeros([Gmu_middle.dimensions[2],Gmu_middle.dimensions[2]]) # build cut-off sing value matrix Theta
                for j in range(Theta.shape[0]):
                    if j >= Smu_right.dimensions[0]:
                        sing_val = 0
                    else:
                        singval = Smu_right[j,j] 
                    Theta[j,j] = 1 / np.max([smin,singval])
            # update Stacks
            if i0 > 0:
                G_tmp = G.get_component(i0-1).to_ndarray()
                A_tmp_x = mat_list[i0-1]
    #            G_tmp_np = np.einsum('ijk,jl->ikl', G_tmp, A_tmp_x)
                G_tmp_np_x = np.tensordot(G_tmp, A_tmp_x, axes=((1),(0)))
                lStack_xnp = lStack_x[-1]
                lStack_xnpres = np.einsum('jm,jkm->km', lStack_xnp, G_tmp_np_x)
                lStack_x.append(lStack_xnpres)
                del rStack_x[-1]
                G0_lStack_tmp = G0_lStack[-1]
                G0_tmp_np = np.tensordot(G_tmp, P_constraints_vec[i0-1], axes=((1),(0)))
                G0_tmp = np.einsum('jm,jkm->km', G0_lStack_tmp, G0_tmp_np)
                # G0_tmp = np.einsum('il,ijk,jl->kl',G0_lStack_tmp, G_tmp, P_constraints_vec[i0-1])
                G0_lStack.append(G0_tmp)
                del G0_rStack[-1]
    
            Ai_x = mat_list[i0]
            lStack_xnp = lStack_x[-1]; rStack_xnp = rStack_x[-1]
            op_pre = np.einsum('il,jl,kl->ijkl',lStack_xnp,Ai_x,rStack_xnp)
    #        op = np.einsum('ijkl,mnol->ijkmno', op_pre, op_pre)
            op_G0 = np.einsum('il,jl,kl->ijkl', G0_lStack[-1], P_constraints_vec[i0], G0_rStack[-1])
            op = np.zeros(op_pre.shape[:-1]+op_pre.shape[:-1])
            op_dim = op.shape
            Gi = G.get_component(i0)


            id_reg_p = np.eye(p)
            if i0 > 0:
                id_reg_r = np.eye(Gi.dimensions[2])
                # op_reg(i1,i2,i3,j1,j2,j3) << Gamma(i1,k1) * Gamma(k1,j1) * id_reg_r(i3,j3)  * id_reg_p(i2,j2)
                op_reg = np.einsum('ij,jk,lm,no->inlkom', Gamma, Gamma, id_reg_r, id_reg_p)
                # print('op_reg', op_reg)
                op += w*w * op_reg
            # input()
            if i0 < d-1:
                id_reg_l = np.eye(Gi.dimensions[0])
                # op_reg(i1,i2,i3,j1,j2,j3) << Theta(i3,k1) * Theta(k1,j3) * id_reg_l(i1,j1)  * id_reg_p(i2,j2)
                op_reg = np.einsum('ij,jk,lm,no->lnimok', Theta, Theta, id_reg_l, id_reg_p)
                op += w*w * op_reg

            op = op.reshape((op_dim[0]*op_dim[1]*op_dim[2], op_dim[3]*op_dim[4]*op_dim[5]))
            

            op = np.vstack([op, np.zeros(op.shape[0])[None,:]])
            op = np.hstack([op, np.zeros(op.shape[0])[:,None]])
            rhs_dim = op_pre.shape[:-1]
            op_pre = op_pre.reshape(op_dim[0]*op_dim[1]*op_dim[2], op_pre.shape[-1])
            op_pre = np.concatenate([op_pre, add_fun_list[None, :]], axis=0)
            op += np.tensordot(op_pre, op_pre, axes=((1),(1)))
            # op += 2*rew_MC.size*constraints_constant*np.tensordot(op_G0, op_G0, axes=((3),(3)))
    #        rhs = np.einsum('ijkl,l->ijk', op_pre, rew_MC)
            rhs = np.tensordot(op_pre, rew_MC, axes=((1),(0)))







            if(n_sweep == 1 and i0 == 0):
                comp = G.get_component(i0).to_ndarray()
                Ax = np.tensordot(op_pre[:-1, :].reshape(comp.shape+(rew_MC.size,)), comp, axes=([0,1,2],[0,1,2]))
                curr_const = np.einsum('il,jl,kl,ijk ->l', G0_lStack[-1], P_constraints_vec[i0], G0_rStack[-1], comp)
                w = min(np.linalg.norm(Ax + current_fun_c*add_fun_list- rew_MC)**2/rew_MC.size + constraints_constant*np.linalg.norm(curr_const)**2, 10000)
                # print('first_res', w, np.linalg.norm(Ax - rew_MC)**2/rew_MC.size, constraints_constant*np.linalg.norm(curr_const)**2)
            op += 1e-0 * w * np.eye(op.shape[0])
            rhs_reshape = rhs
            sol_arr = np.linalg.solve(op, rhs_reshape)
            current_fun_c = sol_arr[-1]
            sol_arr = sol_arr[:-1]
            sol_arr_reshape = sol_arr.reshape((rhs_dim[0], rhs_dim[1], rhs_dim[2]))
            sol = xe.Tensor.from_buffer(sol_arr_reshape)
            G.set_component(i0, sol)
    
        # calculate residuum
    #    Ax = np.einsum('jkli,jkl->i', op_pre, sol_arr_reshape)
        # print(i0)
        comp = G.get_component(d-1).to_ndarray()
        Ax = np.tensordot(op_pre[:-1, :].reshape(comp.shape+(rew_MC.size,)), comp, axes=([0,1,2],[0,1,2]))
        curr_const = np.einsum('il,jl,kl,ijk ->l', G0_lStack[-1], P_constraints_vec[d-1], G0_rStack[-1], sol_arr_reshape)
        # print(curr_const)
        error1 = np.linalg.norm(Ax + current_fun_c*add_fun_list - rew_MC)**2/rew_MC.size
        error2 = constraints_constant*np.linalg.norm(curr_const)**2
        # print('after', error1, error2)
        return w, error1 + error2, current_fun_c

    def calc_dof(self):
        dof = 0  # calculate orders of freedom
        V = self.V[0]
        for i0 in range(V.order()):
            dof += V.get_component(i0).size
        return dof

    def t_to_ind(self, t):
        # print('t, t/self.tau, int(t/self.tau', t, t/self.tau, int(t/self.tau))
        # return int(t/self.tau)
        return int(np.round(t/self.tau, 12))

    def test(self):
        # self.load_valuefun('V_')
        ind = self.t_to_ind(0.5)
        print('ind', ind)
        n = 16
        x = np.zeros(n)
        # x = np.zeros((n, 2))
        print('x', x)
        print('eval', self.eval_V(0, x))
        print('grad', self.calc_grad(0, x))



# vfun = Valuefunction_TT()
# vfun.test()
