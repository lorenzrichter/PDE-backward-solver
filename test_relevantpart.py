




# testOde.problem is bondprice_multidim
# vfun is my value function class
# vfun.eval_V = v auswerten
# vfun.calc_grad = grad v berechnen
# vfun.calc_hessian = hessematrix bestimmen
# am ende plotte ich nur bis zum vorletzten punkt weil beim letzten noch nen bug ist

testOde.problem.X_0 = x0
ind = testOde.t_to_ind(curr_t)
seed = 44
delta_t = tau
K = nos
samples_mat, noise_vec =  get_X_process(testOde.problem, K, delta_t, seed, sigma='nonconstant')
samples_mat = samples_mat.transpose((2,1,0))
noise_vec = np.sqrt(testOde.tau)*noise_vec.transpose((2,1,0))
avg_loss_vec = []
for ind in range(len(vfun.t_vec_p) - 1): # len(vfun.t_vec_p) ist quasi anzahl der zeitschritte. muesste man auch aus samples_mat.shape extrahieren koennen
    samples = samples_mat[:,:,ind]
    curr_t = vfun.t_vec_p[ind] #current_time
    print('t',curr_t,'ind', ind)
    
    v_of_x = vfun.eval_V(curr_t, samples).T
    if ind < len(vfun.t_vec_p) - 1:
        v_t  =  (vfun.eval_V(vfun.t_vec_p[ind+1], samples).T - v_of_x)/tau
    else:
        v_t  =  (testOde.calc_end_reward(samples).T - v_of_x)/tau # testOde.calc_end_reward ist aequivalent zu
    v_x = vfun.calc_grad(curr_t, samples).T
    v_xx = vfun.calc_hessian(curr_t, samples).transpose((2,0,1))
    loss = testOde.problem.pde_loss(curr_t, samples.T, v_of_x, v_t, v_x, v_xx)
    avg_loss_vec.append(np.mean(np.abs(loss)))
print('avg_loss_vec', avg_loss_vec)
plt.figure()
plt.plot(steps[:-2], avg_loss_vec[:-1])
plt.yscale('log')
plt.xlabel('time')
plt.ylabel('mean pointwise PDE error')
plt.show()


