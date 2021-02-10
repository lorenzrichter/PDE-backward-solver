import xerus as xe
import numpy as np
import set_dynamics, main_p

pol_deg_vec = np.arange(0, 1)
num_samples_p = [2000]
n_vec = [100]
# n_vec = [1, 2]
# n_vec = [1, 5, 10, 20, 50, 100]

num_valuefunctions_vec = [101]
# num_valuefunctions_vec = [3, 6, 11]
# num_valuefunctions_vec = [11]
for i2 in num_samples_p:
    for i1 in num_valuefunctions_vec:
        for i3 in n_vec:
            for i0 in pol_deg_vec:
                print('pol_deg, num_valuefunctions, nos, n', i0, i1, i2, i3)
                set_dynamics.set_dynamics(i0, i1, i3)
                main_p.main_p(i2)
                # main_t.main_t()
