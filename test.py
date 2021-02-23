import numpy as np
from utilities import get_X_process 
import problems
import matplotlib.pyplot as plt



cosexp = problems.CosExp()

K = 10000
delta_t = 0.01
seed = 42

samples, xi =  get_X_process(cosexp, K, delta_t, seed)



plt.figure()
plt.scatter(curr_samples.reshape((rew_MC.size)), rew_MC, s=0.1)
plt.scatter(curr_samples.reshape((rew_MC.size)), self.v.eval_V(self.current_time, curr_samples), c='r', s=0.1)
plt.show()
