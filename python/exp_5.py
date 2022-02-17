import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import helper_functions as hf
#from matplotlib.ticker import PercentFormatter

#define random variables, code1, code2
rvs_arr = [st.pareto, st.t, st.cauchy, st.uniform, st.expon, st.gumbel_r]
c_arr = [2.0, 5.0, np.nan, np.nan, np.nan, np.nan]
g_arr =  [0.5, 0.2, 1.0, -1.0, 0.0, 0.0]
code1 = 0
code2 = 0

#set parameters N, M - number of samples, number of trials respectively
N = 25
M = 10000
seeds = hf.get_seeds(1, 500000)

p_data = np.zeros((len(rvs_arr), M, 2))

for r in range(len(rvs_arr)):
    if np.isnan(c_arr[r]):
        X = hf.gen_noparam(M, N, rvs_arr[r], seeds[0])
    else:
        X = hf.gen(M, N, rvs_arr[r], c_arr[r], seeds[0])
    
    for m in range(M):
        #run the EVT estimator
        thresh = hf.get_threshold(X[m,:], code1, code2, N)[0]
        Y_ex = hf.get_excesses(X[m,:], thresh)
        p_data[r,m,:] = hf.get_parameters(code2, Y_ex) #0 - pwme, 1 - mle

fig, axs = plt.subplots(2, 3, sharey=True, tight_layout=True)

#plotting, hardcoded to number of variables here
for r in range(6):
    axs[int(r > 2), r % 3].hist(p_data[r,:,0], bins=25)
    axs[int(r > 2), r % 3].axvline(x=g_arr[r], color="r", linestyle="--")
    axs[int(r > 2), r % 3].set_title(rvs_arr[r].name)

#axs[0,0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
plt.savefig(os.path.join("plots","exp_5" + "N" + str(N) + ".png"))
