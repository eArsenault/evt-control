import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import helper_functions as hf
#from matplotlib.ticker import PercentFormatter

print("initialize")
#define random variables, code1, code2
rvs_arr = [st.pareto, st.t, st.uniform, st.beta, st.expon, st.gumbel_r]
c_arr = [(2.0,), (5.0,), np.nan, (1.0,2.0), np.nan, np.nan]
g_arr =  [0.5, 0.2, -1, -0.5, 0.0, 0.0]
code1 = 0
code2 = 0

#set parameters N, M - number of samples, number of trials respectively
N = 1000
M = 10000
seeds = hf.get_seeds(1, 500000)

p_data = np.zeros((len(rvs_arr), M, 2))
print('step 1')

for r in range(len(rvs_arr)):
    X = hf.gen(M, N, rvs_arr[r], c_arr[r], seeds[0])

    for m in range(M):
        #run the EVT estimator
        thresh = hf.get_threshold(X[m,:], code1, code2, N)[0]
        Y_ex = hf.get_excesses(X[m,:], thresh)
        p_data[r,m,:] = hf.get_parameters(code2, Y_ex) #0 - pwme, 1 - mle

print('step 2')
fig, axs = plt.subplots(3, 2, sharey=True, tight_layout=True)

#plotting, hardcoded to number of variables here
map = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)] #manually define the row-column for each rv

for r in range(6):
    axs[map[r]].hist(p_data[r,:,0], bins=50, histtype="step")
    axs[map[r]].axvline(x=g_arr[r], color="r", linestyle="--")
    axs[map[r]].set_title(rvs_arr[r].name)

#axs[0,0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
print('step 3')
fig.set_size_inches(16.5, 10.5)
plt.savefig(os.path.join("plots","exp_5" + "N" + str(N) + ".png"))
