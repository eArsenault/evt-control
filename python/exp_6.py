import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import helper_functions as hf

#define random variables, code1, code2
rvs_arr = [st.pareto, st.t, st.uniform, st.beta, st.expon, st.gumbel_r]
c_arr = [(2.0,), (5.0,), np.nan, (1.0,2.0), np.nan, np.nan] #must store values in iterables so we can use rvs with > 1 param
code1 = 0
code2 = 0
tol = 0.0001

#hf.ground_truth_eval(rvs_arr, c_arr, tol, "gt_exp_6.npy")
g_arr = np.load("gt_exp_6.npy")
print(g_arr.shape)

#set parameters N, M - number of samples, number of trials respectively
N = 25
M = 10000
seeds = hf.get_seeds(1, 500000)
p_data = np.zeros((len(rvs_arr), M, 2))

print("Initialization complete. Iterating now.")
for r in range(len(rvs_arr)):
    X = hf.gen(M, N, rvs_arr[r], c_arr[r], seeds[0])
    
    for m in range(M):
        #run the EVT estimator
        thresh = hf.get_threshold(X[m,:], code1, code2, N)[0]
        Y_ex = hf.get_excesses(X[m,:], thresh)
        ga, b = hf.get_parameters(code2, Y_ex) #0 - pwme, 1 - mle

        #now evaluate risk-functional, store result
        p_data[r,m,0] = hf.rfunction_exp(X[m,:])
        p_data[r,m,1] = hf.rfunction_evt(X[m,:], ga, b, Y_ex.size)

print("Iteration complete. Plotting now.")
fig, axs = plt.subplots(3, 2, sharey=True, tight_layout=True)

#plotting, hardcoded to number of variables here
map = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)] #manually define the row-column for each rv

for r in range(6):
    axs[map[r]].hist(p_data[r,:,0] - g_arr[r], bins=50, histtype="step", label="EMP")
    axs[map[r]].hist(p_data[r,:,1] - g_arr[r], bins=50, histtype="step", label="EVT")
    axs[map[r]].axvline(x=0, color="r", linestyle="--")
    axs[map[r]].set_title(rvs_arr[r].name)

#axs[0,0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
fig.set_size_inches(12.5, 10.5)
plt.savefig(os.path.join("plots","exp_6" + "N" + str(N) + ".png"))
print("Plotting complete.")
