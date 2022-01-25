import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import helper_functions as hf

#set initial values
alpha = np.array([0.05, 0.005, 0.0005 , 0.00005, 0.000005])
M = 10000
K_val = [i + 5 for i in range(46)]
seeds = hf.get_seeds(len(K_val), 272542)

#set random variables using scipy.stats
rvs_arr = [st.weibull_min, st.lognorm, st.fisk]
c_arr = np.array([0.5, 0.5, 2.0])
p_data = np.zeros((len(rvs_arr), len(K_val), len(alpha), 2, 2)) #(0,:,:,:,0) for VaR, (0,:,:,:,1) for CVaR

#get ground-truth values for each alpha level (LLN)
N_gt = 50000000
#hf.ground_truth(N_gt, alpha, rvs_arr, c_arr, "exp_3_gt.npy")
gt = np.load("exp_3_gt.npy")
print(gt) #access each vector like gt[0,:] (adjust 0 - 5)

for r in range(len(rvs_arr)):
    for k in range(len(K_val)):
        X = hf.gen(M, K_val[k], rvs_arr[r], c_arr[r], seeds[k])
        var_h, var_e, cvar_h, cvar_e = np.zeros((4, M, len(alpha)))

        for m in range(M):
            par = hf.estimator_pwme(X[m,:], 0.10, K_val[k])

            var_h[m,:] = hf.var_emp(X[m,:], alpha, K_val[k])
            cvar_h[m,:] = hf.cvar_emp(X[m,:], alpha, K_val[k], var_h[m,:])
            var_e[m,:] = hf.var_evt(alpha, par)
            cvar_e[m,:] = hf.cvar_evt(alpha, par, var_e[m,:])
            
        p_data[r,k,:,0,0] = var_h.mean(axis=0)
        p_data[r,k,:,1,0] = var_e.mean(axis=0)
        p_data[r,k,:,0,1] = cvar_h.mean(axis=0)
        p_data[r,k,:,1,1] = np.nanmean(cvar_e, axis=0)

#save data
np.save("p_data.npy", p_data)

#plotting
for r in range(len(rvs_arr)):
    fig, axs = plt.subplots(2,2, sharex='col')
    #print(axs)

    for i in range(2): #i switches between h (emp), e (evt)
        for j in range(2): #j switches between VaR, CVaR
            axs[i,j].plot(K_val, np.abs(p_data[r,:,:,i,j] - gt[r + len(rvs_arr)*j,:].reshape((1,len(alpha))).repeat(len(K_val),axis=0)))

    axs[0,0].set_ylabel("Error [emp]")
    axs[1,0].set_ylabel("Error [evt]")
    axs[1,0].set_xlabel("K - # of samples")
    axs[1,1].set_xlabel("K - # of samples")
    axs[0,0].title.set_text("VaR")
    axs[0,1].title.set_text("CVaR")

    plt.savefig(os.path.join("plots","exp_3" + str(r + 1) + "M" + str(M) + ".png"))



