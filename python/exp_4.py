import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import os
import helper_functions as hf
#np.seterr(all='raise')

#set initial values
#alpha = np.array([0.005, 0.0005])
M = 10000
code1 = 0
code2 = 0
K_val = [5*i+20 for i in range(97)] #[28, 29, 30, 31, 32]
seeds = hf.get_seeds(len(K_val), 500000)
beta = 0.90

#set random variables using scipy.stats
rvs_arr = [st.gumbel_r, st.expon, st.lognorm]
c_arr = [np.nan, np.nan, 0.7] #if generating data requires a parameter include it here, else use nan
n_arr = ['Gumbel', 'Exponential', 'Lognormal (c == 0.5)']
p_data = np.zeros((len(rvs_arr), len(K_val), 2)) #(0,:,0) for VaR, (0,:,1) for CVaR

#get ground-truth value (LLN)
#hf.ground_truth_meansemi(50000000, rvs_arr, c_arr, beta, "exp_4_gt.npy")
gt = np.load("exp_4_gt.npy")
print(gt) #access each value, for rv_arr -> gt[i]

for r in range(len(rvs_arr)):
    for k in range(len(K_val)):
        if np.isnan(c_arr[r]):
            X = hf.gen_noparam(M, K_val[k], rvs_arr[r], seeds[k])
        else:
            X = hf.gen(M, K_val[k], rvs_arr[r], c_arr[r], seeds[k])

        semid_h, semid_e= np.zeros((2, M))

        for m in range(M):
            #run the EVT estimator
            thresh = hf.get_threshold(X[m,:], code1, code2, K_val[k])[0]
            Y_ex = hf.get_excesses(X[m,:], thresh)
            ga, b = hf.get_parameters(code2, Y_ex) #0 - pwme, 1 - mle

            #evaluate the meansemi_d
            semid_h[m] = hf.meansemi_emp(X[m,:], beta)
            semid_e[m] = hf.meansemi_evt(X[m,:], beta ,ga, b, Y_ex.size)
            
        p_data[r,k,0] = np.mean(semid_h, axis=0)
        p_data[r,k,1] = np.nanmean(semid_e, axis=0)

#save data
np.save("p_data.npy", p_data)

#plotting
fig, axs = plt.subplots(1,len(rvs_arr), figsize=(9,4), sharex='col')
for r in range(len(rvs_arr)):
    axs[r].plot(K_val, np.abs(p_data[r,:,0] - gt[r] * np.ones(len(K_val))), label="EMP")
    axs[r].plot(K_val, np.abs(p_data[r,:,1] - gt[r] * np.ones(len(K_val))), label="EVT")
    axs[r].set_xlabel("K - # of samples")
    axs[r].title.set_text(n_arr[r])
    
axs[0].set_ylabel("Error")
axs[-1].legend()
#axs[1,0].set_ylabel("Error [evt]")
plt.savefig(os.path.join("plots","exp_4" + "new" + "M" + str(M) + ".png"))