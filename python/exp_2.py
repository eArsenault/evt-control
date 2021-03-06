import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import helper_functions as hf
import os

#test parameters - M is number of trials, N is samples/trial
M = 100
code1 = 1
code2 = 0
N_arr = [i + 21 for i in range(79)]

rvs_arr = [st.weibull_min, st.lognorm, st.fisk]
c_arr = np.array([0.5, 0.5, 2.0])

norm_arr_h = np.zeros((len(N_arr), M))
norm_arr_e = np.zeros((len(N_arr), M))
seed_arr = hf.get_seeds(len(N_arr), 123456)

for r in range(len(rvs_arr)):
    for j in range(len(N_arr)):
        #generate samples from distribution
        X = hf.gen(M, N_arr[j], rvs_arr[r], c_arr[r], seed_arr[j])
        t = np.linspace(0,100,2001)

        #apply EVT estimator to each trial to get M values of z_mk, find qth quantile z_star and indices z_mk <= z_star
        for i in range(M):
            thresh = hf.get_threshold(X[i,:], code1, code2, N_arr[j])[0]
            Y_ex = hf.get_excesses(X[i, :], thresh)
            ga, b = hf.get_parameters(code2, Y_ex) #0 - pwme, 1 - mle

            t_z = t[t >= thresh]
            if t_z.shape == (0,):
                norm_arr_h[j,i] = np.nan
                norm_arr_e[j,i] = np.nan
            else:
                F_gt = hf.cdf_gt(t_z, rvs_arr[r], c_arr[r])

                norm_arr_h[j,i] = np.abs(hf.cdf_emp(t_z, X[i,:]) - F_gt).max() 
                norm_arr_e[j,i] = np.abs(hf.cdf_evt(t_z - thresh, N_arr[j], ga, b, Y_ex.size) - F_gt).max()

        print("Iteration done for N ==", N_arr[j])

    #filter zero-values, these are where z_arr == 0 OR z_arr > z_star
    norm_h_mean = np.nanmean(np.where(norm_arr_h > 0, norm_arr_h, np.nan), axis=1)
    norm_h_std = np.nanstd(np.where(norm_arr_h > 0, norm_arr_h, np.nan), axis=1)

    norm_e_mean = np.nanmean(np.where(norm_arr_e > 0, norm_arr_e, np.nan), axis=1)
    norm_e_std = np.nanstd(np.where(norm_arr_e > 0, norm_arr_e, np.nan), axis=1)

    #plotting
    fig, ax = plt.subplots()
    ax.plot(N_arr, norm_h_mean, label=r'$\hat{F}(t)$', color="orange")
    ax.fill_between(N_arr, norm_h_mean - norm_h_std, norm_h_mean + norm_h_std, alpha=0.2, color="orange", interpolate=True)

    ax.plot(N_arr, norm_e_mean, label=r'$F_{EVT}$(t)', color="green")
    ax.fill_between(N_arr, norm_e_mean - norm_e_std, norm_e_mean + norm_e_std, alpha=0.2, color="green", interpolate=True)

    ax.set_xlabel(r'$n$')
    ax.set_ylabel("sup-norm")
    ax.legend()

    plt.savefig(os.path.join("plots","exp_2" + str(r + 1) + "M" + str(M) + ".png"))