import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import helper_functions as hf
import os

#test parameters - M is number of trials, N is samples/trial
M = 10000
N = 5
q = 0.90

#generate samples from Weibull distribution w\ fixed random seed
rvs_arr = [st.weibull_min, st.lognorm, st.fisk]
c_arr = np.array([0.5, 0.5, 2.0])

for r in range(len(rvs_arr)):
    X = hf.gen(M, N, rvs_arr[r], c_arr[r], 272542)
    t = np.linspace(0,20,1001)

    #apply EVT estimator to each trial to get M values of z_mk, find qth quantile z_star and indices z_mk <= z_star
    z_arr = np.zeros(M)
    for i in range(M):
        z_arr[i] = hf.estimator_pwme(X[i,:], 0.10, N)[-1]

    z_star = np.quantile(z_arr, q)
    z_ind = np.nonzero(np.logical_and(z_arr <= z_star, z_arr > 0))[0]

    #get empirical, evt cdfs for t >= z_star
    t_star = t[t >= z_star]
    Fhat_arr = np.zeros((z_ind.size, t_star.size))
    Fevt_arr = np.zeros((z_ind.size, t_star.size))

    for i in range(z_ind.size):
        index = z_ind[i]
        Y = X[index, :]
        params = hf.estimator_pwme(Y, 0.05, N) #smarter implementations could save this for later, but alas

        Fhat_arr[i,:] = hf.cdf_emp(t_star, Y)
        Fevt_arr[i,:] = hf.cdf_evt(t_star, params)

    #get average, std along t-axis
    Fh_m = np.mean(Fhat_arr, axis=0)
    Fh_std = np.std(Fhat_arr, axis=0)

    Fe_m = np.nanmean(Fevt_arr, axis=0) #evt estimator can introduce nan values
    Fe_std = np.nanstd(Fevt_arr, axis=0)

    #plotting
    fig, ax = plt.subplots()
    ax.plot(t, hf.cdf_gt(t, rvs_arr[r], c_arr[r]), label=r'$F$(t)')

    ax.plot(t_star, Fh_m, label=r'$\hat{F}(t)$', color="orange")
    ax.fill_between(t_star, Fh_m - Fh_std, np.minimum(1,Fh_m + Fh_std), alpha=0.2, color="orange")

    ax.plot(t_star, Fe_m, label=r'$F_{EVT}$(t)', color="green")
    ax.fill_between(t_star, Fe_m - Fe_std, np.minimum(1,Fe_m + Fe_std), alpha=0.2, color="green")

    ax.axvline(x=z_star, color='k', linestyle='--')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r"$P(Y \leq t)$")
    ax.legend()

    plt.savefig(os.path.join("plots","exp_1" + str(r + 1) + "M" + str(M) + ".png"))
