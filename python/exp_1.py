import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import helper_functions as hf
import os

#test parameters - M is number of trials, N is samples/trial
M = 1000
N = 21
q = 0.65
code1 = 1
code2 = 0

def cdf_evt_old(X, est_params):
    K, a, ga, n, z_mk = est_params
    return 1 - (K/n)*(1 + ga*((X - z_mk) / a))**(-1/ga)

def estimator_pwme_old(Z, al, n):
    c, d = [0.1, 0.0] #base is [0.0, 0.0]
    
    X = np.sort(Z)
    var = np.quantile(X, 1 - al)
    i = np.abs(X - var).argmin()

    #python is zero-indexed, so we subtract (i+1)
    K = n - 1.0 - i

    z_mk = X[i]
    i_arr = n - 1.0 - np.arange(i,n) #gives an array of size K + 1, going 1, K-1/K, ... , 1/K, 0/K
    
    #print("K, i_arr:",K, i_arr, "Z_mk:", z_mk)
    P = (1/K) * ((X[i:] - X[i])).sum()
    Q = (1/K) * ((i_arr + c)/(K + d) * (X[i:] - X[i])).sum()

    if Q == 0.0:
        print("Error: Estimator failed")
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        ga = 1 - (P/(2*Q) - 1)**(-1)
        a = P * (P/(2*Q) - 1)**(-1)
        return np.array([K, a, ga, n, z_mk])

#generate samples from Weibull distribution w\ fixed random seed
rvs_arr = [st.weibull_min, st.lognorm, st.fisk]
c_arr = np.array([0.5, 0.5, 2.0])

for r in range(len(rvs_arr)):
    X = hf.gen(M, N, rvs_arr[r], c_arr[r], 272542)
    t = np.linspace(0,20,1001)

    #apply EVT estimator to each trial to get M values of z_mk, find qth quantile z_star and indices z_mk <= z_star
    z_arr = np.zeros(M)
    for i in range(M):
        z_arr[i] = hf.get_threshold(X[i,:], code1, code2, N)[0] #change second arg. to 0 to use standard method
        #print(z_arr[i])

    print("thresholds evaluated")
    z_star = np.quantile(z_arr, q)
    #print("z_star:", z_star)
    z_ind = np.nonzero(np.logical_and(z_arr <= z_star, z_arr > 0))[0]

    #get empirical, evt cdfs for t >= z_star
    t_star = t[t >= z_star]
    Fhat_arr = np.zeros((z_ind.size, t_star.size))
    Fevt_arr = np.zeros((z_ind.size, t_star.size))

    for i in range(z_ind.size):
        index = z_ind[i]
        Y_ex = hf.get_excesses(X[index, :], z_arr[index])
        ga, b = hf.get_parameters(code2, Y_ex) #0 - pwme, 1 - mle

        Fhat_arr[i,:] = hf.cdf_emp(t_star, X[index, :])
        Fevt_arr[i,:] = hf.cdf_evt(t_star - z_arr[index], N, ga, b, Y_ex.size)

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
