import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.distributions.empirical_distribution import ECDF
import random

def get_seeds(N, s):
    #initialize the random state within the function
    random.seed(s)
    seeds = np.zeros(N, dtype=int)
    for i in range(N):
        seeds[i] = random.randint(100000, 999999)
    
    return seeds
    
def sample_weibull(M, K, c, s):
    A = st.weibull_min.rvs(c, loc=0, scale=1, size=(M,K), random_state=np.random.RandomState(seed=s))
    return A

def estimator_pwme(Z, al, n):
    c, d = [0.1, 0.0] #base is [0.0, 0.0]
    
    X = np.sort(Z)
    var = np.quantile(X, 1 - al)
    i = np.abs(X - var).argmin() - 1

    #python is zero-indexed, so we subtract (i+1)
    K = n - i - 1.0

    z_mk = X[i]
    i_arr = n - 1.0 - np.arange(i,n) #gives an array of size K + 1, going 1, K-1/K, ... , 1/K, 0/K
    
    P = (1/K) * ((X[i:] - X[i])).sum()
    Q = (1/K) * ((i_arr + c)/(K + d) * (X[i:] - X[i])).sum()

    if Q == 0.0:
        print("Error: Estimator failed")
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        ga = 1 - (P/(2*Q) - 1)**(-1)
        a = P * (P/(2*Q) - 1)**(-1)
        return np.array([K, a, ga, n, z_mk])

def cdf_emp(X, X_0):
    F = ECDF(X_0)
    return F(X)

def cdf_evt(X, est_params):
    K, a, ga, n, z_mk = est_params
    return 1 - (K/n)*(1 + ga*((X - z_mk) / a))**(-1/ga)
    
def cdf_gt(X, c):
    return st.weibull_min.cdf(X, c, loc=0, scale=1)

#test parameters - M is number of trials, N is samples/trial
M = 100
N_arr = [i + 5 for i in range(46)]

norm_arr_h = np.zeros((len(N_arr), M))
norm_arr_e = np.zeros((len(N_arr), M))
seed_arr = get_seeds(len(N_arr), 272542)

q = 0.90

for j in range(len(N_arr)):
    #generate samples from Weibull distribution w\ fixed random seed
    c = 0.8
    X = sample_weibull(M, N_arr[j], c, seed_arr[j])
    t = np.linspace(0,10,101)

    #apply EVT estimator to each trial to get M values of z_mk, find qth quantile z_star and indices z_mk <= z_star
    z_arr = np.zeros(M)
    for i in range(M):
        z_arr[i] = estimator_pwme(X[i,:], 0.05, N_arr[j])[-1]

    z_star = np.quantile(z_arr, q)
    z_ind = np.nonzero(np.logical_and(z_arr <= z_star, z_arr > 0))[0]

    #get empirical, evt cdfs for t >= z_star
    t_star = t[t >= z_star]

    F_gt = cdf_gt(t_star, c) #evaluate only along t_star for comparing

    for i in range(z_ind.size):
        index = z_ind[i]
        Y = X[index, :]
        params = estimator_pwme(Y, 0.05, N_arr[j]) #smarter implementations could save this for later, but alas

        norm_arr_h[j,index] = np.abs(cdf_emp(t_star, Y) - F_gt).max() 
        norm_arr_e[j,index] = np.abs(cdf_evt(t_star, params) - F_gt).max()
    
    print("Iteration done for N ==", N_arr[j])

#filter zero-values, these are where z_arr == 0 OR z_arr > z_star
norm_h_mean = norm_arr_h.mean(axis=1, where=norm_arr_h>0)
norm_h_std = norm_arr_h.std(axis=1, where=norm_arr_h>0)

norm_e_mean = norm_arr_e.mean(axis=1, where=norm_arr_e>0)
norm_e_std = norm_arr_e.std(axis=1, where=norm_arr_e>0)
print(norm_h_mean.size)

#plotting
fig, ax = plt.subplots()
ax.plot(N_arr, norm_h_mean, label=r'$\hat{F}(t)$', color="orange")
ax.fill_between(N_arr, norm_h_mean - norm_h_std, norm_h_mean + norm_h_std, alpha=0.2, color="orange", interpolate=True)

ax.plot(N_arr, norm_e_mean, label=r'$F_{EVT}$(t)', color="green")
ax.fill_between(N_arr, norm_e_mean - norm_e_std, norm_e_mean + norm_e_std, alpha=0.2, color="green", interpolate=True)

ax.set_xlabel(r'$n$')
ax.set_ylabel("sup-norm")
ax.legend()

plt.show()