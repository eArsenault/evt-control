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
M = 10000
N = 5
q = 0.90

#generate samples from Weibull distribution w\ fixed random seed
c = 0.8
X = sample_weibull(M, N, c, 272542)
t = np.linspace(0,10,101)

#apply EVT estimator to each trial to get M values of z_mk, find qth quantile z_star and indices z_mk <= z_star
z_arr = np.zeros(M)
for i in range(M):
    z_arr[i] = estimator_pwme(X[i,:], 0.05, N)[-1]

z_star = np.quantile(z_arr, q)
z_ind = np.nonzero(np.logical_and(z_arr <= z_star, z_arr > 0))[0]

#get empirical, evt cdfs for t >= z_star
t_star = t[t >= z_star]
Fhat_arr = np.zeros((z_ind.size, t_star.size))
Fevt_arr = np.zeros((z_ind.size, t_star.size))

for i in range(z_ind.size):
    index = z_ind[i]
    Y = X[index, :]
    params = estimator_pwme(Y, 0.05, N) #smarter implementations could save this for later, but alas

    Fhat_arr[i,:] = cdf_emp(t_star, Y)
    Fevt_arr[i,:] = cdf_evt(t_star, params)

#get average, std along t-axis
Fh_m = np.mean(Fhat_arr, axis=0)
Fh_std = np.std(Fhat_arr, axis=0)

Fe_m = np.nanmean(Fevt_arr, axis=0) #evt estimator can introduce nan values
Fe_std = np.nanstd(Fevt_arr, axis=0)

#plotting
fig, ax = plt.subplots()
ax.plot(t, cdf_gt(t, c), label=r'$F$(t)')

ax.plot(t_star, Fh_m, label=r'$\hat{F}(t)$', color="orange")
ax.fill_between(t_star, Fh_m - Fh_std, np.minimum(1,Fh_m + Fh_std), alpha=0.2, color="orange")

ax.plot(t_star, Fe_m, label=r'$F_{EVT}$(t)', color="green")
ax.fill_between(t_star, Fe_m - Fe_std, np.minimum(1,Fe_m + Fe_std), alpha=0.2, color="green")

ax.axvline(x=z_star, color='k', linestyle='--')
ax.set_xlabel(r'$t$')
ax.set_ylabel(r"$P(Y \leq t)$")
ax.legend()

plt.show()
