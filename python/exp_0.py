import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from statsmodels.distributions.empirical_distribution import ECDF

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

def cdf_emp(X_0, X):
    F = ECDF(X_0)
    return F(X)

def cdf_evt(X, est_params):
    K, a, ga, n, z_mk = est_params
    return 1 - (K/n)*(1 + ga*((X - z_mk) / a))**(-1/ga)
    
def cdf_gt(X, c):
    return st.weibull_min.cdf(X, c, loc=0, scale=1)

M = 10
N = 20
c = 0.8
A = sample_weibull(M, N, c, 272542)
t = np.linspace(0,10,101)

i = 0
X = A[i,:]

#first: fit EVT estimator to data to get parameters
params = estimator_pwme(X, 0.05, N)

#second: find the subset of t such that t_u >= z_mk
z = params[-1]
t_u = t[t >= z]

#third: evaluate all three cdfs, plot
y1 = cdf_gt(t, c)
y2 = cdf_emp(X, t_u)
y3 = cdf_evt(t_u, params)

fig, ax = plt.subplots()
ax.plot(t, y1, label=r'$F$')
ax.plot(t_u, y2, label=r'$\hat{F}$')
ax.plot(t_u, y3, label=r'$F_{EVT}$')
ax.legend()

plt.show()

