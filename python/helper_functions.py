import numpy as np
import random
import scipy.stats as st
from scipy.special import expit, logit
from scipy.optimize import minimize
from scipy.linalg import norm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d

ad_quantiles = np.loadtxt("ADQuantiles.csv", delimiter=",",
                          dtype=float, skiprows=1, usecols=range(1, 1000))
ad_pvals = np.round(np.linspace(0.999, 0.001, 1000), 3)  # col names
ad_shape = np.round(np.linspace(-0.5, 1, 151), 2)  # row names

def cdf_emp(X, X_0):
    F = ECDF(X_0)
    return F(X)

def cdf_evt(Y_ex, n, ga, b, K):
    if np.abs(ga) < 0.001: #threshold for ga close to 0
        cdf = 1 - (K / n) * np.exp(-Y_ex / b)
    else:
        #print(ga, b)
        cdf = 1 - (K / n) * (1 + ga * (Y_ex / b))**(-1/ga)
    return cdf
    
def cdf_gt(X, rv, c):
    return rv.cdf(X, c, loc=0, scale=1)

def cdf_gt_noparam(X, rv):
    return rv.cdf(X, loc=0, scale=1)

#equation 7.19, Quantitative Risk Management, McNeil (integrate the EVT VaR from (1-al) to 1)
def cvar_evt(al, var, ga, b, u):
    if np.abs(ga) < 0.001:
        cvar = var + b
    else:
        cvar = var / (1 - ga) + (b - ga * u) / (1 - ga)
    return cvar

def cvar_emp(X, al, var):
    v_mat = np.hstack([var.reshape((len(al),1)) for i in range(len(X))])
    X_mat = np.vstack([X for i in range(len(al))])
    return  var + (X_mat - v_mat).sum(axis=1,where=X_mat>v_mat) / (len(X) * al)

def estimator_pwme(Y_ex):
    c, d = [0.0, 0.0] #base is [0.0, 0.0]
    K = Y_ex.size
    i_arr = np.flip(np.arange(Y_ex.size)) #array that goes [K-1, K-2, ..., 0]

    #print("New K, i_arr",K,i_arr)
    #to ensure Q != 0, we make sure u is chosen st. Y[Y > u] is non-empty
    P = (1/K) * ((Y_ex)).sum()
    Q = (1/K) * ((i_arr + c)/(K + d) * (Y_ex)).sum() 

    ga = 1 - (P/(2*Q) - 1)**(-1)
    b = P * (P/(2*Q) - 1)**(-1)
    return (ga, b)

def exponential_ult(Y, theta):
    return np.sort(np.exp(-(theta / 2) * Y))

def exp_ult_emp(Y, utility, param):
    return utility(Y, param).mean()

def exp_ult_evt(Y, utility, param, code1, code2):
    Y_exp = utility(Y, param)

    thresh = get_threshold(Y_exp, code1, code2, Y.size)[0]
    Y_ex = get_excesses(Y_exp, thresh)
    ga, b = get_parameters(code2, Y_ex)

    Y_K = Y_exp[:(Y.size - Y_ex.size)]
    t1 = Y_K.sum() / Y.size

    #new method goes here
    al = 0.01
    var = var_evt(al, ga, b, Y_K[-1], Y_ex.size, Y.size)
    Y_ex_in = Y_ex[Y_ex < var]

    t2 = Y_ex_in.sum() / Y.size
    t3 = al * cvar_evt(0, var, ga, b, Y_K[-1])

    return t1 + t2 + t3

def gen(M, K, rv, c, s):
    return rv.rvs(c, loc=0, scale=1, size=(M,K), random_state=np.random.RandomState(seed=s))

def gen_noparam(M, K, rv, s):
    return rv.rvs(loc=0, scale=1, size=(M,K), random_state=np.random.RandomState(seed=s))

def get_seeds(N, s):
    #initialize the random state within the function
    random.seed(s)
    seeds = np.zeros(N, dtype=int)
    for i in range(N):
        seeds[i] = random.randint(100000, 999999)
    
    return seeds

def ground_truth(N, al, rvs_arr, c_arr, file_name):
    var_arr = np.zeros((len(rvs_arr), len(al)))
    cvar_arr = np.zeros((len(rvs_arr), len(al)))

    for r in range(len(rvs_arr)):
        Y = gen(1, N, rvs_arr[r], c_arr[r], 272542).flatten()
        var_arr[r,:] = var_emp(Y, al)
        cvar_arr[r,:] = cvar_emp(Y, al, var_arr[r,:])

    np.save(file_name, np.vstack((var_arr, cvar_arr)))

def ground_truth_eval(rvs_arr, c_arr, beta, tol, file_name):
    #set parameters, initial value of algorithm
    N = 100000
    mul = 1
    delta = tol + 1
    X = ground_truth_meansemi(mul * N, rvs_arr, c_arr, beta)

    while (delta > tol) and (mul < 50):
        mul = mul + 1
        Xnew = ground_truth_meansemi(mul * N, rvs_arr, c_arr, beta)
        delta = norm(Xnew - X) #delta is 2-norm of the difference between successive iterates

        X = Xnew

    print("Ground truth computed, delta =",delta, ", iterations =", mul - 1)
    np.save(file_name, X)

def ground_truth_meansemi(N, rvs_arr, c_arr, beta):
    est_arr = np.zeros(len(rvs_arr))

    for r in range(len(rvs_arr)):
        if np.isnan(c_arr[r]):
            Y = gen_noparam(1, N, rvs_arr[r], 272542).flatten()
        else:
            Y = gen(1, N, rvs_arr[r], c_arr[r], 272542).flatten()

        est_arr[r] = meansemi_emp(Y, beta)

    return est_arr

def meansemi_emp(Y, beta):
    mu = Y.sum() / Y.size
    semi = (Y[Y > mu] - mu).sum() / Y.size
    return mu + beta * semi

def meansemi_evt(Y, beta, ga, b, K): #currently no checking if Y_K[-1] > mu
    Y.sort()
    Y_K = Y[:(Y.size - K)]
    Y_ex = Y[(Y.size - K):]
    mu = Y.sum() / Y.size
    
    t1 = (Y_K[Y_K > mu] - mu).sum() / Y.size

    al = 1 - cdf_evt(Y_ex[-1] - Y_K[-1], Y.size, ga, b, K) #evaluate tail CDF at top excess, or second top, etc.
    var = Y_ex[-1]
    Y_ex_in = Y_ex[:(K-1)]

    t2 = (Y_ex_in[Y_ex_in > mu] - mu).sum() / Y.size
    t3 = al * (cvar_evt(0, var, ga, b, Y_K[-1]) - mu)

    return mu + beta * (t1 + t2 + t3)

def mle_evt(Y_ex):
    mle = minimize(neg_loglike, np.array([logit(0.5), np.log(0.5)]), args=(Y_ex))
    return np.array([expit(mle.x[0]), np.exp(mle.x[1])])

#this assumes Y is an Nx1 vector, McNeil 7.14
def neg_loglike(x, Y):
    #the transformations here allow us to constrain ga in (0,1), b in (0, \infty)
    ga = expit(x[0])
    b = np.exp(x[1])
    N = Y.size

    return N * np.log(b) + (1 + 1/ga) * np.log(1 + ga * Y / b).sum()

def get_excesses(X, thresh):
    return np.sort(X[X > thresh]) - thresh

#only used in gpd_ad
def get_excesses_ad(X, q):
    thresh = var_emp(X, 1.0 - q)
    excesses = np.sort(X[X > thresh]) - thresh
    return thresh, excesses

def get_threshold(X, code1, code2, n):
    Y = np.sort(X)
    if code1 == 0:
        #naive, takes 90th percentile (will work so long as n >= 11)
        thresh = Y[np.ceil((1 - 0.10)*n).astype(int) - 1]
        u = 0.10
    elif code1 == 1:
        #implement Bader's selection method here
        u_start=0.79 
        u_end=0.98 
        u_num=20 
        signif=0.1
        cutoff=0.9

        u_vals = np.linspace(u_start, u_end, u_num)
        ad_tests = []
        pvals = []
        n_rejected = 0

        #cannot reliably estimate for u >= n-1/n, as then emp_var returns final value, excess is empty set
        for u in u_vals[u_vals < (n-1)/n]: 
            thresh, stat, ga, b = gpd_ad(X, code2, u) #how does this work?
            if ga <= cutoff:
                ad_tests.append([thresh, u])
                pvals.append(ad_pvalue(stat, ga))
            else:
                n_rejected += 1

        if len(ad_tests) == 0:
            #if test fails, return naive 90th percentile
            return Y[(np.ceil((1 - 0.10)*n) - 1).astype(int)], 0.10

        ad_tests = np.asarray(ad_tests)
        pvals = np.asarray(pvals)

        kf, stop = forward_stop(pvals, signif)
        thresh, u = ad_tests[stop]
    return thresh, u

def get_parameters(code, excesses):
    if code == 0:
        #pwme
        (ga, b) = estimator_pwme(excesses) #assume excesses have been sorted
    elif code == 1:
        #mle
        ga, _, b = st.genpareto.fit(excesses)
    return ga, b

def var_emp(X, al):
    Y = np.sort(X)
    return np.sort(X)[(np.ceil((1 - al)*len(X)) - 1).astype(int)] #evaluate order-statistics at ceil((1 - al)*n)

#Use Bayes rule to get tail probability above threshold, invert
def var_evt(al, ga, b, u, K, n):
    if np.abs(ga) < 0.001:
        var = u - b * np.log(al * n / K)
    else:
        var = u + (b / ga) * ((al * n / K)**(-ga) - 1)
    return var

## functions from Troop (2019) (Y_ex, n, ga, b)
def gpd_ad(X, code, u):
    thresh, Y_ex = get_excesses_ad(X, u)
    ga, b = get_parameters(code, Y_ex)
    Z = st.genpareto.cdf(Y_ex, ga, 0, b)

    n = len(Z)
    i = np.linspace(1, n, n)

    stat = -n - (1/n) * np.sum((2 * i - 1) * (np.log(Z) + np.log1p(-Z[::-1])))
    return thresh, stat, ga, b

def ad_pvalue(stat, ga):
    row = np.where(ad_shape == max(round(ga, 2), -0.5))[0][0]
    if stat > ad_quantiles[row, -1]:
        xdat = ad_quantiles[row, 950:999]
        ydat = -np.log(ad_pvals[950:999])
        lfit = np.polyfit(xdat, ydat, 1)
        m = lfit[0]
        b = lfit[1]
        p = np.exp(-(m*stat+b))
    else:
        bound_idx = min(np.where(stat < ad_quantiles[row, ])[0])
        bound = ad_pvals[bound_idx]
        if bound == 0.999:
            p = bound
        else:
            x1 = ad_quantiles[row, bound_idx-1]
            x2 = ad_quantiles[row, bound_idx]
            y1 = -np.log(ad_pvals[bound_idx-1])
            y2 = -np.log(ad_pvals[bound_idx])
            lfit = interp1d([x1, x2], [y1, y2])
            p = np.exp(-lfit(stat))
    return p

def forward_stop(pvals, signif):
    pvals_transformed = np.cumsum(-np.log(1-pvals))/np.arange(1,len(pvals)+1)
    kf = np.where(pvals_transformed <= signif)[0]
    if len(kf) == 0:
        stop = 0
    else:
        stop = max(kf) + 1
    if stop == pvals.size:
        stop -= 1
    return kf, stop

def main():
    print("Hello world")

if __name__ == "__main__":
    main()