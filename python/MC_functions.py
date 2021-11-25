import numpy as np
import time
from numba import jit

@jit(nopython=True)
def antiderivative(y, p):
    #computes the antiderivative for our moment estimator
    #p is array of parameters
    K, a, g, M, z = p
    if (g * (y - z) + a)**(1/g) == 0.0:
        #print('Fail')
        val = 100
    else:
        val = (K * a**(1/g) / (M * (g - 1))) * (y + a - g * z) / (g * (y - z) + a) ** (1/g)
    return val

@jit(nopython=True)
def cost(x, ub_k, lb_k, N):
    #K = [lb_k, ub_k]
    #N is the size of X as an 1D-array
    z = np.zeros(N)
    c = np.maximum(np.maximum(x - ub_k, lb_k - x), z)
    return c

@jit(nopython=True)
def dynamics(x, u, w, params):
    #computes x_{t+1} = f(x_t,u_t,w_t)
    a, b, nRP = params
    x_t1 = (a * x) + (1 - a) * (b - nRP * u) + w
    return x_t1

@jit(nopython=True)
def dynamics_snap(x, u, w, params, ub, lb):
    #ensures that x_{t+1} \in [lb, ub]
    x_t1 = dynamics(x, u, w, params)
    x_t1[x_t1 > ub] = ub
    x_t1[x_t1 < lb] = lb
    return x_t1

@jit(nopython=True)
def estimator_cvar(Z, eparams):
    al, n = eparams[0:2]
    var = np.quantile(Z, 1 - al)

    if al == 0:
        est = Z.max()
    else:
        #if Z < var, set it to var to Z - var = 0
        #then it won't contribute to expectation
        Z[Z < var] = var 
        est = var + (Z - var).sum() / (n * al)
    return est

@jit(nopython=True)
def estimator_max(Z, eparams):
    est = Z.max()
    return est

@jit(nopython=True)
def estimator_mean(Z, eparams):
    est = Z.mean()
    return est

@jit(nopython=True)
def estimator_moment(Z, eparams):
    al, n = eparams[0:2]

    X = np.sort(Z)
    var = np.quantile(X, 1 - al)
    i = np.abs(X - var).argmin() - 1

    #python is zero-indexed, so we subtract (i+1)
    K = n - i - 1.0
    z_mk = X[i]

    M1 = (1/K) * ((np.log(X[i:]) - np.log(X[i]))**1).sum()
    M2 = (1/K) * ((np.log(X[i:]) - np.log(X[i]))**2).sum()
    if M2 == 0:
        ga = 0.1
    elif (1 - (M1**2)/M2) == 0.0:
        ga = -1000
    else:    
        ga = M1 + 1 - (1/2) / (1 - (M1**2)/M2)

    a = z_mk * M1 * (1 - ga + M1)

    Z_max = 2.5
    aparams = np.array([K, a, ga, n, z_mk])
    if (ga == 0) or (ga == 1) or ((ga < 0) and (z_mk > -1/ga)) or (np.isnan(ga)):
        #print("option1")
        est = X.max()
    elif ((ga < 0) and (Z_max < -1/ga)) or (ga > 0): #H_ga defined for y in (0, 1/(max(0,-ga))
        #print("option2")
        y1 = z_mk
        y2 = Z_max
        est = antiderivative(y2,aparams) - antiderivative(y1,aparams)
    else:
        #print("option3")
        y1 = z_mk
        y2 = -1/ga
        est = antiderivative(y2,aparams) - antiderivative(y1,aparams)
    
    return est

@jit(nopython=True)
def mc_run(x_i, N, M, eparams, sparams, dparams):
    cost_arr = np.zeros((M,N))
    ub, lb, ub_k, lb_k, U_max, U_min, dU, std = sparams
    U = np.array([U_min + dU*j for j in range(int((U_max - U_min) / dU) + 1)])
    #print(U)
    #print(int(eparams[1]))
    #print(cost(x_i, ub_k, lb_k, 1))

    for k in range(M):
        #perform M trials
        total_cost = np.zeros(N)
        x = x_i*np.ones(N)

        for i in range(11):
            #evaluate current cost, add to total -> loop over the time horizon here
            total_cost = total_cost + cost(x, ub_k, lb_k, 1)
            u_min = np.zeros(N)
            Z_min = 100.0*np.ones(N)

            for j in range(int((U_max - U_min) / dU) + 1):
                #action selection, loop over possible actions
                u = U[j]

                #sample w_i n times, run through dynamics
                w_mc = np.array([np.random.normal(0, std) for w in range(int(eparams[1]))])
                for v in range(N):
                    #this whole loop can definitely be vectorized, to a degree
                    x_pmc = dynamics_snap(x[v], u, w_mc, dparams, ub, lb)

                    #use our 'simulator' to get n observations of c(x_{t+1})
                    Z = cost(x_pmc, ub_k, lb_k, int(eparams[1]))

                    #apply estimators to evaluate that action in 4 distinct ways
                    if v == 0:
                        Z_eval = estimator_mean(Z, eparams)
                    elif v == 1:
                        Z_eval = estimator_max(Z, eparams)
                    elif v == 2:
                        Z_eval = estimator_cvar(Z, eparams)
                    else:
                        Z_eval = estimator_moment(Z, eparams)
                    
                    #evaluate the running min
                    if Z_eval < Z_min[v]:
                        u_min[v] = u
                        Z_min[v] = Z_eval

            #update x according to environment, continue
            w = np.random.normal(0, std)
            for v in range(N):
                if v < N - 1:
                    val = dynamics_snap(x[v:(v + 1)], u_min[v], w, dparams, ub, lb)
                else:
                    val = dynamics_snap(x[v:], u_min[v], w, dparams, ub, lb)
                x[v] = val[0]
        
        #store the total cost of this trial
        cost_arr[k,:] = total_cost
    
    return cost_arr

def main():
    #define sysparams [ub, lb, ub_k, lb_k, U_max, U_min, dU]
    sparams = np.array([23.0, 18.0, 21.0, 20.0, 1.0, 0.0, 0.1, 1.0])

    #define our dynamics parameters
    dt = 5/60
    eta = 0.7
    R = 2.0
    P = 14.0
    C = 2.0
    dparams = np.array([np.exp(-dt/(C*R)), 32.0, eta*R*P ])

    #set the alpha
    al = 0.05

    #set the possible values for x, n to loop over (lower n are faster to run!)
    #note that in Python, {i in range(10)} = {0,1,2,...,9}
    x_vals = np.array([20.5])
    n_vals = np.array([float(5*(i + 1)) for i in range(20)] + [float(10*(i + 1) + 100) for i in range(10)])

    for x_init in x_vals:
        for n in n_vals:
            #define our estimator parameters (an array of all floats: we cast to int when necessary)
            #this is so it can be accelerated using numba, cannot have mixed-type arrays/lists
            eparams = np.array([al, n, 0.0, 0.0]) #al, n, est_code, t
            file_name = "..\..\data\mc_x" + str(x_init) + "_std" + str(sparams[-1]) +"_n" + str(int(eparams[1])) + "_al" + str(eparams[0])
            
            start = time.time()
            M = 100000
            cost_arr = mc_run(x_init, 4, M, eparams, sparams, dparams)
            end = time.time()
            np.save(file_name, cost_arr)
            
            print("Elapsed execution time [s]:", end - start)
    
if __name__ == "__main__":
    main()