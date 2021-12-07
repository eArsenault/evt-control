import numpy as np
import matplotlib.pyplot as plt
import os
import MC_functions as mc

def extract_n(file_name):
    l = file_name.split('_')
    return float(l[3][1:])

def process_data(files, n_vals, params):
    al, lamb, M = params
    data_stor = np.zeros((4,7,4))

    for f in files:
        index = np.where(n_vals == extract_n(f))[0][0]

        Y_total = np.load(os.path.join("data_mc2", f))
        for i in range(4):
            Y = Y_total[:,i]

            #since we use the expected value multiple places
            E = np.mean(Y)

            data_stor[0, index, i] = E + lamb * np.std(Y)
            data_stor[1, index, i] = np.max(Y)
            data_stor[2, index, i] = mc.estimator_cvar(Y, [al, M])
            data_stor[3, index, i] = E + lamb * np.sum(Y[Y >= E]) / M
    
    return data_stor

def get_ylabel(t):
    if t == 0:
        yl = "Mean+std"
    elif t == 1:
        yl = "Max"
    elif t == 2:
        yl = "CVaR(Y)"
    elif t == 3:
        yl = "Mean+mean above"

    return yl

def main():
    n = np.array([float(5*(i + 1)) for i in range(3,10)]) #np.array([float(5*(i + 1)) for i in range(20)] + [float(10*(i + 1) + 100) for i in range(10)])
    files = os.listdir(".\data_mc2")

    y = process_data(files, n, [0.0005, 1.0, 500000.0])

    #we plot the various evaluators in order: mean + var, max, cvar, and mean + fraction above
    fig, axs = plt.subplots(2,2)
    for i in [0,1]:
        for j in [0,1]:
            t = i + 2 * j
            axs[i,j].plot(n, y[t,:,0], label='mean')
            axs[i,j].plot(n, y[t,:,1], label='max') 
            axs[i,j].plot(n, y[t,:,2], label='CVaR') 
            axs[i,j].plot(n, y[t,:,3], label='EVT')
            #axs[i,j].set_ylim((7.5,25))
            axs[i,j].set_ylabel(get_ylabel(t))

    axs[0,0].legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()