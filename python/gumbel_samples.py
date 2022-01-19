def gumbel_sample(M=10, K=1000, mu=0, beta=0.1):
    # mean = mu + 0.57721*beta, variance = (pi*beta)^2 / 6
    import numpy as np
    samples = np.empty([M,K])
    for m in range(M):
        s = np.sort(np.random.gumbel(mu, beta, K))
        samples[m,] = s
    return samples
    
    