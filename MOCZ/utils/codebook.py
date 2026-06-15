import numpy as np

def codebook_con(R, K):

    Ri, Ro = R**(-1), R
    theta_k = (2 * np.pi)/K

    zero_cb = [( Ri * np.exp(1j*theta_k*k), Ro * np.exp(1j*theta_k*k) ) for k in range(K)]

    return zero_cb