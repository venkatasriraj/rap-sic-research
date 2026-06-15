import numpy as np
from utils import fftCon

def ffo_est(y, Q, K, R):
    
    Yo, Yi = fftCon(y, K, Q, R)
    theta_k = (2 * np.pi) / K
    min_q = {}
    for q in range(Q):
        sumZeros = 0
        for k in range(K):
            idx = (Q * k + q) % len(Yo)
            sumZeros += min( np.abs(Yi[idx]), np.abs(Yo[idx]) )
        min_q[q] = sumZeros
    q_est= np.argmin(min_q)
    # print(f'Estimatied sub-sector: {q_est}')
    theta_est = ( q_est / Q ) * theta_k
    # print(f'Estimated fractional frequency offset: {theta_est}') 
    return theta_est