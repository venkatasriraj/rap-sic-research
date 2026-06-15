import numpy as np
from utils import codebook_con
from scipy.linalg import toeplitz

def toeplitz_iterator(zeros):

    for k in range(len(zeros)):
        if k == 0:
            c = np.array([[1, -zeros[k]]]).T   # (z-alpha)
        else:
            column = np.zeros(k+2, dtype=complex)
            column[0] = 1
            column[1] = -zeros[k]

            row = np.zeros(k+1, dtype=complex)
            row[0] = 1

            T = toeplitz(column, row)

            c = T @ c

    return c.flatten()

def coeffCon(msg):
    K = len(msg)
    R = np.sqrt( 1 + np.sin(np.pi/K) )
    theta_k = (np.pi * 2)/K

    zero_geometry = codebook_con(R, K)

    zeros = [zero_geometry[mk][msg[mk]] for mk in range(K)]
    # print(f'\nZeroes selected wrt to message to be transmitted: {np.round(zeros, 6)}\n')
    
    x = toeplitz_iterator(zeros)     # x = [ x0, x1, x2, ....., xK]
    
    return x[::-1]