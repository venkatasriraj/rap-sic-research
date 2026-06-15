import numpy as np
import galois
import itertools

def con_Tsyndrome(H, t=2):

    T_syn = {}

    code_size = H.shape[1]

    e = galois.GF2.Zeros(code_size)
    synd = tuple((e @ H.T).tolist())
    T_syn[synd] = e

    for i in range(code_size):
        e = galois.GF2.Zeros(code_size)
        e[i] = 1
        synd = tuple((e @ H.T).tolist())
        T_syn[synd] = e

    for i, j in itertools.combinations(range(code_size), t):
        e = galois.GF2.Zeros(code_size)
        e[i], e[j] = 1, 1
        synd = tuple((e @ H.T).tolist())
        T_syn[synd] = e

    return T_syn