import numpy as np
import galois

def con_Parity(x, n, b, g):
    S = {}

    for i in range(1, b+1):
        S[i] = x**(n-i) % g

    S = dict(sorted(S.items(), reverse=True))

    p = []
    for i, poly in S.items():
        # print(i, poly)
        row = np.zeros(n-b)
        coeffs = poly.coeffs
        row[:len(coeffs)] = coeffs[::-1]
        p.append(row)
    
    P = np.array(p)
    # print(P)
    return P

def con_systematic(x, n, b, g):
    
    P = con_Parity(x, n, b, g)

    g_matrix = np.hstack((-P, np.eye(b))).astype(int)
    g_matrix %= 2

    h_matrix = np.hstack((np.eye(n-b), P.T)).astype(int)

    G = galois.GF2(g_matrix)
    H = galois.GF2(h_matrix)

    return G, H