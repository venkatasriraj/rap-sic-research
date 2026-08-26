"""
t: Number of error bits to be corrected
if c_hat @ H.T != 0 for anyone of the shifts of the received codeword
after error correction and affine translation we will be considering the
last shift as the decoded message and estimated rotation as default
"""
import galois
import random
import itertools
import numpy as np

class ACPC:

    def __init__(self, m, t):
        self.m = m
        self.n = 2**m - 1
        self.t = t
        self.one = galois.GF2(1)
        self.GFm = galois.GF(2**self.m)
        self.x = galois.Poly.Identity(galois.GF2)
        self.B, self.k_outer = None, None
        self.G, self.H, self.H_bch, self.g_affine = self.matixCon()
        self.T_synd = self.con_Tsyndrome()

# generation of coset by visiting all the unvisited powers of the 
# polynomial variable x
    def con_coset(self):
        coset = {}
        leader = -1
        visited = set()

        for i in range(self.n):
            if i in visited:
                continue
            else:
                leader = i
                cos_lis = np.array([])
                while i not in cos_lis:
                    cos_lis = np.append(cos_lis, i).astype(int)
                    visited.add(i)
                    i *= 2
                    i %= self.n
                coset[leader] = cos_lis  
        return coset

    def gen_bch_poly(self):
        x_n = self.x**self.n + self.one

        factors_m, _ = galois.factors(x_n)

        cyclotomic_coset = self.con_coset()

        con_power = np.arange(1, 2*self.t+1)
        g_coset = np.array([])
        for i in con_power:
            leader = [keys for keys, arr in cyclotomic_coset.items() if i in arr ]
            g_coset = np.append(g_coset, leader).astype(int)
        g_bch_coset = np.unique(g_coset)

        primitive_element = self.GFm.primitive_element
        poly = self.one
        for i in g_bch_coset:
            poly *= self.GFm.minimal_poly(primitive_element**i)
        return poly, factors_m


    def gen_poly(self):
        bch_poly, factors_m = self.gen_bch_poly()

        bch_poly_factors, _ = galois.factors(bch_poly)

        gin_choice = [i for i in factors_m[1:] if i not in bch_poly_factors]
        inner_poly = gin_choice[2]

        poly = inner_poly * bch_poly
        return poly, bch_poly

    def con_Parity(self, info_len, gen_poly):
        S = {}
        for i in range(1, info_len+1):
            S[i] = self.x**(self.n - i) % gen_poly

        S = dict(sorted(S.items(), reverse=True))

        p = []
        for i, poly in S.items():
            row = np.zeros(self.n - info_len)
            coeffs = poly.coeffs
            row[:len(coeffs)] = coeffs[::-1]
            p.append(row)
        
        P = np.array(p)
        return P

    def con_systematic(self, info_len, gen_poly):       
        P = self.con_Parity(info_len, gen_poly)

        g_matrix = np.hstack((-P, np.eye(info_len))).astype(int)
        g_matrix %= 2

        h_matrix = np.hstack((np.eye(self.n - info_len), P.T)).astype(int)

        G = galois.GF2(g_matrix)
        H = galois.GF2(h_matrix)
        return G, H

    def con_Tsyndrome(self):
        T_syn = {}
        code_size = self.H.shape[1]

        e = galois.GF2.Zeros(code_size)
        synd = tuple((e @ self.H_bch.T).tolist())
        T_syn[synd] = e

        for i in range(code_size):
            e = galois.GF2.Zeros(code_size)
            e[i] = 1
            synd = tuple((e @ self.H_bch.T).tolist())
            T_syn[synd] = e

        for i, j in itertools.combinations(range(code_size), self.t):
            e = galois.GF2.Zeros(code_size)
            e[i], e[j] = 1, 1
            synd = tuple((e @ self.H_bch.T).tolist())
            T_syn[synd] = e
        return T_syn

    def msg_encoding(self, msg):
        if not isinstance(msg, galois.GF2):
            msg = galois.GF2(msg)
        codeword = msg @ self.G + self.g_affine
        return codeword

    def codeword_decoding(self, codeword):
        if not isinstance(codeword, galois.GF2):
            codeword = galois.GF2(codeword)
        
        # synd = tuple( (codeword @ self.H_bch.T).tolist() )
        synd = tuple( (codeword @ self.H_bch.T).tolist() )
        if synd not in self.T_synd:
            # print("The received codeword has more than 2 errors.")
            # return None
            error_vec = galois.GF2.Zeros(len(codeword))
        else:
            error_vec = self.T_synd[synd]
        codeword_hat = codeword - error_vec
        B = self.n - self.H.shape[0]
        for i in range(self.n):
            c_hat = np.roll(codeword_hat, -i) - self.g_affine
            if np.all(c_hat @ self.H.T == 0):
                l_est = self.n - i
                msg_hat = c_hat[-B:]
                break
            else:
                l_est = self.n - i
                msg_hat = c_hat[-B:]
        return msg_hat, l_est

    def matixCon(self):
        gen_polynomial, bch_gen_polynomial = self.gen_poly()
        self.B = self.n - gen_polynomial.degree
        self.k_outer = self.n - bch_gen_polynomial.degree
        G, H = self.con_systematic(self.B, gen_polynomial)
        G_bch, H_bch = self.con_systematic(self.k_outer, bch_gen_polynomial)

        aff_trans = galois.GF2.Zeros(G_bch.shape[0])
        aff_trans[0] = 1
        g_affine = aff_trans @ G_bch
        return G, H, H_bch, g_affine