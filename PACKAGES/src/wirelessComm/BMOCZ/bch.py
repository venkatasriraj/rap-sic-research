"""
Implementation of binary primitive BCH code 
primitive: n = 2**m - 1; m = positive integer
message length(B): n - mt
Code encoder matrix(G): B x K
    G: systematic matrix, generator matrix 
Code decoder matris(H): K x B
    H: check matrix, parity matrix
primitive element: 
"""
import galois
import itertools
import numpy as np

class BCH:

    def __init__(self, m, t):
        self.m = m
        self.n = 2**m - 1
        self.t = t
        self.one = galois.GF2(1)
        self.GFm = galois.GF(2**m)
        self.B = None
        self.x = galois.Poly.Identity(galois.GF2)
        self.G, self.H = self.matrixCon()
        self.Tsynd = self.conTsyndrome()

    def conCoset(self):
        coset = {}
        leader = -1
        visited = set()

        for i in range(self.n):
            if i not in visited:
                leader = i
                cos_lis = np.array([], dtype=int)
                while i not in cos_lis:
                    cos_lis = np.append(cos_lis, i)
                    visited.add(i)
                    i *= 2
                    i %= self.n
                coset[leader] = cos_lis
        return coset

    def genBCHpoly(self):
        primitive_element = self.GFm.primitive_element
        poly = self.one
        x_n = self.x**self.n + poly

        factors_m, _ = galois.factors(x_n)

        cyclotomic_coset = self.conCoset()
        con_power = np.arange(1, 2*self.t+1)
        g_coset = np.array([], dtype=int)
        for i in con_power:
            leader = [keys for keys, arr in cyclotomic_coset.items() if i in arr]
            g_coset = np.append(g_coset, leader)
        g_bch_coset = np.unique(g_coset)

        for i in g_bch_coset:
            poly *= self.GFm.minimal_poly(primitive_element**i)
        return poly, factors_m

    def conParity(self, infoLen, genPoly):
        S = {}
        for i in range(1, infoLen+1):
            S[i] = self.x**(self.n-i) % genPoly
        S = dict(sorted(S.items(), reverse=True))

        P = []
        for i, poly in S.items():
            row = np.zeros(self.n - infoLen)
            coeffs = poly.coeffs
            row[:len(coeffs)] = coeffs[::-1]
            P.append(row)
        return np.array(P)

    def conSystematic(self, infoLen, genPoly):
        P = self.conParity(infoLen, genPoly)

        gMatrix = np.hstack((-P, np.eye(infoLen))).astype(int) % 2

        hMatrix = np.hstack((np.eye(self.n - infoLen), P.T)).astype(int)
        G = galois.GF2(gMatrix)
        H = galois.GF2(hMatrix)
        return G, H

    def conTsyndrome(self):
        Tsyn = {}
        codeSize = self.H.shape[1]
        for T in range(self.t+1):
            for pos in itertools.combinations(range(codeSize), T):
                e = galois.GF2.Zeros(codeSize)
                for p in pos:
                    e[p] = 1
                synd = tuple(( e @ self.H.T ).tolist())
                Tsyn[synd] = e
        return Tsyn 

    def msgEncoding(self, msg):
        if not isinstance(msg, galois.GF2):
            msg = galois.GF2(msg)
        return msg @ self.G

    def msgDecoding(self, codeword):
        if not isinstance(codeword, galois.GF2):
            codeword = galois.GF2(codeword)
        synd = tuple( (codeword @ self.H.T).tolist() )
        if synd not in self.Tsynd:
            error_vec = galois.GF2.Zeros(len(codeword))
        else:
            error_vec = self.Tsynd[synd]
        codeword_hat = codeword - error_vec
        B = self.n - self.H.shape[0]
        # print(self.H.shape, self.G.shape)
        return codeword_hat[-self.B:]         

    def matrixCon(self):
        bchPoly, factors = self.genBCHpoly()
        self.B = self.n - bchPoly.degree
        G, H = self.conSystematic(self.B, bchPoly)
        return G, H