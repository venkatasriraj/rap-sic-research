"""
t: Number of error bits to be corrected
if c_hat @ H.T != 0 for anyone of the shifts of the received codeword
after error correction and affine translation we will be considering the
last shift as the decoded message and estimated rotation as default
"""
import galois
import itertools
import numpy as np
from .bch import BCH

class ACPC(BCH):

    def __init__(self, m, t):
        super().__init__(m, t)
        self.B, self.k_outer = None, None
        self.G, self.H, self.H_bch, self.g_affine = self.matixCon()
        self.T_synd = self.conTsyndrome()

    def genPoly(self):
        bch_poly, factors_m = self.genBCHpoly()

        bch_poly_factors, _ = galois.factors(bch_poly)

        gin_choice = [i for i in factors_m[1:] if i not in bch_poly_factors]
        inner_poly = gin_choice[2]

        poly = inner_poly * bch_poly
        return poly, bch_poly

    def msgEncoding(self, msg):
        if not isinstance(msg, galois.GF2):
            msg = galois.GF2(msg)
        codeword = msg @ self.G + self.g_affine
        return codeword

    def msgDecoding(self, codeword):
        if not isinstance(codeword, galois.GF2):
            codeword = galois.GF2(codeword)
        
        synd = tuple( (codeword @ self.H_bch.T).tolist() )
        if synd not in self.T_synd:
            error_vec = galois.GF2.Zeros(len(codeword))
        else:
            error_vec = self.T_synd[synd]
        codeword_hat = codeword - error_vec
        B = self.n - self.H.shape[0]
        flag = False
        for i in range(self.n):
            c_hat = np.roll(codeword_hat, -i) - self.g_affine
            if np.all(c_hat @ self.H.T == 0):
                # l_est = self.n - i
                l_est = i
                msg_hat = c_hat[-B:]
                flag = True
                break
            # else:
            #     # l_est = self.n - i
            #     l_est = i
            #     msg_hat = c_hat[-B:]
        if not flag:
            l_est = 0
            msg_hat = codeword_hat[-B:]
        return msg_hat, l_est

    def matixCon(self):
        gen_polynomial, bch_gen_polynomial = self.genPoly()
        self.B = self.n - gen_polynomial.degree
        self.k_outer = self.n - bch_gen_polynomial.degree
        G, H = self.conSystematic(self.B, gen_polynomial)
        G_bch, H_bch = self.conSystematic(self.k_outer, bch_gen_polynomial)

        aff_trans = galois.GF2.Zeros(G_bch.shape[0])
        aff_trans[0] = 1
        g_affine = aff_trans @ G_bch
        return G, H, H_bch, g_affine