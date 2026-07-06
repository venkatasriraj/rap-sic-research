import numpy as np
import random
import galois
from scipy.linalg import toeplitz

from BMOCZ import BiMOCZ

class BMOCZTransmitter(BiMOCZ):

    def __init__(self, K):
        super().__init__(K)
        self.zero_geometry = self.codebook_con()
        # self.msg = msg if msg else galois.GF2.Random(K)

    def toeplitz_iterator(self, zeros):
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

    def coeffCon(self, msg):
        zeros = [self.zero_geometry[mk][msg[mk]] for mk in range(self.K)]
        # print(f'\nZeroes selected wrt to message to be transmitted: {np.round(zeros, 6)}\n')
        x = self.toeplitz_iterator(zeros)        
        return x[::-1]          # x = [ x0, x1, x2, ....., xK]

    # ZMS: Zero Marker Signal
    def coeffConZM(self, msg):
        # Rzm = (( self.R + self.R**-1 )/2).astype(complex)
        zeros = [self.zero_geometry[mk][msg[mk]] for mk in range(self.K)]
        zeros += self.Rzm
        # zeros += self.Rpz
        x = self.toeplitz_iterator(zeros)
        return x[::-1]