import numpy as np
from .mocz import MOCZ

class BMOCZTransmitter(MOCZ):

    def __init__(self, K):
        super().__init__(K)
        self.zero_geometry = self.codebook_con()

    def codebook_con(self):
    
            Ri, Ro = self.R**(-1), self.R
            theta_k = (2 * np.pi)/self.K
    
            zero_cb = [( Ri * np.exp(1j*theta_k*k), Ro * np.exp(1j*theta_k*k) ) for k in range(self.K)]
            return zero_cb

    def zeroSelection(self, msg):
        return [self.zero_geometry[mk][msg[mk]] for mk in range(self.K)]
    
    def coeffCon(self, msg, singlePZ = []):
        zeros = self.zeroSelection(msg)
        zeros += singlePZ
        # print(f'\nZeroes selected wrt to message to be transmitted: {np.round(zeros, 6)}\n')
        return self.toeplitz_iterator(zeros)  # x = [ x0, x1, x2, ....., xK]

    # ZMS: Zero Marker Signal
    # def coeffConZM(self, msg):
    #     # Rzm = (( self.R + self.R**-1 )/2).astype(complex)
    #     zeros = [self.zero_geometry[mk][msg[mk]] for mk in range(self.K)]
    #     zeros += self.Rzm
    #     # zeros += self.Rpz
    #     x = self.toeplitz_iterator(zeros)
    #     # return x[::-1]
    #     return x

    # def coeffConSinglePZ(self, msg, singlePZ):
    #     zeros = [self.zero_geometry[mk][msg[mk]] for mk in range(self.K)]
    #     zeros += singlePZ
    #     x = self.toeplitz_iterator(zeros)
    #     # return x[::-1]
    #     return x