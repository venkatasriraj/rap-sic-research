"""
Implementation of Smooshed-BMOCZ (SBMOCZ) scheme for CFO estimation
zeta = 0: Huffman BMOCZ
zeta > 0: Smooshed-BMOCZ
"""
import numpy as np
from .mocz import MOCZ 

class SBMOCZ(MOCZ):

    def __init__(self, K, zeta=1):
        super().__init__(K, M=1)
        self.zeta = zeta
        self.theta = ( (2*np.pi-zeta)*np.arange(K)/K  + (2*np.pi + zeta*(K-1))/(2*K) )
        self.Rs = np.sqrt( 1 + np.sin( (2*np.pi-zeta) / (2 * K) ) )

    def coeffCon(self, msg):
        zeros = [self.Rs**(2*msg[k]-1)*np.exp(1j*self.theta[k]) for k in range(self.K)]
        return self.toeplitz_iterator(zeros)

    def rotationEst(self, y, Ns=128):
        phi_hat = (2*np.pi/Ns) * (np.argmax(np.abs( np.fft.fft(y, n=Ns) )))
        # Here we are not flipping the powers of e**(-jphi) when compared to 
        # Huffman BMCOZ
        y_corrected = y @ np.diag( np.exp(-1j*phi_hat)**(np.arange(len(y))) )
        return y_corrected, phi_hat

    def msgDecoder(self, y, Q=2):
        Y_eval, Y_ctr_eval = self.fftCon(y, Q)
        msgRx = (1 - np.sign( Y_eval[1::2] - Y_ctr_eval[1::2])) / 2
        return msgRx