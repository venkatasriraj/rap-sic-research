"""
For PAPR analysis for MOCZ, after applying fft when all phases align in 
 time-domain the peak will be observed. For average power in frequency domain, 
 instead we will be calculating in time domain using the parseval's power theorem. 
"""
import numpy as np

class BiMOCZ:

    def __init__(self, K):
        self.K = K
        self.R = np.sqrt(1 + np.sin(np.pi/K))
        self.theta_K = (np.pi * 2)/self.K

    def codebook_con(self):

        Ri, Ro = self.R**(-1), self.R
        theta_k = (2 * np.pi)/self.K

        zero_cb = [( Ri * np.exp(1j*theta_k*k), Ro * np.exp(1j*theta_k*k) ) for k in range(self.K)]
        return zero_cb

    def PAPR(self, signal):
        signal_max = np.abs( np.sum(signal) )
        signal_power = np.mean( np.abs(signal)**2 )
        papr = signal_max / signal_power
        return papr

    