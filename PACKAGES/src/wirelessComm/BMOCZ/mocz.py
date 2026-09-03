"""
For PAPR analysis for MOCZ, after applying fft when all phases align in 
 time-domain the peak will be observed. For average power in frequency domain, 
 instead we will be calculating in time domain using the parseval's power theorem.
 NOTE: How is PAPR defined for MOCZ? 
"""
import numpy as np
from scipy.linalg import toeplitz

class MOCZ:

    def __init__(self, K, M = 1):
        self.K = K
        self.R = np.sqrt(1 + np.sin(np.pi/K))
        self.M = M
        self.theta_K = (np.pi * 2) / (K * M)
        
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
        x = c.flatten()
        # polynomial in the increasing power of x is transmitted
        return x[::-1]

    def fftCon(self, y, Q):
        N_r = len(y)        
        N_fft = Q * self.K * self.M

        scaling_vec = self.R ** np.arange(N_r)
        y_ctr = np.conjugate(y[::-1])

        y_scaled = y * scaling_vec
        y_ctr_scaled = y_ctr * scaling_vec

        y_pad = np.pad(y_scaled, (0, N_fft - N_r), mode='constant')
        y_ctr_pad = np.pad(y_ctr_scaled, (0, N_fft - N_r), mode='constant')

        Y_eval = np.abs( np.fft.ifft(y_pad) )
        Y_ctr_eval = np.abs( np.fft.ifft(y_ctr_pad) )
        return Y_eval, Y_ctr_eval

    @staticmethod
    def PAPR(signal):
        # signal_max = np.abs( np.sum(signal) )
        signal_max = np.max( np.abs(signal) )**2
        signal_power = np.mean( np.abs(signal)**2 )
        papr = signal_max / signal_power
        # print(f"Signal Power: {signal_power}, PAPR: {papr}, Signal Energy: "
        #         f"{signal_power * len(signal)}, Max Abs Coeff: {np.max(np.abs(signal))}")
        return papr
    
    @staticmethod
    def bin2dec(binData):
        power2 = 2**np.arange(len(binData))[::-1]
        return int(np.sum(binData * power2))

    @staticmethod
    def dec2bin(decData):
        binData = [decData % 2]
        decData //= 2
        while decData>0:
            binData += [decData%2]
            decData //= 2
        return np.asarray(binData)[::-1]