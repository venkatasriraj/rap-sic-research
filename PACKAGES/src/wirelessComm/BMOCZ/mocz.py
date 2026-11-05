"""
For PAPR analysis for MOCZ, after applying fft when all phases align in 
 time-domain the peak will be observed. For average power in frequency domain, 
 instead we will be calculating in time domain using the parseval's power theorem.
 NOTE: How is PAPR defined for MOCZ? 
"""
import numpy as np
from scipy.linalg import toeplitz

class MOCZ:

    def __init__(self, K, M = 1, Q = 32):
        self.K = K
        self.R = np.sqrt(1 + np.sin(np.pi/K))
        self.M = M
        self.Q = Q
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
        # polynomial in the increasing power of x (polynomial degree) is transmitted
        return x[::-1]

    def fftCon(self, y, Q):# q='None'
        N_r = len(y)        
        # overSamFac = self.Q if q == 'None' else q
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

    def fftDizet(self, y, Q=2):
        Y_eval, Y_ctr_eval = self.fftCon(y, Q)
        message_received = ( 1 - np.sign( Y_eval[::Q] - Y_ctr_eval[::Q] ) ) / 2 
        return message_received.astype(int)
    #  ------ Method to etimate the rotation of zeros using single pilot placed in z-domain

    def estRotation(self, y, Q, singlePZ):
        lenSignal = len(y)
        N_fft = self.K * self.M * Q
        scaling_vec = np.abs(singlePZ[0]) ** np.arange(lenSignal)
        y_scaled = y * scaling_vec
        y_pad = np.pad(y_scaled, (0, N_fft - lenSignal), mode='constant')
        Y_singlePZ = np.abs( np.fft.ifft(y_pad) )
        subSector = np.argmin(Y_singlePZ)
        rotate_hat = ( np.pi * 2 * subSector / (N_fft) )
        return rotate_hat, subSector//Q

    # instead selecting the sub-sector which has minimum distance across all sub-sectors, 
    # we select the sub-sector independently across all sectors and select the sub-sector
    # which has maximum occurence
    
    # def majorityVoteDecoder(self, y, Q):
    #     Y_eval, Y_ctr_eval = self.fftCon(y, Q)
    #     min_q = np.zeros(Q*self.M, dtype=int)
    #     for k in range(self.K):
    #         min_dist, min_idx = np.inf, 0
    #         for q in range(Q * self.M):
    #             idx = (Q * self.M * k + q) % len(Y_eval)
    #             temp = min(Y_eval[idx], Y_ctr_eval[idx])
    #             if temp < min_dist:
    #                 min_dist = temp
    #                 min_idx = q
    #         min_q[min_idx] += 1
    #     q_est = np.argmax(min_q)
    #     msgDecoded = ( 1 - np.sign(Y_eval[q_est::Q*self.M] - Y_ctr_eval[q_est::Q*self.M]) )/2
    #     # if self.M == 1:
    #     #     return q_est, np.asarray(msgDecoded, dtype=np.int8)
    #     return q_est//Q, msgDecoded.astype(int) 

    
    @staticmethod
    def bin2dec(binData):
        if len(binData) == 0:
            return 0
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

    @staticmethod
    def DFT(x):
        X_dft = np.fft.fft(x)
        X_dftShifted = np.fft.fftshift(X_dft)
        # X_mag = np.abs(X_dftShifted)**2
        omega = np.fft.fftfreq(len(x), d=1)
        omega_shifted = np.fft.fftshift(omega)
        return X_dftShifted, omega_shifted

    # @staticmethod
    def DTFT(self, x, resolution = 1024, Mode='uc', omegaMode='pi'):
        rad = 1 if Mode == 'uc' else self.R if Mode == 'oc' else self.R**-1
        if omegaMode == 'pi':
            omega = np.linspace(-np.pi, np.pi, resolution)
        else:
            omega = np.linspace(0, np.pi*2, resolution)
        X_dtft = np.zeros(len(omega), dtype=complex)
        for n in range(len(x)):
            X_dtft += x[n] * rad**n * np.exp(1j * omega * n)
            # since we will be evaluating the polynomial on unit circle
            # which has positive powers of Z 
        return X_dtft, omega

    @staticmethod
    def AACF(x):
        x_ctr = np.conjugate(x[::-1])
        return np.convolve(x, x_ctr)

    def zeroSet(self):
        zero_0 = [self.zero_geometry[k][0] for k in range(self.K)]
        zero_1 = [self.zero_geometry[k][1] for k in range(self.K)]
        return np.append(zero_0, zero_1)

    def zeroSelection(self, msg):   
        return np.asarray([self.zero_geometry[k][msg[k]] for k in range(self.K)])

    def coeffConToep(self, msg):
        return self.toeplitz_iterator(self.zeroSelection(msg))

    def coeffCon(self, msg):
        zeros = self.zeroSelection(msg)
        k = len(zeros)
        N = k * 2
        z = np.exp(2j * np.pi * np.arange(N) / N)
        X_Z = np.prod( z[:, None] - zeros[None, :], axis=1 )
        return np.fft.fft(X_Z)[:k+1]/N  # increasing powers, monic