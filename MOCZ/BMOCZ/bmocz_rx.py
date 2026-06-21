import numpy as np
import itertools
import random
import galois

from .bmocz import BiMOCZ

class BMOCZReceiver(BiMOCZ):

    def __init__(self, K):
        super().__init__(K)

    def fftCon(self, y, Q):
        N_r = len(y)        
        N_fft = Q * self.K

        scaling_vec = self.R ** np.arange(N_r)
        y_ctr = np.conjugate(y[::-1])

        y_scaled = y * scaling_vec
        y_ctr_scaled = y_ctr * scaling_vec

        y_pad = np.pad(y_scaled, (0, N_fft - N_r), mode='constant')
        y_ctr_pad = np.pad(y_ctr_scaled, (0, N_fft - N_r), mode='constant')

        Y_eval = np.abs( np.fft.ifft(y_pad) )
        Y_ctr_eval = np.abs( np.fft.ifft(y_ctr_pad) )
        return Y_eval, Y_ctr_eval

    def ffo_est(self, y, Q):
        Yo, Yi = self.fftCon(y, Q)
        min_q = {}
        for q in range(Q):
            sumZeros = 0
            for k in range(self.K):
                idx = (Q * k + q) % len(Yo)
                sumZeros += min( np.abs(Yi[idx]), np.abs(Yo[idx]) )
            min_q[q] = sumZeros
        # print(min_q)
        #   ---- invalid use of np.argmin on a dict
        # q_est= np.argmin(min_q)
        q_est = min(min_q, key=min_q.get)
        # print(f'Estimatied sub-sector: {q_est}')
        theta_est = ( q_est / Q ) * self.theta_K
        # print(f'Estimated fractional frequency offset: {theta_est}') 
        return theta_est
    
    def ffoEstCor(self, y, Q):
        phi_hat = self.ffo_est(y, Q)
        M_theta = np.diag( np.exp(-1j*phi_hat) ** np.flip( np.arange(len(y)) ) )
        # print(M_theta)
        y_ffo = y @ M_theta
        return y_ffo


    def fftDizet(self, y, Q):
        Y_eval, Y_ctr_eval = self.fftCon(y, Q)
        message_received = ( 1 - np.sign( Y_eval[::Q] - Y_ctr_eval[::Q] ) ) / 2 
        return message_received.astype(int)
    
    @staticmethod
    def ber(msg_hat, msg):
        ber = np.mean(msg_hat != msg)
        return ber

    def fftConZM(self, y):
        N_r = len(y)
        N_fft = self.K * 2
        Rzm = ( (self.R + self.R**-1)/2 ).astype(complex)
        scaling_vec = Rzm ** np.arange(N_r)
        y_scaled = y * scaling_vec
        y_pad = np.pad(y_scaled, (0, N_fft - N_r), mode='constant')
        Y_zm = np.abs( np.fft.ifft(y_pad) )
        return Y_zm

    def ZMDetection(self, y):
        Y_zm = self.fftConZM(y)
        # print(Y_zm)

        # gives the integer phase offset
        k_est =  np.argmin(Y_zm) // 2
        # l_est = q_est % Q 
        # # to get sector to which it is rotated
        # k_est = q_est // Q
        # print(f"Integer Estimate: {k_est}, Fractional Estimate: {l_est}")

        # print(f"Sector estimation using ZMDetection: {k_est}")
        return k_est