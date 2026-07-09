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
            for k in range( int(self.K) ):
                idx = (Q * k + q) % len(Yo)
                sumZeros += min( Yi[idx], Yo[idx] ) 
            min_q[q] = sumZeros
        q_est = min(min_q, key=min_q.get)
        # print(f'Estimatied sub-sector: {q_est}')
        theta_est = ( q_est / Q ) * self.theta_K
        # print(f'Estimated fractional frequency offset: {theta_est}') 
        return theta_est
    
    def ffoEstCor(self, y, Q):
        phi_hat = self.ffo_est(y, Q)
        M_theta = np.diag( np.exp(-1j*phi_hat) ** np.flip( np.arange(len(y)) ) )
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

    def PZRotationCorrected(self, y, Q):
        fftSize = self.K * Q 
        y_pad = np.pad( y, (0, fftSize - len(y)), mode='constant' )
        Y_pz = np.abs( np.fft.ifft( y_pad ) )
        #  --- MODIFICATION: estimatinf rotation using all 3 pz
        min_q = {}
        sep = int(np.ceil(fftSize/ 4))  # ---- when Block-length % 4 = 0
        # sep = int(np.ceil(fftSize / 4)) + 1       # for BL % 2 = 0
        # sep = Q
        for k in range(fftSize):
            sumZeros = Y_pz[(k-sep)%fftSize] + Y_pz[k] + Y_pz[(k+sep)%fftSize]
            min_q[k] = sumZeros
        q_est = min(min_q, key=min_q.get)
        argMin = np.argmin(Y_pz)
        # rotation_est = ( argMin - ( self.K * Q / 2 ) ) * 2 * np.pi / (self.K * Q)
        # ---- replaced argMin to q_est 
        rotation_est = ( q_est * 2 * np.pi / (self.K * Q) ) - np.pi
        rotationMatrix = np.diag( np.exp(-1j * rotation_est) ** np.flip( np.arange(len(y)) ) )
        y_rotated = y @ rotationMatrix
        return y_rotated
    
    def fftSig(self, y):
        y1 = y[:self.K]
        y2 = np.pad( y[self.K:], (0, self.K - len(y[self.K:])), mode='constant' )
        return y1 + y2

    def PZfftCon(self, y):
        y_ctr = np.conjugate(y[::-1])

        scaling_vec = self.R ** np.arange(len(y))
        y_scaled = y * scaling_vec
        y_ctr_scaled = y_ctr * scaling_vec

        y_fft = self.fftSig(y_scaled)
        y_ctr_fft = self.fftSig(y_ctr_scaled)

        Y_eval = np.abs( np.fft.ifft(y_fft) )
        Y_ctr_eval = np.abs( np.fft.ifft(y_ctr_fft) )
        return Y_eval, Y_ctr_eval
    
    def PZDecodedMsg(self, y, Q):
        y_rotated = self.PZRotationCorrected(y, Q)
        y_fft = self.fftSig(y_rotated)
        # msg_decoded = self.fftDizet(y_rotated, 2)
        Yo, Yi = self.PZfftCon(y_rotated)
        msg_received = ( 1 + np.sign( Yi - Yo ) ) / 2
        return msg_received
    
    def fftConPZ(self, y):
        y_fft = self.fftSig(y)
        scaling_vec = np.abs(self.Rzm[0]) ** np.arange(self.K)
        y_scaled = y_fft * scaling_vec
        Y_pz = np.abs( np.fft.ifft(y_scaled) )
        min_q = {}
        sep = int(np.ceil(self.K / 4))
        for k in range(self.K):
            sumZeros = Y_pz[(k-sep)%self.K] + Y_pz[(k)] + Y_pz[(k+sep)%self.K]
            min_q[k] = sumZeros
        q_est = min(min_q, key=min_q.get)
        return (self.K // 2) - q_est

    def BLodd(self, y):
        if self.K % 2 != 0:
            y_fft = np.pad( y, (0, self.K * 2 - len(y)), mode='constant')
        else:
            y_fft = self.fftSig(y)
        Y_pz = np.abs( np.fft.ifft(y_fft) )
        argMin = np.argmin(Y_pz)
        if self.K % 2 != 0:
            int_rotation_est = int( (self.K - argMin) / 2 )
        else:
            int_rotation_est = ( int(self.K / 2) - argMin )
        return int_rotation_est
    
    def intRotationEst(self, y):
        fftSize = self.K * 4
        y_fft = np.pad( y, (0, fftSize - len(y)), mode='constant' )
        Y_pz = np.abs( np.fft.ifft(y_fft) )
        sep = fftSize // 4
        min_q = {}
        for k in range(fftSize):
            min_q[k] = Y_pz[(k-sep)%fftSize] + Y_pz[k] + Y_pz[(k+sep)%fftSize]
        int_est = min(min_q, key=min_q.get)
        return fftSize//2 - int_est

    @staticmethod
    def mae(sig_rx, sig_recon):
        return np.mean(np.abs(sig_rx - sig_recon))

    @staticmethod
    def per(msg_rx, msg_tx):
        if np.all( msg_rx == msg_tx ):
            return 1
        else:
            return 0

    # ---- VERIFY THIS AND CHECK WHERE IT IS NOT WORKING
    def fftConZM(self, y):
        N_r = len(y)
        N_fft = self.K * 2
        # Rzm = ( (self.R + self.R**-1)/2 ).astype(complex)
        # scaling_vec = Rzm ** np.arange(N_r)
        # y_scaled = y * scaling_vec
        y_pad = np.pad(y, (0, N_fft - N_r), mode='constant')
        Y_zm = np.abs( np.fft.ifft(y_pad) )
        return Y_zm

    def ZMDetection(self, y):
        Y_zm = self.fftConZM(y)
        k_est =  np.argmin(Y_zm) // 2
        return k_est