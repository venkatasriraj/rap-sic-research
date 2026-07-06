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

    def PZInteger(self, y):
        y0 = sum(y[::4])
        y1 = sum(y[1::4])
        y2 = sum(y[2::4])
        y3 = sum(y[3::4])
        y_fft = np.array([y0, y1, y2, y3])
        Y_pz = np.abs( np.fft.ifft(y_fft) )
        int_est = np.argmax(Y_pz)
        return int_est * self.K / 4

    def PZRotationCorrected(self, y, Q):
        y_pad = np.pad( y, (0, self.K * Q - len(y)), mode='constant' )
        Y_pz = np.abs( np.fft.ifft( y_pad ) )
        argMin = np.argmin(Y_pz)
        print(f"argMin: {argMin}")
        rotation_est = np.pi - ( 2 * np.pi * argMin / (self.K * Q) )
        print(f"Rotation Estimation in rad: {rotation_est}")
        rotationMatrix = np.diag( np.exp(-1j * rotation_est) ** np.arange(len(y))[::-1] )
        y_rotated = y @ rotationMatrix
        return y_rotated
    
    def PZDecodedMsg(self, y, Q):
        y_rotated = self.PZRotationCorrected(y, Q)
        y_1 = y_rotated[:self.K]
        y_2 = np.pad( y_rotated[self.K:], (0, self.K - len(y_rotated[self.K:])), mode='constant' )
        y_fft = y_1 + y_2
        msg_decoded = self.fftDizet(y_fft, 1)
        return msg_decoded

    
    def fftConPZ(self, y):
        y_1 = y[:self.K] 
        y_2 = np.pad( y[self.K:], (0, self.K - len(y[self.K:])), mode='constant' )
        y_fft = y_1 + y_2
        scaling_vec = np.abs(self.Rzm[0]) ** np.arange(self.K)
        y_scaled = y_fft * scaling_vec
        Y_pz = np.abs( np.fft.ifft(y_scaled) )
        min_q = {}
        sep = int(self.K / 4)
        for k in range(self.K):
            sumZeros = Y_pz[(k-sep)%self.K] + Y_pz[(k)] + Y_pz[(k+sep)%self.K]
            min_q[k] = sumZeros
        q_est = min(min_q, key=min_q.get)
        return (self.K // 2) - q_est

    def BLodd(self, y):
        # scaling_vec = np.abs(self.Rzm[0]) ** np.arange(len(y))
        # y_scaled = y * scaling_vec
        if self.K % 2 != 0:
            y_fft = np.pad( y, (0, self.K * 2 - len(y)), mode='constant')
        else:
            y_1 = y[:self.K]
            y_2 = np.pad( y[self.K:], (0, self.K - len(y[self.K:])), mode='constant' )
            y_fft = y_1 + y_2
        Y_pz = np.abs( np.fft.ifft(y_fft) )
        argMin = np.argmin(Y_pz)
        if self.K % 2 != 0:
            int_rotation_est = int( (self.K - argMin) / 2 )
        else:
            int_rotation_est = ( int(self.K / 2) - argMin )
            # print(f"Arg Min: {argMin}")
        return int_rotation_est

    @staticmethod
    def mae(sig_rx, sig_recon):
        return np.mean(np.abs(sig_rx - sig_recon))

    @staticmethod
    def per(msg_rx, msg_tx):
        if np.all( msg_rx == msg_tx ):
            return 1
        else:
            return 0