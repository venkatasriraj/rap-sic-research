"""
Implementation of Jutted-BMOCZ(JBMOCZ) for timing offset based rotation estimation
zeta = 1: Huffman BMOCZ
zeta > 1: jutted-BMOCZ
"""
import numpy as np
from scipy.linalg import toeplitz
from .mocz import MOCZ

class JBMOCZ(MOCZ):

    def __init__(self, K, zeta=1, Ns=128):
        super().__init__(K, M=1)
        self.zeta = zeta
        self.Ns = Ns
        self.zero_geometry = self.codebook_con()
        self.template = self.templateCon()

    def codebook_con(self):
        Ri, Ro = self.R**-1, self.R
        zero_geometry = [(Ri*self.zeta**-1, Ro*self.zeta) if k == 0 else (Ri*self.theta_K**k, Ro*self.theta_K**k) for k in range(self.K)]
        return zero_geometry

    def coeffCon(self, msg):
        zeros = [self.zero_geometry[k][msg[k]] for k in range(self.K)]
        coeff = self.toeplitz_iterator(zeros)
        P = self.K+1
        # - dft based coeffiecients
        coeff_dtft = coeff * np.exp(1j*self.theta_K*np.arange(P))
        # coeff_dft = self.theta_K**np.flip( np.arange(len(coeff)) ) * coeff
        # coeff_ft = np.asarray([ np.dot(coeff, np.exp(1j*self.theta_K*p)**np.flip(np.arange(P))) for p in range(P) ])
        # print(x, np.shape(x))
        coeff_dft = np.fft.fft(coeff_dtft) / P
        signalPower = np.mean(np.abs(coeff_dft)**2)
        return coeff_dft/np.sqrt(signalPower)

    def templateCon(self):
        # zeros = [self.zero_geometry[k][0] for k in range(self.K)]
        msg = np.ones(self.K, dtype=np.uint8)
        coeff_dft = self.coeffCon(msg)
        return abs(np.fft.fft(coeff_dft, n=self.Ns))

    def rotationEst(self, y):
        row = np.concatenate((self.template[0], self.template[:0:-1]), axis=None)
        templateMatrix = toeplitz(self.template, row)
        rxTemplate = abs( np.fft.fft(y, n=self.Ns) )
        innerPdt = rxTemplate @ templateMatrix 
        indx = np.argmax(innerPdt)
        phi_hat = 2 * np.pi * (indx-1) / self.Ns
        y_corrected = np.diag(np.exp(-1j * phi_hat)**( np.flip(np.arange(len(y))) )) @ y
        return y_corrected, phi_hat

    def fftDiZeT(self, y, Q=2):
        Y_eval, Y_ctr_eval = self.fftCon(y, Q)
        msgRx = (1 - np.sign( Y_eval[::Q] - Y_ctr_eval[::Q])) / 2
        return msgRx.astype(int)