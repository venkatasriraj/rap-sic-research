"""
Implementation of Jutted-BMOCZ(JBMOCZ) for timing offset based rotation estimation
zeta = 1: Huffman BMOCZ
zeta > 1: jutted-BMOCZ
NOTE: the template vector is right circular shifted for each column vector to 
generate the Template Matrix. So, each column has a phase shift of -(2*pi/Ns). 
We are detecting the rotation si such that si = phi.
"""
import numpy as np
from scipy.linalg import toeplitz
from .mocz import MOCZ

class JBMOCZ(MOCZ):

    def __init__(self, K, zeta=1, Ns=1024):
        super().__init__(K, M=1)
        self.zeta = zeta
        self.Ns = Ns
        self.zero_geometry = self.codebook_con()
        self.template = self.templateCon()

    def codebook_con(self):
        Ri, Ro = self.R**-1, self.R
        zero_geometry = [(Ri*self.zeta**-1, Ro*self.zeta) if k == 0 else (Ri*np.exp(1j*self.theta_K*k), Ro*np.exp(1j*self.theta_K*k)) for k in range(self.K)]
        return zero_geometry

    def coeffCon(self, msg):
        zeros = [self.zero_geometry[k][msg[k]] for k in range(self.K)]
        return self.toeplitz_iterator(zeros)

    def templateCon(self):
        msg = np.ones(self.K, dtype=np.uint8)
        coeff = self.coeffCon(msg)
        return abs(np.fft.fft(coeff, n=self.Ns))

    def rotationEst(self, y):
        row = np.concatenate((self.template[0], self.template[:0:-1]), axis=None)
        templateMatrix = toeplitz(self.template, row)
        rxTemplate = abs( np.fft.fft(y, n=self.Ns) )
        innerPdt = rxTemplate @ templateMatrix
        indx = np.argmax(innerPdt)
        phi_hat = (2 * np.pi * indx / self.Ns) 
        y_corrected = y @ np.diag( np.exp(-1j * phi_hat * ( np.arange(len(y)) ) ) )
        return y_corrected, abs(phi_hat)

    # JBMOCZ Simulator
    def simulator(self, noIter, perParam, ch):
        BER, PCR, PAPR, rotationEst = 0, 0, 0, 0
        for _ in range(noIter):
            msgTx = np.random.randint(0, 2, self.K)
            sigTx = self.coeffCon(msgTx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            rotation = np.random.uniform(0, 2*np.pi)
            sigRx = ch.transmit(sigTx, rotation)

            sigRxCorrected, rotation_hat = self.rotationEst(sigRx)
            msgRx = self.fftDizet(sigRxCorrected)

            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += self.PAPR(sigTx)
            rotationEst += np.abs(rotation_hat - rotation) / rotation
        return dict({
                    'ber': BER/noIter,
                    'pcr': PCR / noIter,
                    'papr': PAPR / noIter,
                    'rotationEst': rotationEst / noIter
        })