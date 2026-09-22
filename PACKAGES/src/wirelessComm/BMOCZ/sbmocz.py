"""
Implementation of Smooshed-BMOCZ (SBMOCZ) scheme for CFO estimation
zeta = 0: Huffman BMOCZ
zeta > 0: Smooshed-BMOCZ
"""
import numpy as np
from .mocz import MOCZ 

class SBMOCZ(MOCZ):

    def __init__(self, K, zeta=1, Ns=256):
        super().__init__(K, M=1)
        self.zeta = zeta
        self.Ns = Ns
        self.theta = ( (2*np.pi-zeta)*np.arange(K)/K  + (2*np.pi + zeta*(K-1))/(2*K) )
        self.Rs = np.sqrt( 1 + np.sin( (2*np.pi-zeta) / (2 * K) ) )

    def coeffCon(self, msg):
        zeros = [self.Rs**(2*msg[k]-1)*np.exp(1j*self.theta[k]) for k in range(self.K)]
        return self.toeplitz_iterator(zeros)

    def rotationEst(self, y):
        phi_hat = (2*np.pi/self.Ns) * (np.argmax(np.abs( np.fft.fft(y, n=self.Ns) )))
        # Here we are not flipping the powers of e**(-jphi) when compared to 
        # Huffman BMCOZ
        y_corrected = y @ np.diag( np.exp(-1j*phi_hat)**(np.arange(len(y))) )
        return y_corrected, phi_hat

    def smooshedDecoder(self, y, Q=2):
        Y_eval, Y_ctr_eval = self.fftCon(y, Q)
        msgRx = (1 - np.sign( Y_eval[1::2] - Y_ctr_eval[1::2])) / 2
        return msgRx

    def simulator(self, noIter, perParam, ch):
        BER, PCR, PAPR, rotationEst = 0, 0, 0, 0
        for _ in range(noIter):
            msgTx = np.random.randint(0, 2, self.K)
            sigTx = self.coeffCon(msgTx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            rotation = np.random.uniform(0, 2*np.pi)
            sigRx = ch.transmit(sigTx, rotation)

            sigRx_corrected, rotation_hat = self.rotationEst(sigRx)
            msgRx = self.smooshedDecoder(sigRx_corrected)
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += self.PAPR(sigTx)
            rotationEst += np.abs(rotation-rotation_hat)/rotation
        return dict({
                'ber': BER/noIter,
                'pcr': PCR/noIter,
                'papr': PAPR/noIter,
                'rotationEst': rotationEst/noIter
        })