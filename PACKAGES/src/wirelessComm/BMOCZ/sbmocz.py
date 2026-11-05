"""
Implementation of Smooshed-BMOCZ (SBMOCZ) scheme for CFO estimation
zeta = 0: Huffman BMOCZ
zeta > 0: Smooshed-BMOCZ

Smooshed decoder was written for Q=2 and it can work for Lh=2*K - (K+1)
"""
import numpy as np
from .mocz import MOCZ 

class SBMOCZ(MOCZ):

    def __init__(self, K, zeta=1, Ns=256, Rs=None):
        super().__init__(K, M=1)
        self.zeta = zeta
        self.Ns = Ns
        self.theta = ( (2*np.pi-zeta)*np.arange(K)/K  + (2*np.pi + zeta*(K-1))/(2*K) )
        # self.theta = (2*np.pi-zeta)*np.arange(K)/K
        self.R = np.sqrt( 1 + np.sin( (2*np.pi-zeta) / (2 * K) ) ) if Rs == None else Rs
        self.zero_geometry = self.codebook_con()
        self.zeroCollection = self.zeroSet()

    def codebook_con(self):
        Ri, Ro = self.R**-1, self.R
        zero_geometry = [(Ri*np.exp(1j*self.theta[k]), Ro*np.exp(1j*self.theta[k])) for k in range(self.K)]
        return zero_geometry

    # def zeroSelection(self, msg):   
    #     return [self.zero_geometry[k][msg[k]] for k in range(self.K)]
    #     # return [self.R**(2*msg[k]-1)*np.exp(1j*self.theta[k]) for k in range(self.K)]

    # def coeffCon(self, msg):
    #     return self.toeplitz_iterator(self.zeroSelection(msg))

    def rotationEst(self, y):
        phi_hat = (2*np.pi/self.Ns) * (np.argmax(np.abs( np.fft.fft(y, n=self.Ns) )))
        # Here we are not flipping the powers of e**(-jphi) when compared to 
        # Huffman BMCOZ
        y_corrected = y @ np.diag( np.exp(1j*phi_hat)**(np.arange(len(y))) )
        return y_corrected, phi_hat

    def smooshedDecoder(self, y, Q=1):
        Y_eval, Y_ctr_eval = self.fftCon(y, 2*Q)
        # msgRx = (1 - np.sign( Y_eval[1::2] - Y_ctr_eval[1::2])) / 2
        ind = np.arange(1, self.K*2, 2) * Q
        msgRx = (1 - np.sign( Y_eval[ind] - Y_ctr_eval[ind])) / 2
        return msgRx

    def simulator(self, noIter, perParam, ch, Energy='None', FLAG="AWGN", noiseFlag=True):
        BER, PCR, PAPR, rotationEst, rotation_hat = 0, 0, 0, 0, 1e-7
        for _ in range(noIter):
            msgTx = np.random.randint(0, 2, self.K)
            sigTx = self.coeffCon(msgTx)
            # sigPower = np.mean(np.abs(sigTx)**2)
            # sigTx /= np.sqrt(sigPower)
            l2_norm = np.linalg.norm(sigTx, ord=2)
            E = self.K+1 if Energy == 'None' else Energy
            sigTx = (sigTx / l2_norm) * np.sqrt(E) 

            rotation = np.random.uniform(0, 2*np.pi)
            if FLAG == "FS":
                sigRx, _ = ch.frequencySelective(sigTx, noiseFlag)
            elif FLAG == "CFO":
                sigRx = ch.CFO(sigTx, rotation, noiseFlag)
            elif FLAG == "AWGN":
                sigRx = ch.awgn(sigTx)
            elif FLAG == "FS_CFO":
                sigRx, _ = ch.frequencySelective(sigTx, noiseFlag, rotation)

            Q = int( np.ceil(len(sigRx)/self.K) )
            if FLAG == 'FS_CFO' or FLAG == 'CFO' or FLAG == 'BF_CFO':
                sigRxCorrected, rotation_hat = self.rotationEst(sigRx)
                sigRx = sigRxCorrected
            msgRx = self.smooshedDecoder(sigRx, Q=2)
            
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += self.PAPR(sigTx)
            mae = np.abs(rotation - rotation_hat) if np.abs(rotation - rotation_hat) < np.pi else 2*np.pi-np.abs(rotation - rotation_hat)
            rotationEst += mae/rotation
        return dict({
                'ber': BER/noIter,
                'pcr': PCR/noIter,
                'papr': PAPR/noIter,
                'rotationEst': rotationEst/noIter
        })

    def transmitterRAP(self, msgTx, ENERGY):
        sigTx = self.coeffCon(msgTx)         
        l2_norm = np.linalg.norm(sigTx, ord=2) 
        sigTx = (sigTx / l2_norm) * np.sqrt(ENERGY)
        return sigTx

    def receiverRAP(self, sigRx, CFO=False):
        rotation_hat = 0
        if CFO:
            sigRx, rotation_hat = self.rotationEst(sigRx, self.Q)
        msgRx = self.fftDizet(sigRx, self.Q) 
        return msgRx, rotation_hat    