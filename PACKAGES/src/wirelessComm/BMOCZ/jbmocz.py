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

    def __init__(self, K, zeta=1, Ns=1024, Rj=None, mPeak=[0]):
        super().__init__(K, M=1)
        self.zeta = zeta
        self.Ns = Ns
        self.mPeak = mPeak
        if Rj != None:
            self.R = Rj
        self.zero_geometry = self.codebook_con() if len(mPeak) == 1 else self.codebookModified()
        self.template = self.templateCon()
        self.zeroCollection = self.zeroSet()

    def codebook_con(self):
        Ri, Ro = self.R**-1, self.R
        zero_geometry = [(Ri*self.zeta**-1, Ro*self.zeta) if k == 0 else (Ri*np.exp(1j*self.theta_K*k), Ro*np.exp(1j*self.theta_K*k)) for k in range(self.K)]
        return zero_geometry

    def codebookModified(self):
        Ri, Ro = self.R**-1, self.R
        zero_geometry = []
        for k in range(self.K):
            if k in self.mPeak:
                zero_geometry.append( ( Ri*self.zeta**-1 * np.exp(1j*self.theta_K*k), Ro*self.zeta*np.exp(1j*self.theta_K*k) ) )
            else:
                zero_geometry.append( (Ri*np.exp(1j*self.theta_K*k), Ro*np.exp(1j*self.theta_K*k)) )
        return zero_geometry

    # def zeroSelection(self, msg):
    #     return [self.zero_geometry[k][msg[k]] for k in range(self.K)]

    # def coeffCon(self, msg):
    #     return self.toeplitz_iterator(self.zeroSelection(msg))

    def templateCon(self):
        msg = np.ones(self.K, dtype=np.uint8)
        coeff = self.coeffCon(msg)
        template, _ = self.DTFT(coeff, resolution=self.Ns, omegaMode='zero')
        return np.abs(template) # /np.max(np.abs(template))
        # ---- preiviously used 
        # template = abs(np.fft.fft(coeff, n=self.Ns))
        # return template/np.max(template)

    def rotationEst(self, y):
        row = np.concatenate((self.template[0], self.template[:0:-1]), axis=None)
        templateMatrix = toeplitz(self.template, row)
        # rxTemplate = abs( np.fft.fft(y, n=self.Ns) )
        # rxTemplate /= np.max(rxTemplate)
        rxTemplate, _ = self.DTFT(y, resolution=self.Ns, omegaMode='zero')
        rxTemplate = np.abs(rxTemplate) # /np.max(np.abs(rxTemplate))
        innerPdt = rxTemplate @ templateMatrix
        indx = np.argmax(innerPdt)
        phi_hat = (2 * np.pi * indx / self.Ns) 
        y_corrected = y @ np.diag( np.exp(1j * phi_hat * ( np.arange(len(y)) ) ) )
        return y_corrected, phi_hat # abs(phi_hat)

    # JBMOCZ Simulator
    def simulator(self, noIter, perParam, ch, Energy='None', FLAG="FS", noiseFlag=True, errorAnalysis=False):
        if not errorAnalysis:
            BER, PCR, PAPR, rotationEst, rotation_hat = 0, 0, 0, 0, 1e-7
        else:
            BER, PCR = 0, 0
            errorDist, errorPosition = np.zeros(self.K+2, dtype=int), np.zeros(self.K+1, dtype=int)
        E = self.K+1 if Energy == 'None' else Energy
        Q = int( np.ceil( (self.K+ch.taps) / self.K) )
        for _ in range(noIter):
            msgTx = np.random.randint(0, 2, self.K)
            sigTx = self.coeffCon(msgTx)
            l2_norm = np.linalg.norm(sigTx, ord=2) # energy = l2_norm**2
            sigTx = (sigTx / l2_norm) * np.sqrt(E) 

            rotation = np.random.uniform(0, 2*np.pi)
            if FLAG == "FS":
                sigRx, _ = ch.frequencySelective(sigTx, noiseFlag)
            elif FLAG == "CFO":
                sigRx = ch.CFO(sigTx, rotation, noiseFlag)
                # corrected signal
                sigRx, rotation_hat = self.rotationEst(sigRx)
            elif FLAG == "AWGN":
                sigRx = ch.awgn(sigTx)
            elif FLAG == "FS_CFO":
                sigRx, _ = ch.frequencySelective(sigTx, noiseFlag, rotation)
                # corrected signal
                sigRx, rotation_hat = self.rotationEst(sigRx)
            # Q = int( np.ceil(len(sigRx)/self.K) )
            # if FLAG == 'FS_CFO' or FLAG == 'CFO':
            #     sigRx, rotation_hat = self.rotationEst(sigRx)
            msgRx = self.fftDizet(sigRx, Q)
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            if not errorAnalysis:
                PAPR += perParam.PAPR(sigTx)
                mae = np.min([abs(rotation - rotation_hat), abs(rotation - (2*np.pi-rotation_hat))])
                rotationEst += mae/rotation
            else:
                errorDist[perParam.bitError(msgRx, msgTx)] += 1
                errorPosition[perParam.errorPos(msgRx, msgTx)] += 1
        if not errorAnalysis:
            return dict({
                    # 'ber': BER/(noIter * self.K),
                    'ber': BER/noIter,
                    'pcr': PCR / noIter,
                    'papr': PAPR / noIter,
                    'rotationEst': rotationEst / noIter
            })
        else:
            return dict({
                'ber': BER/noIter,
                'pcr': PCR/noIter,
                'errorDist': errorDist,
                'errorPosition': errorPosition
            })

    def transmitterRAP(self, msgTx, ENERGY):
        sigTx = self.coeffCon(msgTx)         
        l2_norm = np.linalg.norm(sigTx, ord=2) 
        sigTx = (sigTx / l2_norm) * np.sqrt(ENERGY)
        return sigTx

    def receiverRAP(self, sigRx, CFO=False):
        rotation_hat = 0
        if CFO:
            sigRx, rotation_hat = self.rotationEst(sigRx)
        msgRx = self.fftDizet(sigRx, Q=2) 
        return msgRx, rotation_hat    