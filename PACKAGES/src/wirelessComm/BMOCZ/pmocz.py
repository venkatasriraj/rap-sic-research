"""
Implementation of P-MOCZ where the given sector is further divided into subsectors. 
Depending on the bits received the zeros will be placed on one of the subsector.
NOTE: Can we do fractional rotation estimation for PMOCZ? NO.
OSF (Q): Over-sampling Factor
"""
import numpy as np
import math
from wirelessComm.BMOCZ import MOCZ

class PMOCZ(MOCZ):

    def __init__(self, K, M):
        super().__init__(K, M)
        self.bkLen = int(math.log2(M))
        self.ckLen = 1
        self.zero_geometry = self.codebook_con()

    def codebook_con(self):
        Ro, Ri = self.R, self.R**-1
        zero_cb = [ [(Ri*np.exp(1j*self.theta_K*(k*self.M+m)), Ro*np.exp(1j*self.theta_K*(k*self.M+m))) for m in range(self.M)] for k in range(self.K)]
        return zero_cb

    def zeroSelection(self, msg):
        blockLen = self.bkLen + self.ckLen
        assert len(msg) == self.K * (blockLen)
        msgBlocks = [msg[blockLen*k:blockLen*(k+1)] for k in range(self.K)]
        zeros = [ self.zero_geometry[k][self.bin2dec(msgBlocks[k][self.ckLen:self.ckLen+self.bkLen])][msgBlocks[k][0]] for k in range(self.K)]
        return zeros

    def coeffCon(self, msg, singlePZ=[]):
        zeros = self.zeroSelection(msg)
        zeros += singlePZ
        return self.toeplitz_iterator(zeros)

    def fftDiZet(self, y, Q):
        Y_eval, Y_ctr_eval = self.fftCon(y, Q)
        msgEst = np.array([])
        for k in range(self.K):
            Y_eval_sub = Y_eval[k*Q*self.M: (k+1)*Q*self.M : Q]
            Y_ctr_eval_sub = Y_ctr_eval[k*Q*self.M: (k+1)*Q*self.M: Q]
            Y_min, Y_ctr_min = min(Y_eval_sub), min(Y_ctr_eval_sub)
            circleFlag = 1 if Y_min < Y_ctr_min else 0
            subSectorEst = np.argmin(Y_eval_sub) if circleFlag == 1 else np.argmin(Y_ctr_eval_sub)
            subSectorBits = self.dec2bin(subSectorEst) if subSectorEst != 0 else np.zeros(self.bkLen)
            if len(subSectorBits) != self.bkLen:
                subSectorBits = np.concatenate(([0]*(self.bkLen - len(subSectorBits)), subSectorBits), axis=None)
            msgEst = np.concatenate((msgEst, circleFlag, subSectorBits), axis=None)
        return np.asarray(msgEst, dtype=np.int8)

    # -- Decoder for Pilot-Zero based PMOCZ
    def singlePZDecodedMsg(self, y, Q, singlePZ):
        rotate_hat, subSector = self.estRotation(y, Q, singlePZ)
        rotation_hat = rotate_hat - np.angle(singlePZ[0])
        rotationMatrix = np.diag( np.exp(-1j*rotation_hat) ** np.flip(np.arange(len(y))) ) 
        y_corrected = y  @ rotationMatrix
        return self.fftDiZet(y_corrected, Q), rotation_hat