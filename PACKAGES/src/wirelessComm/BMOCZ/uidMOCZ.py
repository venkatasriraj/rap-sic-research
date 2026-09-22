"""
Designing a class for Pilot-Zero based MOCZ to include UserId in the 
zero-constellation
Decoder mechanism:
1) Estimate the rotation using the pilotZero-1
2) Obtain the index of pilotZero-2
3) obtain msgSector Index and decode the messages similar to BMOCZ
    - estimate l_est (msgSector)
    - derotate the CFO corrected signal and decode the message
"""
import numpy as np
import math
from wirelessComm.BMOCZ import MOCZ

class UidMOCZ(MOCZ):

    def __init__(self, K, M):
        super().__init__(K, M)
        self.locations = int(K * M) 
        self.zero_geometry = self.codebook_con()
        self.pilotZero1 = [-1.25*self.R]
        self.pz2Rad = 1.75*self.R
        self.UIDbits = int(math.log2(K * M**2)) 

    def codebook_con(self):
        Ri, Ro = self.R**-1, self.R
        theta = 2 * np.pi / self.K
        return [(Ri*np.exp(1j*theta*k), Ro*np.exp(1j*theta*k)) for k in range(self.K)]

    def uId_Sectors(self, userId):
        # assert userId < self.locations * self.M
        uid1 = userId % self.locations
        msgSector = userId // self.locations
        pz2_i = uid1 // self.M
        pz2_j = uid1 % self.M
        return msgSector, pz2_i, pz2_j

    def sectors_uId(self, subSector, uId1): # pz2_i, pz2_j
        uid = int(subSector * self.locations + uId1)
        # if uid == 0 or uid == self.locations*self.M:
        #     return self.K * self.M**2
        return uid # + pz2_i*self.M + pz2_j

    def coeffCon(self, msg):
        userIdBits = msg[:self.UIDbits]
        userId = self.bin2dec(userIdBits)
        msgTx = msg[self.UIDbits:]
        msgSector, pz2_i, pz2_j = self.uId_Sectors(userId)
        pilotZeros = np.concatenate((self.pilotZero1, [ self.pz2Rad * np.exp(1j*self.theta_K*(pz2_i*self.M + pz2_j))]), axis=None)
        zeroSelection = [ self.zero_geometry[mk][msgTx[mk]] * np.exp(1j*self.theta_K*msgSector) for mk in range(self.K)]
        zeroSelection = np.concatenate((zeroSelection, pilotZeros), axis=None)
        return self.toeplitz_iterator(zeroSelection)

    def ffo_est(self, y, Q):
        Yo, Yi = self.fftCon(y, Q)
        min_q = {}
        for q in range(Q * self.M):
            sumZeros = 0
            for k in range( int(self.K) ):
                idx = (Q * self.M * k + q) % len(Yo)
                sumZeros += min( Yi[idx], Yo[idx] ) 
            min_q[q] = sumZeros
        q_est = min(min_q, key=min_q.get)
        msg_rx = ( 1 - np.sign( Yo[q_est::Q*self.M] - Yi[q_est::Q*self.M]) ) / 2
        return q_est//Q, np.asarray(msg_rx, dtype=np.int8)

    # BMOCZ singlePZDecoder based methods for better performance of the system
    def fftSig(self, y):
        y1 = y[:self.K]
        y2 = np.pad( y[self.K:], (0, self.K - len(y[self.K:])), mode='constant' )
        return y1 + y2

    def PZfftCon(self, y):
        y_ctr = np.conjugate(y[::-1])

        scaling_vec = self.R**np.arange(len(y))
        y_scaled = y * scaling_vec
        y_ctr_scaled = y_ctr * scaling_vec

        y_fft = self.fftSig(y_scaled) if self.M == 1 else y_scaled
        y_ctr_fft = self.fftSig(y_ctr_scaled) if self.M == 1 else y_ctr_scaled

        Y_eval = np.abs( np.fft.ifft(y_fft, n=self.K*self.M) )
        Y_ctr_eval = np.abs( np.fft.ifft(y_ctr_fft, n=self.K*self.M) )
        return Y_eval, Y_ctr_eval

    def msgDecoder(self, y):
        Yo, Yi = self.PZfftCon(y)
        SubSec = np.zeros(self.M)
        for m in range(self.M):
            sumZeros = 0
            for k in range(self.K):
                idx = k*self.M + m
                sumZeros += min( Yo[idx], Yi[idx] )
            SubSec[m] = sumZeros
        minSubSec = np.argmin(SubSec)
        msgDecoded = ( 1 - np.sign(Yo[minSubSec::self.M] - Yi[minSubSec::self.M]) ) / 2
        return minSubSec, msgDecoded

    def uIdDecoder(self, y, Q):
        # --STAGE - 1
        rotate_hat, _ = self.estRotation(y, Q, self.pilotZero1)
        rotation_hat = rotate_hat - np.angle(self.pilotZero1[0])
        rotationMatrix = np.diag( np.exp(-1j*rotation_hat)**np.flip(np.arange(len(y))) )
        y_cfoCorrected = y @ rotationMatrix
        msgSector_est, msg_rx = self.msgDecoder(y_cfoCorrected)
        # msgSector_est, msg_rx = self.ffo_est(y_cfoCorrected, Q)
        # msgSector_est, msg_rx = self.majorityVoteDecoder(y_cfoCorrected, Q)
        # --- STAGE - 2
        _, uid1_est = self.estRotation(y_cfoCorrected, Q, [self.pz2Rad])
        userId_est = self.sectors_uId(msgSector_est, uid1_est)
        uidBitsEst = self.dec2bin(userId_est)
        if len(uidBitsEst) != self.UIDbits:
            uidBitsEst = np.concatenate( ([0]*(self.UIDbits-len(uidBitsEst)), uidBitsEst), axis=None )
        msg_hat = np.concatenate((uidBitsEst, msg_rx), axis=None)
        return msg_hat, userId_est, rotation_hat

    def simulator(self, noIter, perParam, ch, Q=64):
        BER, PCR, PAPR, rotationEst, payloadEst, uIdEst = 0, 0, 0, 0, 0, 0
        blockLen = self.K + self.UIDbits
        for _ in range(noIter):
            rotation = np.random.uniform(0, 2*np.pi)
            msgTx = np.random.randint(0, 2, blockLen)
            userId = self.bin2dec(msgTx[:self.UIDbits])

            sigTx = self.coeffCon(msgTx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            sigRx = ch.transmit(sigTx, rotation)
            msgRx, userId_est, rotation_hat = self.uIdDecoder(sigRx, Q)
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += self.PAPR(msgTx)
            rotationEst += abs(rotation - rotation_hat) / rotation
            uIdEst += 0 if userId == userId_est else 1
            payloadEst += perParam.ber(msgRx[self.UIDbits:], msgTx[self.UIDbits:])
        return dict({
                'ber': BER / noIter,
                'pcr': PCR / noIter,
                'papr': PAPR / noIter,
                'rotationEst': rotationEst / noIter,
                'payloadEst': payloadEst/noIter,
                'uIdEst': uIdEst/noIter
        })