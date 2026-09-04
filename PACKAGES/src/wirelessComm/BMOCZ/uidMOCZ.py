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
from wirelessComm.BMOCZ import MOCZ

class UidMOCZ(MOCZ):

    def __init__(self, K, M):
        super().__init__(K, M)
        self.locations = int(K * M / 2 )
        self.zero_geometry = self.codebook_con()
        self.pilotZero1 = [-1.25*self.R]
        self.pz2Rad = 1.75*self.R

    def codebook_con(self):
        Ri, Ro = self.R**-1, self.R
        theta = 2 * np.pi / self.K
        return [(Ri*np.exp(1j*theta*k), Ro*np.exp(1j*theta*k)) for k in range(self.K)]

    def uId_Sectors(self, userId):
        uid1 = userId % self.locations
        msgSector = userId // self.locations
        pz2_i = uid1 // self.M
        pz2_j = uid1 % self.M
        return msgSector, pz2_i, pz2_j

    def sectors_uId(self, subSector, uId1): # pz2_i, pz2_j
        if int(subSector * self.locations + uId1) == 0:
            return self.K * self.M**2 / 2
        return int(subSector * self.locations + uId1) # + pz2_i*self.M + pz2_j

    def coeffCon(self, msgTx, userId):
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
        # theta_est = ( q_est / Q ) * self.theta_K 
        msg_rx = ( 1 - np.sign( Yo[q_est::Q*self.M] - Yi[q_est::Q*self.M]) ) / 2
        return q_est//Q, np.asarray(msg_rx, dtype=np.int8)

    def uIdDecoder(self, y, Q):
        # --STAGE - 1
        rotate_hat, _ = self.estRotation(y, Q, self.pilotZero1)
        rotation_hat = rotate_hat - np.angle(self.pilotZero1[0])
        rotationMatrix = np.diag( np.exp(-1j*rotation_hat)**np.flip(np.arange(len(y))) )
        y_cfoCorrected = y @ rotationMatrix
        # --- STAGE - 2
        _, uid1_est = self.estRotation(y, Q, [self.pz2Rad])
        msgSector_est, msg_rx = self.ffo_est(y_cfoCorrected, Q)
        userId_est = self.sectors_uId(msgSector_est, uid1_est)
        return msg_rx, userId_est