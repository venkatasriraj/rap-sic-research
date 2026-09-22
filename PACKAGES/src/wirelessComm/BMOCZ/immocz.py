"""
Implementation of IM-MOCZ
"""
import numpy as np
import math
from .mocz import MOCZ

class IMMOCZ(MOCZ):

    def __init__(self, K, M=1, PZradius=1.25):
        super().__init__(K, M)
        self.zero_geometry = self.codebook_con()
        self.pilot = [-PZradius*self.R]
        self.addBitsLen = int(math.log2(M))

    def codebook_con(self):
        Ri, Ro = self.R**-1, self.R
        theta = 2 * np.pi / self.K
        return [(Ri*np.exp(1j*theta*k), Ro*np.exp(1j*theta*k)) for k in range(self.K)]

    def coeffCon(self, msg):
        addBits = msg[:self.addBitsLen]
        subSector = self.bin2dec(addBits)
        msgTx = msg[self.addBitsLen:]
        zeroSelection = [self.zero_geometry[k][msgTx[k]]*np.exp(1j*self.theta_K*subSector) for k in range(self.K)]
        zeroSelection = np.concatenate((zeroSelection, self.pilot), axis=None)
        return self.toeplitz_iterator(zeroSelection)

    def ffoDecoder(self, y, Q):
        Yo, Yi = self.fftCon(y, Q)
        min_q = {}
        for q in range(Q * self.M):
            sumZeros = 0
            for k in range(self.K):
                idx = ( Q * self.M * k + q ) % len(Yo)
                sumZeros += min(Yi[idx], Yo[idx])
            min_q[q] = sumZeros
        q_est = min(min_q, key=min_q.get)
        msg_rx = ( 1 - np.sign( Yo[q_est::Q*self.M] - Yi[q_est::Q*self.M] ) )/2
        return q_est//Q, np.asarray(msg_rx, dtype=np.uint8)

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

    def imDecoder(self, y, Q):
        rotate_hat, mHat_pz = self.estRotation(y, Q, self.pilot)
        rotation_hat = rotate_hat - np.angle(self.pilot[0])
        rotationMatrix = np.diag( np.exp(-1j * rotation_hat * np.flip( np.arange(len(y)) ) ) )
        y_cfoCorrected = y @ rotationMatrix
        # msgSector_est, msg_rx = self.ffoDecoder(y_cfoCorrected, Q)
        msgSector_est, msg_rx = self.msgDecoder(y_cfoCorrected)
        if self.M == 1:
            return msg_rx, msgSector_est, rotation_hat
        addBitsEst = self.dec2bin(msgSector_est)
        if len(addBitsEst) != self.addBitsLen:
            addBitsEst = np.concatenate(([0]*(self.addBitsLen-len(addBitsEst)), addBitsEst), axis=None)
        msg_rx = np.concatenate((addBitsEst, msg_rx), axis=None)
        return msg_rx, msgSector_est, rotation_hat

    def simulator(self, noIter, perParam, ch, Q=64):
        BER, PCR, PAPR, rotationEst, sectorEst, payloadEst = 0, 0, 0, 0, 0, 0
        blockLen = self.K + self.addBitsLen
        for _ in range(noIter):
            rotation = np.random.uniform(0, 2*np.pi)
            msgTx = np.random.randint(0, 2, blockLen)
            sector = self.bin2dec(msgTx[:self.addBitsLen])

            sigTx = self.coeffCon(msgTx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            sigRx = ch.transmit(sigTx, rotation)
            msgRx, sector_hat, rotation_hat = self.imDecoder(sigRx, Q)
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += self.PAPR(sigTx)
            rotationEst += abs(rotation - rotation_hat) / rotation
            sectorEst += 1 if sector_hat != sector else 0
            payloadEst += perParam.ber(msgRx[self.addBitsLen:], msgTx[self.addBitsLen:])
        return dict({
                'ber': BER/noIter,
                'pcr': PCR/noIter,
                'papr': PAPR/noIter,
                'rotationEst': rotationEst/noIter,
                'sectorEst': sectorEst/noIter,
                'payloadEst': payloadEst/noIter
        })