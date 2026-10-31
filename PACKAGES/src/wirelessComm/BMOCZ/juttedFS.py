"""
implementation of jutted and BMOCZ for multipath channels
BMOCZ oversampled DiZeT decoder is used to estimate the fractional rotation offset
Jutted is used to estimate the integer rotation offset

Since the template of jutted is designed for single-tap channels and it varies
with channel taps. The template credibility decreases as channel taps increases.

"""

import numpy as np
from .mocz import MOCZ

class JUTTEDfs(MOCZ):

    def __init__(self, K, zeta, M=1, Rj=None, multiplePeak=False):
        super().__init__(K, M)
        self.zeta = zeta
        if Rj != None:
            self.R = Rj
        self.multiplePeak = multiplePeak
        self.mPeak = [0, 7, 17]
        self.zero_geometry = self.codebook_con() if not multiplePeak else self.codebookModified()
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

    def ffoEst(self, y, Q):
        Yo, Yi = self.fftCon(y, Q)
        min_q = np.zeros(Q)
        for q in range(Q):
            sumZeros = 0
            for k in range( self.K ):
                idx = (Q * k + q) % len(Yo)
                sumZeros += min( Yo[idx], Yi[idx] )
            min_q[q] = sumZeros
        q_est = np.argmin(min_q)
        theta_est = ( q_est / Q ) * self.theta_K
        return theta_est, q_est

    def ffoCorrection(self, y, Q):
        theta_hat, q_est = self.ffoEst(y, Q)
        M_theta = np.diag( np.exp(-1j * theta_hat * np.arange(len(y))[::-1] ) )
        return y @ M_theta, theta_hat

    def iroCorrection(self, y_ffo, Q):
        # now use the jutted zero to estimate the IRO
        Yo, Yi = self.fftCon(y_ffo, Q=Q)
        max_idx, maxMin = -1, 0
        for k in range(0, len(Yo), Q):
            minVal = min( Yo[k], Yi[k] )
            if minVal > maxMin:
                maxMin = minVal
                max_idx = k/Q
        phi_hat = self.theta_K * max_idx
        # print(phi_hat, max_idx)
        M_theta = np.diag( np.exp( -1j*phi_hat * np.arange(len(y_ffo))[::-1] ) )
        return y_ffo @ M_theta, phi_hat

    def CFOcorrection(self, y, Q):
        # fractional CFO estimation
        Yo, Yi = self.fftCon(y, Q)
        arr = np.zeros(Q)
        for q in range(Q):
            sumZeros = 0
            for k in range( self.K ):
                idx = (Q * k + q) 
                sumZeros += min( Yo[idx], Yi[idx] )
            arr[q] = sumZeros
        q_min = np.argmin(arr)
        theta_est = ( q_min / Q ) 

        # integer CFO estimation
        arr = np.zeros(self.K)
        for k in range(q_min, self.K*Q, Q):
            arr[k//Q] = min( Yi[k], Yo[k] )
        if not self.multiplePeak:
            u_est = np.argmax(arr)
        else:
            st = dict()
            idx = np.argsort(arr)[::-1][:3] # consider 2 sectors which have maximum distance
            idx = np.sort(idx)
            distance = idx[:, None] - idx
            # print(idx)
            Flag = False
            for i in range(3):
                if Flag:
                    break
                for j in range(i, 3):
                    if i != j:
                        if distance[j, i] == 7:
                            st[2] = idx[[i, j]]
                            u_est = idx[i]
                            Flag = True
                            break
                        elif distance[j, i] == 10:
                            st[3] = idx[[i, j]]
                            u_est = (idx[i] - 7) # % self.K
                            Flag = True
                            break
                        elif distance[j, i] == 17:
                            st[5] = idx[[i, j]]
                            u_est = idx[i]
                            Flag = True
                            break
                        else:
                            u_est = 0
            # print(u_est)
            # if len(st.keys()) == 0:
            #     u_est = 0
        phi_est = (u_est + theta_est) * self.theta_K
        # M_theta = np.diag( np.exp( -1j*phi_est * np.arange(len(y))[::-1] ) )
        M_theta = np.diag( np.exp( 1j*phi_est * np.arange(len(y)) ) )
        return y @ M_theta, phi_est

    def fftConZeta(self, y, Q):
        N_fft = Q * self.K * self.M

        scaling_vec = (self.R * self.zeta) ** np.arange(len(y))
        y_ctr = np.conjugate(y[::-1])

        y_scaled = y * scaling_vec
        y_ctr_scaled = y_ctr * scaling_vec  

        Y_eval = np.abs( np.fft.ifft(y_scaled, n=N_fft) )
        Y_ctr_eval = np.abs( np.fft.ifft(y_ctr_scaled, n=N_fft) )
        return Y_eval, Y_ctr_eval

    def iroCorrectionZeta(self, y_ffo, Q):
        Yo, Yi = self.fftConZeta(y_ffo, Q)
        min_K = np.zeros(self.K)
        for k in range(0, len(Yo), Q):
            min_K[k//Q] = min(Yo[k], Yi[k])
        phi_hat = self.theta_K * np.argmin(min_K)    
        M_theta = np.diag( np.exp( -1j*phi_hat * np.arange(len(y_ffo))[::-1] ) )
        return y_ffo @ M_theta, phi_hat

    def simulator(self, noIter, perParam, ch, Q=64, FLAG="FS", noiseFlag=True):
        BER, PCR, PAPR, rotationEst, rotation_hat = 0, 0, 0, 0, 1e-7
        for _ in range(noIter):
            msgTx =  np.random.randint(0, 2, self.K)
            sigTx = self.coeffCon(msgTx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            rotation = np.random.uniform(0, 2*np.pi)
            if FLAG == "BF":
                sigRx, _ = ch.blockFading(sigTx, noiseFlag)
            elif FLAG == "FS":
                sigRx, _ = ch.frequencySelective(sigTx, noiseFlag)
            elif FLAG == "CFO":
                sigRx = ch.CFO(sigTx, rotation, noiseFlag)
            elif FLAG == "AWGN":
                sigRx = ch.awgn(sigTx)
            elif FLAG == "FS_CFO":
                sigRx, _ = ch.frequencySelective(sigTx, noiseFlag, rotation)
            elif FLAG == "BF_CFO":
                sigRx, _ = ch.blockFading(sigTx, noiseFlag, rotation)
            
            if FLAG == 'FS_CFO' or FLAG == 'CFO' or FLAG == 'BF_CFO':
                # sigRx_ffo, theta_hat = self.ffoCorrection(sigRx, Q)
                # sigRxCorrected, phi_hat = self.iroCorrection(sigRx_ffo, Q)
                # rotation_hat = theta_hat + phi_hat
                # sigRx = sigRxCorrected
                sigRx, rotation_hat = self.CFOcorrection(sigRx, Q)
            msgRx = self.fftDizet(sigRx, Q)

            mae = np.min([np.abs(rotation-rotation_hat), np.abs(rotation - (2*np.pi-rotation_hat))])
            # mae = np.abs((2*np.pi-rotation_hat) % (2*np.pi) - rotation) # if np.abs(rotation - rotation_hat) > np.pi else 2*np.pi-np.abs(rotation - rotation_hat)
            # mae = np.abs(rotation - rotation_hat) if np.abs(rotation - rotation_hat) < np.pi else 2*np.pi-np.abs(rotation - rotation_hat)
            # print(f"rotation: {rotation}, EST: {rotation_hat}, MAE: {mae}")
            rotationEst += mae/rotation
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += self.PAPR(sigTx)
        return dict({
            'ber': BER/(noIter*self.K),
            'pcr': PCR/noIter,
            'papr': PAPR/noIter,
            'rotationEst': rotationEst/noIter
        }) 

    def simulatorBCH(self, noIter, perParam, ch, bch, Q=64, FLAG='FS', noiseFlag=True):
        BER, PCR, PAPR, rotationEst, rotation_hat = 0, 0, 0, 0, 1e-7
        for _ in range(noIter):
            msgTx =  np.random.randint(0, 2, bch.B)
            msgEnc = bch.msgEncoding(msgTx)

            sigTx = self.coeffCon(msgEnc)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            rotation = np.random.uniform(0, 2*np.pi)
            if FLAG == "FS":
                sigRx, _ = ch.frequencySelective(sigTx, noiseFlag)
            elif FLAG == "CFO":
                sigRx = ch.CFO(sigTx, rotation, noiseFlag)
            elif FLAG == "AWGN":
                sigRx = ch.awgn(sigTx)
            elif FLAG == "FS_CFO":
                sigRx, _ = ch.frequencySelective(sigTx, noiseFlag, rotation)

            if FLAG == 'FS_CFO' or FLAG == 'CFO':
                sigRx, rotation_hat = self.CFOcorrection(sigRx, Q)
            estCodeword = self.fftDizet(sigRx, Q)

            msgRx = bch.msgDecoding(estCodeword)
            mae = np.min([np.abs(rotation-rotation_hat), np.abs(rotation - (2*np.pi-rotation_hat))])
            rotationEst += mae/rotation
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += self.PAPR(sigTx)
        return dict({
            'ber': BER/(noIter*bch.B),
            'pcr': PCR/noIter,
            'papr': PAPR/noIter,
            'rotationEst': rotationEst/noIter
        })