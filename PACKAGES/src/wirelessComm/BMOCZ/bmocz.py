import numpy as np
from .mocz import MOCZ

class BMOCZ(MOCZ):

    def __init__(self, K):
        super().__init__(K)
        self.zero_geometry = self.codebook_con()

    def codebook_con(self):
        Ri, Ro = self.R**(-1), self.R
        zero_cb = [( Ri * np.exp(1j*self.theta_K*k), Ro * np.exp(1j*self.theta_K*k) ) for k in range(self.K)]
        return zero_cb

    def zeroSelection(self, msg):
        return [self.zero_geometry[mk][msg[mk]] for mk in range(self.K)]
    
    def coeffCon(self, msg, singlePZ = []):
        zeros = self.zeroSelection(msg)
        zeros += singlePZ
        # print(f'\nZeroes selected wrt to message to be transmitted: {np.round(zeros, 6)}\n')
        return self.toeplitz_iterator(zeros)  # x = [ x0, x1, x2, ....., xK]

    ## ---- Receiver methods starting from here

    def ffo_est(self, y, Q):
        Yo, Yi = self.fftCon(y, Q)
        min_q = {}
        for q in range(Q):
            sumZeros = 0
            for k in range( int(self.K) ):
                idx = (Q * k + q) % len(Yo)
                sumZeros += min( Yi[idx], Yo[idx] ) 
            min_q[q] = sumZeros
        q_est = min(min_q, key=min_q.get)
        # print(f'Estimatied sub-sector: {q_est}')
        theta_est = ( q_est / Q ) * self.theta_K
        # print(f'Estimated fractional frequency offset: {theta_est}') 
        return theta_est, q_est
    
    def ffoEstCor(self, y, Q):
        phi_hat, q_est = self.ffo_est(y, Q)
        M_theta = np.diag( np.exp(-1j*phi_hat) ** np.flip( np.arange(len(y)) ) )
        y_ffo = y @ M_theta
        return y_ffo

    #--- Conclusion of MOCZ Decoder 
    # -- Beginning of MOCZ Pilot-Zero Decoder

    def fftSig(self, y):
        y1 = y[:self.K]
        y2 = np.pad( y[self.K:], (0, self.K - len(y[self.K:])), mode='constant' )
        return y1 + y2

    def PZfftCon(self, y):
        y_ctr = np.conjugate(y[::-1])

        scaling_vec = self.R ** np.arange(len(y))
        y_scaled = y * scaling_vec
        y_ctr_scaled = y_ctr * scaling_vec

        y_fft = self.fftSig(y_scaled)
        y_ctr_fft = self.fftSig(y_ctr_scaled)

        Y_eval = np.abs( np.fft.ifft(y_fft) )
        Y_ctr_eval = np.abs( np.fft.ifft(y_ctr_fft) )
        return Y_eval, Y_ctr_eval

    def singlePZDecodedMsg(self, y, Q, singlePZ):
        rotate_hat, subSector = self.estRotation(y, Q, singlePZ) 
        rotation_hat = rotate_hat - np.angle(singlePZ[0]) # if rotate_hat >= 0 else (2*np.pi + rotate_hat)
        rotationMatrix = np.diag( np.exp(-1j*rotation_hat) ** np.flip(np.arange(len(y))) )
        y_corrected = y @ rotationMatrix
        Yo, Yi = self.PZfftCon(y_corrected)
        msgDecoded = ( 1 + np.sign(Yi - Yo) ) / 2
        return msgDecoded.astype(int), rotate_hat

    # def pilotFROdecoder(self, y, Q):
    #     Y_eval, Y_ctr_eval = self.fftCon(y, Q)
    #     min_q = {}
    #     for q in range(Q):
    #         sumZero = 0
    #         for k in range(self.K):
    #             idx = (Q * k + q) % len(Y_eval)
    #             sumZero += min(Y_eval[idx], Y_ctr_eval[idx])
    #         min_q[q] = sumZero
    #     q_est = min(min_q, key=min_q.get)
    #     msgDecoded = ( 1 + np.sign(Y_ctr_eval[q_est::Q] - Y_eval[q_est::Q]) ) / 2
    #     # msgDecoded = ( 1 - np.sign(Y_eval[::Q] - Y_ctr_eval[::Q]) )/2
    #     return msgDecoded.astype(int)

    # def majorityVoteDecoder(self, y, Q):
    #     Y_eval, Y_ctr_eval = self.fftCon(y, Q)
    #     min_q = np.zeros(Q, dtype=int)
    #     for k in range(self.K):
    #         min_dist, min_idx = np.inf, 0
    #         for q in range(Q):
    #             idx = (Q * k + q) % len(Y_eval)
    #             temp = min(Y_eval[idx], Y_ctr_eval[idx])
    #             if temp < min_dist:
    #                 min_dist = temp
    #                 min_idx = q
    #         min_q[min_idx] += 1
    #     q_est = np.argmax(min_q)
    #     msgDecoded = ( 1 - np.sign(Y_eval[q_est::Q] - Y_ctr_eval[q_est::Q]) )/2
    #     return msgDecoded

    # def pilotDecoder_MV(self, y, Q, singlePZ):
    #     rotate_hat, subSector = self.estRotation(y, Q, singlePZ)
    #     rotation_hat = rotate_hat - np.angle(singlePZ[0])
    #     rotationMatrix = np.diag( np.exp(1j * rotation_hat * ( np.arange(len(y)) ) ) )
    #     y_corrected = y @ rotationMatrix
    #     subSectorEst, msgRx = self.majorityVoteDecoder(y_corrected, Q)
    #     return msgRx, rotate_hat

    # def pilotDecoder_FRO(self, y, Q, singlePZ):
    #     rotate_hat, subSector = self.estRotation(y, Q, singlePZ)
    #     rotation_hat = rotate_hat - np.angle(singlePZ[0])
    #     rotationMatrix = np.diag( np.exp(1j * rotation_hat * ( np.arange(len(y)) ) ) )
    #     y_corrected = y @ rotationMatrix
    #     msgRx = self.pilotFROdecoder(y_corrected, Q)
    #     return msgRx, rotate_hat      

    # def pilotDecoder_FracInt(self, y, Q, singlePZ):
    #     y_ffoCorrected = self.ffoEstCor(y, Q)
    #     return self.singlePZDecodedMsg(y_ffoCorrected, Q, singlePZ)  

    #  ----- Conculsion of MOCZ Pilot-Zero Decoder

    def simulator(self, noIter, perParam, ch, PZrad=1.25, Q=64):
        singlePZ = [-PZrad*self.R]
        BER, PCR, PAPR, rotationEst = 0, 0, 0, 0
        for _ in range(noIter):
            msgTx = np.random.randint(0, 2, self.K)
            sigTx = self.coeffCon(msgTx, singlePZ)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            rotation = np.random.uniform(0, 2*np.pi)
            sigRx = ch.transmit(sigTx, rotation)

            msgRx, rotation_hat = self.singlePZDecodedMsg(sigRx, Q, singlePZ)

            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += self.PAPR(sigTx)
            rotationEst += np.abs(rotation_hat - rotation) / rotation
        return dict({
                'ber': BER/noIter,
                'pcr': PCR/noIter,
                'papr': PAPR/noIter,
                'rotationEst': rotationEst/noIter
        })

    def simulatorACPC(self, noIter, perParam, ch, acpc, Q=64):
        BER, PCR, PAPR, lEst = 0, 0, 0, 0
        for _ in range(noIter):
            msgTx = np.random.randint(0, 2, acpc.B)
            msgEnc = acpc.msg_encoding(msgTx)

            sigTx = self.coeffCon(msgEnc)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            rotation = np.random.uniform(0, 2*np.pi)
            l = np.floor(rotation / (2*np.pi/self.K))

            sigRx = ch.transmit(sigTx, rotation)
            sigFROcorrected = self.ffoEstCor(sigRx, Q)
            estCodeword = self.fftDizet(sigFROcorrected, Q)
            msgRx, l_hat = acpc.codeword_decoding(estCodeword)

            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += self.PAPR(sigTx)
            lEst += abs(l_hat - l) / (l+1)
        return dict({
                'ber': BER/noIter,
                'pcr': PCR/noIter,
                'papr': PAPR/noIter,
                'lEst': lEst/noIter
        })