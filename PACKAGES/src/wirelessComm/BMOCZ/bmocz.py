import numpy as np
from .mocz import MOCZ
from scipy.linalg import toeplitz

class BMOCZ(MOCZ):

    def __init__(self, K, Ns=512, PZrad=1.578, Q=16):
        super().__init__(K, Q=Q)
        self.zero_geometry = self.codebook_con()
        self.Ns = Ns
        self.PZ = [-PZrad * self.R] if PZrad != 0 else []
        self.template = self.templateCon()
        self.zeroCollection = np.append(self.zeroSet(), self.PZ)

    def codebook_con(self):
        Ri, Ro = self.R**(-1), self.R
        zero_cb = [( Ri * np.exp(1j*self.theta_K*k), Ro * np.exp(1j*self.theta_K*k) ) for k in range(self.K)]
        return zero_cb

    def zeroSelection(self, msg):
        zeros = [self.zero_geometry[mk][msg[mk]]*np.exp(1j*np.pi*0/self.K) for mk in range(self.K)]
        zeros += self.PZ
        return np.asarray(zeros)
    
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
        theta_est = ( q_est / Q ) * self.theta_K
        return theta_est, q_est
    
    def ffoEstCor(self, y, Q):
        theta_hat, q_est = self.ffo_est(y, Q)
        # M_theta = np.diag( np.exp(-1j*theta_hat) ** np.flip( np.arange(len(y)) ) )
        M_theta = np.diag( np.exp(1j*theta_hat) ** np.arange(len(y)) )
        return y @ M_theta, theta_hat

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

    def singlePZDecodedMsg(self, y, Q):
        rotate_hat, subSector = self.estRotation(y, Q, self.PZ) 
        rotation_hat = rotate_hat - np.angle(self.PZ[0]) # if rotate_hat >= 0 else (2*np.pi + rotate_hat)
        rotationMatrix = np.diag( np.exp(-1j*rotation_hat) ** np.flip(np.arange(len(y))) )
        y_corrected = y @ rotationMatrix
        Yo, Yi = self.PZfftCon(y_corrected)
        msgDecoded = ( 1 + np.sign(Yi - Yo) ) / 2
        # msgDecoded = self.fftDizet(y_corrected)
        return msgDecoded.astype(int), rotate_hat

    def pilotFROdecoder(self, y, Q):
        Y_eval, Y_ctr_eval = self.fftCon(y, Q)
        min_q = {}
        for q in range(Q):
            sumZero = 0
            for k in range(self.K):
                idx = (Q * k + q) % len(Y_eval)
                sumZero += min(Y_eval[idx], Y_ctr_eval[idx])
            min_q[q] = sumZero
        q_est = min(min_q, key=min_q.get)
        msgDecoded = ( 1 + np.sign(Y_ctr_eval[q_est::Q] - Y_eval[q_est::Q]) ) / 2
        # msgDecoded = ( 1 - np.sign(Y_eval[::Q] - Y_ctr_eval[::Q]) )/2
        return msgDecoded.astype(int)

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

    def pilotDecoder_FracInt(self, y, Q=64):
        # integer rotation est using the pilot-zero
        rotate_hat, subSector = self.estRotation(y, Q, self.PZ) 
        rotation_hat = rotate_hat - np.angle(self.PZ[0]) # if rotate_hat >= 0 else (2*np.pi + rotate_hat)
        rotationMatrix = np.diag( np.exp(-1j*rotation_hat) ** np.flip(np.arange(len(y))) )
        y_IROcorrected = y @ rotationMatrix
        msgRx = self.pilotFROdecoder(y_IROcorrected, Q=2)
        return msgRx, rotation_hat

        # y_ffoCorrected = self.ffoEstCor(y, Q)
        # return self.singlePZDecodedMsg(y_ffoCorrected, Q, singlePZ)  

    #  ----- Conculsion of MOCZ Pilot-Zero Decoder

    def simulator(self, noIter, perParam, ch, Energy='None', Q=64, FLAG="FS", noiseFlag=True, errorAnalysis=False):
        if not errorAnalysis:
            BER, PCR, PAPR, rotationEst, rotation_hat = 0, 0, 0, 0, 1e-7
        else:
            BER, PCR = 0, 0
            errorDist, errorPosition = np.zeros(self.K+2, dtype=int), np.zeros(self.K+1, dtype=int)
        E = self.K+1 if Energy == 'None' else Energy
        for _ in range(noIter):
            msgTx =  np.random.randint(0, 2, self.K)
            sigTx = self.coeffCon(msgTx)
            # sigPower = np.mean(np.abs(sigTx)**2)
            # sigTx /= np.sqrt(sigPower)
            l2_norm = np.linalg.norm(sigTx, ord=2) # energy = l2_norm**2
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

            if ( FLAG == 'CFO' or FLAG == 'FS_CFO'):
                if len(self.PZ) == 1:
                    # msgRx, rotation_hat = self.singlePZDecodedMsg(sigRx, Q)
                    sigRx, rotation_hat = self.rotationEstTemplate(sigRx)
                else:
                    # in this case only fractional rotation offset FRO is corrected
                    sigRx, rotation_hat = self.ffoEstCor(sigRx, Q)
            msgRx = self.fftDizet(sigRx, Q)
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            if not errorAnalysis:
                # mae = np.abs(rotation - rotation_hat) if np.abs(rotation - rotation_hat) < np.pi else 2*np.pi-np.abs(rotation - rotation_hat)
                mae = np.min([np.abs(rotation-rotation_hat), np.abs(rotation - (2*np.pi-rotation_hat))])
                rotationEst += mae/rotation
                PAPR += perParam.PAPR(sigTx)
            else:
                errorDist[perParam.bitError(msgRx, msgTx)] += 1
                errorPosition[perParam.errorPos(msgRx, msgTx)] += 1
        if not errorAnalysis:
            return dict({
                # 'ber': BER/(noIter*self.K),
                'ber': BER/noIter,
                'pcr': PCR/noIter,
                'papr': PAPR/noIter,
                'rotationEst': rotationEst/noIter
            })
        else:
            return dict({
                'ber': BER/noIter,
                'pcr': PCR/noIter,
                'errorDist': errorDist,
                'errorPosition': errorPosition
            })

        

    def simulatorACPC(self, noIter, perParam, ch, acpc, Energy='None', Q=64, FLAG='FS', noiseFlag=True, errorAnalysis=False):
        if not errorAnalysis:
            BER, PCR, PAPR, rotationEst, theta_hat = 0, 0, 0, 0, 0
        else:
            BER, PCR = 0, 0
            errorDist, errorPosition = np.zeros(self.K+2, dtype=int), np.zeros(self.K+1, dtype=int)
        E = self.K+1 if Energy == 'None' else Energy
        for _ in range(noIter):
            msgTx = np.random.randint(0, 2, acpc.B)
            msgEnc = acpc.msgEncoding(msgTx)

            sigTx = self.coeffCon(msgEnc)
            l2_norm = np.linalg.norm(sigTx, ord=2) # energy = l2_norm**2
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

            if ( FLAG == 'CFO' or FLAG == 'FS_CFO'):
                sigRx, theta_hat = self.ffoEstCor(sigRx, Q) 
            estCodeword = self.fftDizet(sigRx, Q)
            msgRx, l_hat = acpc.msgDecoding(estCodeword)
            rotation_hat = l_hat * self.theta_K + theta_hat
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            if not errorAnalysis:
                # mae = np.abs(rotation - rotation_hat) if np.abs(rotation - rotation_hat) < np.pi else 2*np.pi-np.abs(rotation - rotation_hat)
                mae = np.min([np.abs(rotation-rotation_hat), np.abs(rotation - (2*np.pi-rotation_hat))])
                rotationEst += mae/rotation
                PAPR += perParam.PAPR(sigTx)
            else:
                errorDist[perParam.bitError(msgRx, msgTx)] += 1
                errorPosition[perParam.errorPos(msgRx, msgTx)] += 1
        if not errorAnalysis:
            return dict({
                # 'ber': BER/(noIter * acpc.B),
                'ber': BER/noIter,
                'pcr': PCR/noIter,
                'papr': PAPR/noIter,
                'rotationEst': rotationEst/noIter
            })
        else:
            return dict({
                'ber': BER/noIter,
                'pcr': PCR/noIter,
                'errorDist': errorDist,
                'errorPosition': errorPosition
            })

    def templateCon(self):
        msg = np.ones(self.K, dtype=np.uint8)
        coeff = self.coeffCon(msg)
        template = (abs(np.fft.fft(coeff, n=self.Ns))**1)
        maxVal = np.max(template)
        return template/maxVal

    def rotationEstTemplate(self, y):
        row = np.concatenate((self.template[0], self.template[:0:-1]), axis=None)
        templateMatrix = toeplitz(self.template, row)
        rxTemplate = (abs( np.fft.fft(y, n=self.Ns) )**1)
        maxVal = np.max(rxTemplate)
        rxTemplate /= maxVal
        innerPdt = rxTemplate @ templateMatrix
        indx = np.argmax(innerPdt)
        phi_hat = (2 * np.pi * indx / self.Ns) 
        y_corrected = y @ np.diag( np.exp(-1j * phi_hat * ( np.arange(len(y)) ) ) )
        return y_corrected, abs(phi_hat)

    def transmitterRAP(self, msgTx, ENERGY, acpc='None'):
    #     if acpc == 'None':
    #         msgTx = np.random.randint(0, 2, acpc.B)
    #     else:
    #         msgTx = np.random.randint(0, 2, self.K)
        sigTx = self.coeffCon(msgTx) 
        l2_norm = np.linalg.norm(sigTx, ord=2) 
        sigTx = (sigTx / l2_norm) * np.sqrt(ENERGY)
        return sigTx

    def receiverRAP(self, sigRx, CFO=False, acpc='None'):
        rotation_hat = 0
        if CFO:
            sigRx, rotation_hat = self.ffoEstCor(sigRx, self.Q)
        codeWord = self.fftDizet(sigRx, Q=2) 
        if acpc != 'None':
            msgRx, l_hat = acpc.msgDecoding(codeWord)
            rotation_hat += l_hat * self.theta_K 
        return codeWord, rotation_hat, sigRx