"""
Design of OFDM Tranmitter and receiver class

Write an automated code which figure out the pilot positions
based on the coherence bandwidth and coherence time.

P = N / subcarrier spacing (dp)
P >= L (number of taps)
"""
import numpy as np
from scipy import linalg

class OFDM:

    def __init__(self, N=256, cpLen=32, pilotSpacing=4, D=None):
        self.N = N
        self.cpLen = cpLen
        self.pilotSpacing = pilotSpacing
        self.pilotSym = 2
        self.pilot, self.pilotIdx = self.pilotCon()
        self.D = cpLen-1 if D is None else D

    def pilotCon(self):
        msgTx = np.zeros(self.N, dtype=complex)
        pilot = np.arange(self.N)[::self.pilotSpacing]
        msgTx[pilot] = self.pilotSym
        ind = np.setdiff1d(np.arange(self.N), pilot)
        msgTx[ind] = 0
        return msgTx, pilot

    def txModule(self, symbols):
        ofdmSym = np.fft.ifft(symbols) * self.N # np.sqrt(self.N)
        return np.concatenate((ofdmSym[self.N-self.cpLen:], ofdmSym), axis=None)

    def rxModule(self, sig):
        ofdmSym = sig[:self.cpLen+self.N]
        return np.fft.fft(ofdmSym[self.cpLen:]) #/ np.sqrt(self.N)

    def cohTxModule(self, symbols):
        pilotInserted = self.pilotInsertion(symbols)
        # print(f"Pilot Inserted: {pilotInserted}")
        ofdmSym = np.fft.ifft(pilotInserted) * self.N # np.sqrt(self.N)
        return np.concatenate((ofdmSym[self.N-self.cpLen:], ofdmSym), axis=None)

    def cohRxModule(self, sig):
        ofdmSym = sig[:self.cpLen+self.N]
        freqSym = np.fft.fft(ofdmSym[self.cpLen:]) #/ np.sqrt(self.N)
        # print(f"Freq Sym Est: {freqSym}")
        pilotExtracted = self.channelEqualization(freqSym)
        symbolsEst = self.pilotExtraction(pilotExtracted)
        return symbolsEst

    def pilotInsertion(self, symbol):
        msgTx = self.pilot.copy()
        ind = np.setdiff1d(np.arange(self.N), self.pilotIdx)
        msgTx[ind] = symbol
        return msgTx

    def pilotExtraction(self, symbol):
        ind = np.setdiff1d(np.arange(self.N), self.pilotIdx)
        return symbol[ind]

    def channelEqualization(self, symbol):   
        # chHat = self.channelEst(symbol[self.pilotIdx])
        chHat = self.channelEstLS(symbol[self.pilotIdx])
        # print(f"Ch Est: {chHat}")
        chMat = np.diag(chHat)
        symbolEqualised = symbol @ linalg.inv(chMat)
        return symbolEqualised

    def channelEstLS(self, pilotRx):
        F = np.exp(-2j * np.pi * np.outer(self.pilotIdx, np.arange(self.D+1)) / self.N) # P x L matrix
        h = np.linalg.lstsq(F, pilotRx/self.pilotSym, rcond=None)[0]
        F_all = np.exp(-2j * np.pi * np.outer(np.arange(self.N), np.arange(self.D+1)) / self.N)
        return F_all @ h

    def channelEst(self, sigRx):
        num, den = 0, 0
        # print(sigRx, self.pilotSym)
        for j in sigRx:
            num += np.conjugate(self.pilotSym) * j
            den += np.conjugate(self.pilotSym) * self.pilotSym
        return num / den

    @staticmethod
    def bin2dec(binData):
        if len(binData) == 0:
            return 0
        power2 = 2**np.arange(len(binData))[::-1]
        return int(np.sum(binData * power2))

    @staticmethod
    def dec2bin(decData):
        binData = [decData % 2]
        decData //= 2
        while decData>0:
            binData += [decData%2]
            decData //= 2
        return np.asarray(binData)[::-1]