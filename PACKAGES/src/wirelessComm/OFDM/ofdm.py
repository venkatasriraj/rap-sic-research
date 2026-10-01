"""
Design of OFDM Tranmitter and receiver class

Write an automated code which figure out the pilot positions
based on the coherence bandwidth and coherence time.
"""
import numpy as np

class OFDM:

    def __init__(self, N=256, cpLen=32, Mary=2):
        self.N = N
        self.cpLen = cpLen
        self.M = Mary

    def bpskModulation(self, msg):
        return 2 * np.asarray(msg).astype(int) - 1

    def bpskDemodulation(self, sig):
        return np.where(sig>=0, 1, 0).astype(int)

    def txModule(self, msg):
        modulatedSym = self.bpskModulation(msg)
        ofdmSym = np.fft.ifft(modulatedSym)
        return np.concatenate((ofdmSym[:-self.cpLen-1:-1], ofdmSym), axis=None)

    def rxModule(self, sig):
        ofdmSym = sig[self.cpLen:]
        freqSym = np.fft.fft(ofdmSym)
        return self.bpskDemodulation(freqSym)

    def channelEst():

        pass