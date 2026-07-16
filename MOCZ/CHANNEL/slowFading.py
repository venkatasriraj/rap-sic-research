import numpy as np

from CHANNEL import Channel

class SlowFadingChannel(Channel):

    def __init__(self, noise_var, pathLoss = 1, ch_var = 1):
        super().__init__(noise_var, pathLoss)  #  sif not None else np.random.uniform(0,1)
        self.ch_var = ch_var

    def transmit(self, signal):
        ch_coeff = np.sqrt(self.pathLoss) * ( np.sqrt(self.ch_var/2) * ( np.random.randn() + 1j*np.random.randn() ) )
        rxSig = ch_coeff * signal
        return rxSig + self.awgn_noise(len(rxSig)), ch_coeff

    def conRayleigh(self, n):
        sigma = np.sqrt(self.ch_var/2)
        h = np.sqrt(self.pathLoss) * ( sigma * ( np.random.randn(n) + 1j*np.random.randn(n)) )
        return h 