import numpy as np

from CHANNEL import Channel

class SlowFadingChannel(Channel):

    def __init__(self, var = 0.1, ch_var = 1):
        self.var = var   #  sif not None else np.random.uniform(0,1)
        self.ch_var = ch_var
        self.coeff = np.sqrt(ch_var/2) * ( np.random.randn() + 1j*np.random.randn() )

    def transmit(self, signal):
        awgn_noise = np.sqrt(self.var/2) * self.awgn_noise(len(signal))
        return signal * self.coeff + awgn_noise, self.coeff