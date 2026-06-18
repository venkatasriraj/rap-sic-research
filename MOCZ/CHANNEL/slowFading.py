import numpy as np

from CHANNEL import Channel

class SlowFadingChannel(Channel):

    def __init__(self, var = 0.1, ch_var = 1):
        super().__init__(var)  #  sif not None else np.random.uniform(0,1)
        self.ch_var = ch_var
        
    def transmit(self, signal):
        ch_coeff = np.sqrt(self.ch_var/2) * ( np.random.randn() + 1j*np.random.randn() )
        return signal * ch_coeff + self.awgn_noise(len(signal)), ch_coeff