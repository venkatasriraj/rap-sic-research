import numpy as np

from CHANNEL import Channel

class SlowFadingChannel(Channel):

    def __init__(self, var = 0.1, ch_var = 1):
        super().__init__(var)  #  sif not None else np.random.uniform(0,1)
        self.ch_var = ch_var
        
    def transmit(self, signal):
        ch_coeff = np.sqrt(self.ch_var/2) * ( np.random.randn() + 1j*np.random.randn() )
        coeffPower = np.abs(ch_coeff)**2
        ch_coeff_norm = ch_coeff / np.sqrt(coeffPower)
        return signal * ch_coeff_norm + self.awgn_noise(len(signal)), ch_coeff_norm
        # return self.awgn_noise(len(signal)), ch_coeff_norm

    def conRayleigh(self, n):
        sigma = np.sqrt(self.ch_var/2)
        h = sigma * ( np.random.randn(n) + 1j*np.random.randn(n))
        h_power = np.abs(h)**2
        h_norm = h / np.sqrt(h_power)
        return h_norm