import numpy as np

from CHANNEL import Channel

class MultiPathFading(Channel):

    def transmit(self, signal, rotation):
        # attenuation provided by the channel
        r = 1 # np.random.random()
        h = r * [ np.exp(1j * rotation * i) for i in range(len(signal)-1, -1, -1) ]
        return signal * h + self.awgn_noise(len(signal))

    