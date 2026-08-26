import numpy as np

from .channel import Channel

class MultiPathFading(Channel):

    # this code is written for zero rotation in MOCZ
    def __init__(self, noise_var, chVar=1, taps=1, pathLoss=1):
        super().__init__(noise_var=noise_var, pathLoss=pathLoss)
        self.chVar = chVar
        self.taps = taps

    def transmit(self, signal, rotation):
        # attenuation provided by the channel
        r = 1 # np.random.random()
        h = r * [ np.exp(1j * rotation * i) for i in range(len(signal)-1, -1, -1) ]
        return signal * h + self.awgn_noise(len(signal))

    def multitapCh(self):
        chTapVar = self.chVar / self.taps
        return np.sqrt(1/chTapVar) * ( np.random.randn(self.taps) + 1j * np.random.randn(self.taps) )

    def multipathTransmit(self, signal):
        freqSelCh = self.multitapCh()
        rxSig = np.convolve(signal, freqSelCh)
        return rxSig + self.awgn_noise(len(rxSig)), freqSelCh