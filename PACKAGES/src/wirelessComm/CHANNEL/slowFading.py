import numpy as np

from .channel import Channel

class SlowFadingChannel(Channel):

    def __init__(self, noise_var, pathLoss = 1, chVar = 1, taps=1, power=None, delay=None):
        super().__init__(noise_var, pathLoss)  #  sif not None else np.random.uniform(0,1)
        self.chVar = chVar
        self.taps = taps
        self.power = power
        self.delay = delay

    def awgn(self, signal):
        return signal + self.awgn_noise(len(signal))

    def blockFading(self, signal, noiseFlag=True):
        ch_coeff = ( np.sqrt(self.chVar/2) * ( np.random.randn() + 1j*np.random.randn() ) )
        power = abs(ch_coeff)**2
        ch_coeff /= power
        rxSig = ch_coeff * signal
        if noiseFlag:
            return rxSig + self.awgn_noise(len(signal)), ch_coeff
        return rxSig, ch_coeff
            
        # we also need to estimate the path loss along with channel coefficient
    
    def conRayleigh(self, n):
        sigma = np.sqrt(self.chVar/2)
        h = np.sqrt(self.pathLoss) * ( sigma * ( np.random.randn(n) + 1j*np.random.randn(n)) )
        return h

    def CFO(self, signal, rotation, noiseFlag=True):
        # attenuation provided by the channel
        r = 1 # np.random.random()
        h = r * [ np.exp(1j * rotation * i) for i in range(len(signal)-1, -1, -1) ]
        if noiseFlag:
            return (signal + self.awgn_noise(len(signal))) * h
        return signal * h

    def multitapCh(self):
        h = np.sqrt(self.chVar / 2 ) * ( np.random.randn(self.taps) + 1j * np.random.randn(self.taps) )
        # channel power is also normalised to 1
        power = np.mean(np.abs(h)**2)
        return h / np.sqrt(power)

    def frequencySelective(self, signal, noiseFlag=True):
        freqSelCh = self.multitapCh()
        rxSig = np.convolve(signal, freqSelCh)
        if noiseFlag:
            return rxSig + self.awgn_noise(len(rxSig)), freqSelCh
        return rxSig, freqSelCh

    def PDP_LTI(self, delay, power, signal):
        delayMax = max(delay)
        h = np.zeros(delayMax, dtype=complex)
        for d, p in zip(delay, power):
            h[d-1] += np.sqrt(p/2) * ( np.random.randn() + 1j*np.random.randn() )
        h /= np.sqrt(np.sum(power))
        rxSig = np.convolve(signal, h)
        return rxSig + self.awgn_noise(len(rxSig))

    def PDP_LTV(self, delay, power, signal, Tc):
        delayMax = max(delay)
        rxLen = len(signal)+delayMax
        rxSig = np.zeros(rxLen)
        Tc_k = -1
        for delaytime in delay:
            if delaytime // Tc != Tc_k:
                Tc_k == delaytime // Tc
                h = np.zeros(delayMax, dtype=complex)
                for d, p in zip(delay, power):
                    h[d-1] += np.sqrt(p/2) * (np.random.randn() + 1j*np.random.randn())
                h /= np.sqrt(np.sum(power))
            rxSig[delaytime:rxLen] = np.convolve(signal, h)
        return rxSig + self.awgn_noise(len(rxSig))