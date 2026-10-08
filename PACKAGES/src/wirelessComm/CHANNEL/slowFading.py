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
    
    def conRayleigh(self, n):
        sigma = np.sqrt(self.chVar/2)
        h = np.sqrt(self.pathLoss) * ( sigma * ( np.random.randn(n) + 1j*np.random.randn(n)) )
        return h

    def CFO(self, signal, rotation, noiseFlag=True):
        # attenuation provided by the channel
        h = [ np.exp(1j * rotation * i) for i in range(len(signal)-1, -1, -1) ]
        if noiseFlag:
            return (signal + self.awgn_noise(len(signal))) * h
        return signal * h

    def blockFading(self, signal, noiseFlag=True, rotation=0):
        ch_coeff = ( np.sqrt(self.chVar/2) * ( np.random.randn() + 1j*np.random.randn() ) )
        rxSig = ch_coeff * signal
        if rotation != 0:
            return self.CFO(rxSig, rotation, noiseFlag), ch_coeff
        elif noiseFlag:
            return rxSig + self.awgn_noise(len(signal)), ch_coeff
        return rxSig, ch_coeff
            
        # we also need to estimate the path loss along with channel coefficient

    def exponentialPDP(self):
        h = np.array([])
        power = 0
        for i in range(self.taps):
            h_i = np.sqrt((self.chVar**i)/2) * ( np.random.randn() + 1j*np.random.randn() )
            h = np.append(h, h_i)
            power += self.chVar**i
        return h / np.sqrt(power)

    def frequencySelective(self, signal, noiseFlag=True, rotation=0):
        freqSelCh = self.exponentialPDP()
        rxSig = np.convolve(signal, freqSelCh)
        if rotation != 0:
            return self.CFO(rxSig, rotation, noiseFlag), freqSelCh
        elif noiseFlag:
            return rxSig + self.awgn_noise(len(rxSig)), freqSelCh
        return rxSig, freqSelCh

    def PDP_LTI(self, delay, power, signal, noiseFlag=True):
        delayMax = max(delay)
        h = np.zeros(delayMax, dtype=complex)
        for d, p in zip(delay, power):
            h[d-1] += np.sqrt(p/2) * ( np.random.randn() + 1j*np.random.randn() )
        h /= np.sqrt(np.sum(power))
        rxSig = np.convolve(signal, h)
        if noiseFlag:
            return rxSig + self.awgn_noise(len(rxSig))
        return rxSig

    def PDP_LTV(self, delay, power, signal, Tc, noiseFlag=True):
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
        if noiseFlag:
            return rxSig + self.awgn_noise(len(rxSig))
        return rxSig