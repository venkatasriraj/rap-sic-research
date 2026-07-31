import numpy as np

class Channel:

    def __init__(self, noise_var, pathLoss):
        self.noise_var = noise_var
        self.pathLoss = pathLoss 

    def transmit():
        raise NotImplementedError(
            "Derived channel must implement transmit()"
        )

    # here we are using average SNR; transmitted signal has unit power
    # while the channel coefficient is complex gaussian with unit variance;
    # (which can also be used for capture effect)
    def awgn_noise(self, l):
        return np.sqrt(self.noise_var/2) * (np.random.randn(l) + 1j*np.random.randn(l))
    
    # def awgn_noisePL(self, l, rxSigPower = None):
    #     if rxSigPower == None:
    #         rxSigPower = self.pathLoss
    #     noiseVar = rxSigPower * 10**(-self.SNR / 10)
    #     return np.sqrt(noiseVar/2) * (np.random.randn(l) + 1j*np.random.randn(l))

    def sic_transmit(self, signals, chCoeffs):
        sig_interferenced = 0
        for sig, chCoeff in zip(signals, chCoeffs):
            sig_interferenced += sig *  chCoeff
        return sig_interferenced + self.awgn_noise(len(signals[0]))