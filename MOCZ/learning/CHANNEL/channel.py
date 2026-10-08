import numpy as np

class Channel:

    def __init__(self, noise_var, pathLoss):
        self.noise_var = noise_var
        self.pathLoss = pathLoss

    def transmit():
        raise NotImplementedError(
            "Derived channel must implement transmit()"
        )

    def awgn_noise(self, l):
        return np.sqrt(self.noise_var/2) * (np.random.randn(l) + 1j*np.random.randn(l))

    def sic_transmit(self, signals, chCoeffs):
        sig_interferenced = 0
        for sig, chCoeff in zip(signals, chCoeffs):
            sig_interferenced += sig * chCoeff
        return sig_interferenced + self.awgn_noise(len(signals[0]))