import numpy as np

class Channel:

    def transmit():
        raise NotImplementedError(
            "Derived channel must implement transmit()"
        )

    def awgn_noise(self, l):
        return (np.random.randn(l) + 1j*np.random.randn(l))
