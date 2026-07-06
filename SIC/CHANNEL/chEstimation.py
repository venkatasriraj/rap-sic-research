import numpy as np

class ChannelEstimation:

    @staticmethod
    def leastSquares(received_sig, recon_sig):
        if len(received_sig) != len(recon_sig):
            return ValueError("Length Mismatch between received "
                                "and reconstructed signal")
        num, den = 0, 0
        for i,j in zip( recon_sig, received_sig ):
            num += np.conjugate(i) * j
            den += np.conjugate(i) * i
        return num / den