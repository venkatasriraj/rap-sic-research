import numpy as np

class ChannelEstimation:

    @staticmethod
    def leastSquares(received_sig, recon_sig):
        if len(received_sig) != len(recon_sig):
            return ValueError("Length Mismatch between received "
                                "and reconstructed signal")
        h_real, h_imag = 0, 0
        for i,j in zip( recon_sig, received_sig ):
            h_real += np.real( ( i * np.conjugate(j) ) / ( i * np.conjugate(i) ) )
            h_imag += np.imag( ( i * np.conjugate(j) ) / ( i * np.conjugate(i) ) )
        return ( h_real - 1j*h_imag )/len(recon_sig)