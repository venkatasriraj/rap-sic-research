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

# invalid implementation - performance of the system is bad
    @staticmethod
    def modifiedLS(received_sig, recon_sig, coeffThresh=0.001):

        if len(received_sig) != len(recon_sig):
            raise ValueError("Length mismatch between received and reconstructed signal.")
        h_real, h_imag = 1e-12, 1e-12
        for i, j in zip( recon_sig, received_sig):
            if np.abs(i) >= coeffThresh:
                x = ( i * np.conjugate(j) ) / ( i * np.conjugate(i) )       # j * np.conjugate(i) = real(x) -1j*imag(x) 
                h_real += np.real(x)
                h_imag += np.imag(x)
        count = np.count_nonzero( np.greater_equal(np.abs(recon_sig), coeffThresh) )
        # print(np.abs(recon_sig), count)
        return ( h_real -1j*h_imag )/count