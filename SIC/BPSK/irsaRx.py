"""
Designing a class for study of IRSA with BPSK Receiver
"""
import numpy as np
from scipy.signal import find_peaks

from BPSK import BPSKBase

class IRSAReceiver(BPSKBase):

    def __init__(self, accessCode, AC_Threshold=0.7):
        self.accessCode = accessCode
        self.acThreshold = AC_Threshold

    @staticmethod
    def ber(pkt_hat, pkt):
        ber = np.mean(pkt_hat != pkt)
        return ber

    def decodeSlot(self, samples):
        start, AC_SamLen = self.findAccessCode(samples)
        if start is None:
            return None, None
        bits = self.demodulate(samples[start:])
        pkt = self.bits_to_bytes(bits)
        # we will be returnin bits[0:AC_SamLen] which will be used for
        #  channel estimation using access code.
        return pkt, samples[start: start+AC_SamLen]

    # --- we will be correlating on baseband signal to identify the peak
    def findAccessCode(self, samples):
        aCodeBits = self.bytes_to_bits(self.accessCode)
        aCodeSam = self.modulate(aCodeBits)
        corrOP = np.correlate(samples, aCodeSam, mode='valid')
        
        samplesEnergy = np.correlate( np.abs(samples)**2, np.ones(len(aCodeSam)), mode='valid')
        samplesEnergy = np.sqrt( np.maximum(samplesEnergy, 1e-12) )
        aCodeSamEnergy = np.sqrt( np.sum( np.abs(aCodeSam)**2 ) )

        corrOP = np.abs(corrOP) / (samplesEnergy * aCodeSamEnergy + 1e-12)
        # find_peaks have edge problem. It ignore peaks at edges so we need to pad -inf at the edges
        # and distance is used to find the max_peak among the peaks identified among the given set of samples
        corrOP = np.pad(corrOP, (1,1), constant_values=-np.inf)
        peak_indices, properties = find_peaks( corrOP, height=self.acThreshold, distance=len(samples))
        if len(peak_indices) == 0:
            return None, None
        peak_indices -= 1
        return peak_indices[0], len(aCodeSam)

    # we will be implementing normalized auto-correlation instead of find access code
    # since we can't measure it in number of bits because auto-correlation output varies 
    # wrt noise and channel coefficient. we will be normalizing it wrt the signal and template energy
    # --- signal energy will be found over the template width.


