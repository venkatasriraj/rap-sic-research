"""
Designing a class for BPSK Receiver
"""
import numpy as np

class BPSKReceiver:

    @staticmethod
    def bits_to_bytes(bits):
        bits = np.asarray(bits, dtype=np.uint8)
        n = (len(bits) // 8) * 8
        return [int(''.join( map(str, bits[i:i+8]) ), 2) for i in range(0, n, 8)]

    def demodulate(self, samples):
        bits = [1 if np.real( 0.5*(x+1) ) > 0.5 else 0 for x in samples]
        pkt = self.bits_to_bytes(bits)
        return pkt

    @staticmethod
    def ber(pkt_hat, pkt):
        ber = np.mean(pkt_hat != pkt)
        return ber

    def findAccessCode(self, samples):

        return None