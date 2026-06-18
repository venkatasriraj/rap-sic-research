"""
Designing a class for BPSK Transmitter
"""

import numpy as np
import random

class BPSKTransmitter:

    @staticmethod
    def bytes_to_bits(pkt):
        bits = []
        for data in pkt:
            for i in range(7,-1,-1):
                bits.append( (data >> i) & 1 )
        return bits

    def modulate(self, pkt):
        pkt_bits = self.bytes_to_bits(pkt)
        iqSamples = []
        for i in pkt_bits:
            iqSamples.append( 2*i -1 )
        return np.array(iqSamples)