"""
Designing a class for study of IRSA with BPSK Transmitter
"""

import numpy as np
import random

from .bpskModulation import BPSKBase

class IRSATransmitter(BPSKBase):

    def __init__(self, slotLen):
        self.slotLen = slotLen

    def pkt_to_iq(self, pkt):
        pkt_bits = self.bytes_to_bits(pkt)
        samples = self.modulate(pkt_bits)
        if len(samples) < self.slotLen:
            samples = np.append(samples, [0]*(self.slotLen - len(samples))) 
        return np.array(samples, dtype=np.complex64)