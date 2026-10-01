"""
Implementation of M-ary Phase Shift Keying Modulation schemes
"""
import numpy as np
import math
class PSK:

    def __init__(self, M):
        self.M = M
        self.bitsPerSym = int(math.log2(M))
        self.theta = 2 * np.pi / M

    def modulation(msg):
        
        pass

    def demodulation(signal):

        pass