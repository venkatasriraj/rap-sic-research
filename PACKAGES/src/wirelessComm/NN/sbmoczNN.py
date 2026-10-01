"""
NN based parameter optimization implementation of SBMOCZ
"""

import torch
import torch.nn as nn

class SmooshedNN:

    def __init__(self, epochs, batchSize, Lhidden):
        self.epochs = epochs
        self.batchSize = batchSize
        self.Lhidden = Lhidden