"""
Learning Zero Constellations for Binary MOCZ in Fading Channels
- ML based learning the constellation
"""
import numpy as np
from scipy.linalg import toeplitz
import torch
import torch.nn as nn

class MLconstellation:

    def __init__(self, K, snr, batchSize, epochs, learningRate, Lhidden):
        self.K = K
        self.snr = snr
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.Lhidden = Lhidden

    def toeplitz_iterator(self, zeros):
        for k in range(len(zeros)):
            if k == 0:
                c = np.array([[1, -zeros[k]]]).T   # (z-alpha)
            else:
                column = np.zeros(k+2, dtype=complex)
                column[0] = 1
                column[1] = -zeros[k]

                row = np.zeros(k+1, dtype=complex)
                row[0] = 1

                T = toeplitz(column, row)

                c = T @ c
        x = c.flatten()
        # polynomial in the increasing power of x (polynomial degree) is transmitted
        return x[::-1]

    
    def coeffCon(self, msg, R, theta):
        zeros = [R**( 2*msg[k]-1)*np.exp(1j*theta) for k in range(self.K)]
        return self.toeplitz_iterator(zeros)

    def architecture(self):
        R = torch.tensor(np.sqrt(1 + np.sin(np.pi/self.K)))
        theta = torch.tensor(2*np.pi*np.range(self.K)/self.K)
        model = nn.Sequential([
            nn.Linear(self.K, self.Lhidden),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(self.Lhidden, self.Lhidden),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(self.Lhidden, self.K)
        ])
        optimizer = torch.optim.Adam(
            list(model.parameters) + [R, theta]
        )

    def train(self):

        pass

    def simulator(self):

        pass