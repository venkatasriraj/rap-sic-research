"""
Implementation of deconvolution algorithms using the decoded message
from DiZeT decoder to estimate h for succesive interference cancellation(SIC)
T: convolution matrix; shape(T) = (N x Lh); N = K + Lh
"""
import numpy as np
from scipy.linalg import toeplitz
from wirelessComm import BMOCZ, SlowFadingChannel, PerformanceParameters

K, Q = 4, 2
Lh, chVar = 2, 0.86
energy = K + 1
system = BMOCZ(K, PZrad=0, Q=Q)
ch = SlowFadingChannel(noise_var=10**(-5/10), chVar=chVar, taps=Lh)
perParam = PerformanceParameters()

msgTx = np.random.randint(0, 2, K)
sigTx = system.transmitterRAP(msgTx, energy)

sigRx, h = ch.frequencySelective(sigTx, noiseFlag=True)
# print(f'Channel impulse response: {h[:, None]}')
msgRx, _, _ = system.receiverRAP(sigRx)
ber = perParam.ber(msgRx, msgTx)
print(f"BER: {ber}")

# The reconstructed signal is a function of the estimated message (msgRx)
# Construct the convolution matrix T(x_hat)

sigRecon = system.transmitterRAP(msgRx, energy)
# print(sigRecon)
T = toeplitz(np.concatenate((sigRecon, [0]), axis=None), [1, 0])
# print(T)
Tinv = np.linalg.pinv(T)
U, S, Vh = np.linalg.svd(T, full_matrices=True)
# print(Tinv @ T)

# print(sigRx)
# r = Tinv @ sigRx
# h[:, None]: transforms row vector to column vector
# print(r.shape)
# r[:, -1]: transforms column vector to row vector
# print(r[:, -1].shape)

slotDeconvolved = Tinv @ sigRx
print(h)
print(slotDeconvolved)