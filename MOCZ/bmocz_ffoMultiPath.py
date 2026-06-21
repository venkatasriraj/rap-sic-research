"""
Identification of a scenario to analyze the performance of DiZeT FFT based decoder 
and estimation of fractional frequency offset.
Test Case: A multipath channel which provides the rotation less than (2 * pi)/K
    Previous Result: q_est = 0 (always)

NOTE: # x = [ x0, x1, x2, ....., xK] don't forger the order of degree in which the
signal is being transmitted.
"""
import numpy as np

from BMOCZ import (
    BMOCZTransmitter,
    BMOCZReceiver
)
from CHANNEL import MultiPathFading


K = 4
Q = 8
theta_K = np.pi * 2 / K
rotationPossible = np.arange(0, theta_K, theta_K/8)
# print(f"Possible rotations: {rotationPossible}")
msg = [0]*K

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
ch = MultiPathFading(noise_var=0.01)

sig_tx = tx.coeffCon(msg)
# print(f"Transmitted sequence: {sig_tx}")
sig_rx = ch.transmit(sig_tx, rotationPossible[3])
# print(f"Received sequence: {sig_rx}")

sig_ffo = rx.ffoEstCor(sig_rx, Q)
msg_rx = rx.fftDizet(sig_ffo, Q)
ber = rx.ber(msg_rx, msg)
print(f'BER: {ber}') 