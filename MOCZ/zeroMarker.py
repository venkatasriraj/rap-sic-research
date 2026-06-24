"""
Implementation of Zero-Marker in BMOCZ to solve rotation of zeros due to CFO.
"""

import numpy as np

from BMOCZ import (
    BMOCZTransmitter,
    BMOCZReceiver
)
from CHANNEL import MultiPathFading

K = 3
Q = 7
theta_K = np.pi * 2 / K
rotationPossible = np.arange(0, theta_K, theta_K/8)
intRotationPossible = theta_K * np.arange(K)
# print(f"Possible Integer Rotaions: {intRotationPossible}\n")
# print(f"Possible Fractional Rotations: {rotationPossible}\n")

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
ch = MultiPathFading(noise_var=0.01)

msg = [np.random.randint(2) for i in range(K)]

sig_tx = tx.coeffConZM(msg)
rotation = np.random.uniform(0, 2*np.pi)
sig_rx = ch.transmit(sig_tx, rotationPossible[6])
print(f"Rotation applied: {rotation} (in rad), {rotation * 180 / np.pi} (in deg)")
# print(f"Integer rotation applied: {intRotationPossible[6]}")
msg_un = rx.fftDizet(sig_rx, Q)
sig_ffo = rx.ffoEstCor(sig_rx, Q)
msg_rx = rx.fftDizet(sig_ffo, Q)
# -- Zero-Marker rotation identification
integer_rotate = rx.ZMDetection(sig_ffo)

msg_rotated = np.roll(msg_rx, -integer_rotate)
print(f"Uncorrected msg: {msg_un}")
print(f"Transmitted Message: {msg}, \nReceived Message: {msg_rx}, \n Rotated Message: {msg_rotated}")
ber = rx.ber(msg_rotated, msg)
print(f"BER: {ber}")