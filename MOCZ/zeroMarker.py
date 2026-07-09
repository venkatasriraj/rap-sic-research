"""
Implementation of Zero-Marker in BMOCZ to solve rotation of zeros due to CFO.
"""

import numpy as np

from BMOCZ import (
    BMOCZTransmitter,
    BMOCZReceiver
)
from CHANNEL import MultiPathFading

K = 7
Q = 8
theta_K = np.pi * 2 / K
rotationPossible = np.arange(0, theta_K, theta_K/Q)
intRotationPossible = theta_K * np.arange(K)

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
ch = MultiPathFading(noise_var=0.01)

# experiment for pilot zero-selection
# l = tx.pilotZeroSelection()

msg = np.random.randint(0, 2, K)
# msg = np.ones(K, dtype=np.uint8)

sig_tx = tx.coeffConZM(msg)
sig_power = np.mean(np.abs(sig_tx))
print(f"Signal Power: {sig_power}, SNR: {sig_power / 0.01}")
rotation = np.random.uniform(0, 2*np.pi)
sig_rx = ch.transmit(sig_tx, rotation)
print(f"Rotation applied: {rotation} (in rad), {rotation * 180 / np.pi} (in deg)")
msg_un = rx.fftDizet(sig_rx, Q)
# print(f"Uncorrected msg: {msg_un}")

sig_ffo = rx.ffoEstCor(sig_rx, Q)
msg_rx = rx.fftDizet(sig_ffo, Q)
# ----- Zero-Marker rotation identification
# --- ZMDetection is giving BER even without NOISE check the implementation
# --- This METHOD doesn't work since the pilots and their plcaing have been changed
# integer_rotate = rx.ZMDetection(sig_ffo)

# msg_rotated = np.roll(msg_rx, -integer_rotate)
# print(f"Transmitted Message: {msg}, \nReceived Message: {msg_rx}, \n Rotated Message: {msg_rotated}")
# ber = rx.ber(msg_rotated, msg)
# print(f"BER: {ber}")

papr = tx.PAPR(sig_tx)
print(f"PAPR obtained for zero-pilot: {papr}")

# -- pz case wise implementations
# --- for CASE-1(a): BL is MULTIPLE of 4
# x = rx.fftConPZ(sig_ffo)
# print(f"Integer rotation estimate: {x}")
# msg_rotated = np.roll(msg_rx, x)
# print(f"\nTransmitted Message: {msg}, \n Rotated Message: {msg_rotated}, \n\nReceived Message: {msg_rx}")
# ber = rx.ber(msg_rotated, msg)
# print(f"BER: {ber}")

# x = rx.fftConPZ(sig_tx)
# print(f"Integer rotation estimate: {x}")
# msg_rotated = np.roll(msg, x)
# print(f"\nTransmitted Message: {msg}, \n Rotated Message: {msg_rotated}, \n\nReceived Message: {msg_rx}")
# ber = rx.ber(msg_rotated, msg)
# print(f"BER: {ber}")


##------------------Working code BLOCK ----------
# x = rx.BLodd(sig_ffo)
# print(f"Integer rotation estimate: {x}")
# msg_rotated = np.roll(msg_rx, x)
# print(f"\nTransmitted Message: {msg}, \n Rotated Message: {msg_rotated}, \n\nReceived Message: {msg_rx}")
# ber = rx.ber(msg_rotated, msg)
# print(f"BER: {ber}\n")


#  -- complete end to end: rotation estimation using pilot-zero no matter the 
# block-length and message decoding
msg_decoded = rx.PZDecodedMsg(sig_rx, Q).astype(int)
print(f"\nMessage Transmitted: {msg} \nMessage Decoded: {msg_decoded}")
ber = rx.ber(msg_decoded, msg)
print(f"BER: {ber}")

# ---- unable to find the correct relation to INTEGER ESTIMATE for EVEN and ODD BL
# x = rx.intRotationEst(sig_ffo)
# print(f"Integer rotation estimate: {x}")
# msg_rotated = np.roll(msg_rx, x)
# print(f"\nTransmitted Message: {msg}, \n Rotated Message: {msg_rotated}, \n\nReceived Message: {msg_rx}")
# ber = rx.ber(msg_rotated, msg)
# print(f"BER: {ber}")