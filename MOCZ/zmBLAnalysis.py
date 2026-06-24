"""
Monte-carlo simulations for Zero-Marker in identifying and correcing the 
phase rotation due to CFO.
We will be analyzing the system performance over a given block-length.
For a block-length(K) we will be analyzing how
- BER vs K
- PAPR vs K

NOTE: Channel estimation is halted in the case of multipath fading 
and will be persued later. 
- MSE of h_est vs K for SNR = 15dB.

- The choice of Q for DiZeT decoder is also something to be looked at.
"""
import numpy as np
import matplotlib.pyplot as plt

from BMOCZ import (
    BMOCZReceiver,
    BMOCZTransmitter
)
from CHANNEL import (
    MultiPathFading,
    # ChannelEstimation
)

K = np.arange(2, 41)
Q = 7
noIter = 1000
snr = 15
signal_power = 1 
noise_var = signal_power * 10**(-snr/10)

ch = MultiPathFading(noise_var=noise_var)
# chEst = ChannelEstimation()
BER_15 = {}
PAPR_15 = {}
# chCoeff_15 = {}

for k in K:
    tx = BMOCZTransmitter(k)
    rx = BMOCZReceiver(k)
    ber, papr, mse = 0, 0, 0
    for i in range(noIter):
        msg = [np.random.randint(2) for i in range(k)]

        sig_tx = tx.coeffConZM(msg)
        sig_power = np.mean(np.abs(sig_tx)**2)
        sig_norm = sig_tx / sig_power

        rotation = np.random.uniform(0, 2*np.pi)
        sig_rx = ch.transmit(sig_norm, rotation)

        sig_ffo = rx.ffoEstCor(sig_rx, Q)
        msg_rx = rx.fftDizet(sig_ffo, Q)

        int_est = rx.ZMDetection(sig_ffo)
        msg_hat = np.roll(msg_rx, -int_est)

        ber += rx.ber(msg_hat, msg)
        papr += tx.PAPR(sig_tx)
    BER_15[k] = ber / noIter
    PAPR_15[k] = papr / noIter

plt.figure(1, dpi=400)
plt.plot(BER_15.keys(), BER_15.values(), '-')
plt.grid(True)
plt.xlabel("Block-Length(K)")
plt.ylabel("BER")
plt.title(f"BER vs K over {noIter} iterations for SNR {snr}.")
plt.savefig("results/zmBER.jpeg")

plt.figure(2, dpi=400)
plt.plot(PAPR_15.keys(), PAPR_15.values(), '-')
plt.grid(True)
plt.xlabel("Block-Length(K)")
plt.ylabel("Peak to Average Power Ratio (PAPR)")
plt.title(f"PAPR vs K over {noIter} iterations for SNR {snr}")
plt.savefig("results/zmPAPR.jpeg")

# plt.show()