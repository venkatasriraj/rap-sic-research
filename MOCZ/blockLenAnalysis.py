"""
Monte-carlo simulations for estimated channel over varying block-length(K).
We will be considering the slow-fading channel with AWGN.
These values are calculate without normalizing the signal coefficicents
in time-domain.
PAPR analysis should be at Transmitter before PA and DAC.
For SNR =  we will be analyzing how
- BER vs K
- PAPR vs K
- MSE of h_est vs K
"""

import numpy as np
import matplotlib.pyplot as plt

from BMOCZ import (
    BMOCZReceiver,
    BMOCZTransmitter
)
from CHANNEL import (
    SlowFadingChannel,
    ChannelEstimation
)

K = np.arange(7, 41)
noIter = 1000
snr = 15

chEst = ChannelEstimation()

BER_15 = {}
PAPR_15 = {}
chCoeff_15 ={}
for k in K:
    tx = BMOCZTransmitter(k)
    rx = BMOCZReceiver(k)
    ber = 0
    papr = 0
    chError = 0
    for i in range(noIter):
        msg = [np.random.randint(2) for i in range(k)]

        sig_tx = tx.coeffCon(msg)
        sig_power = np.mean(np.abs(sig_tx)**2)

        noise_var = sig_power * 10**(-snr/10)

        ch = SlowFadingChannel(noise_var)
        sig_rx, ch_coeff = ch.transmit(sig_tx)

        Q = int( 2**( np.log( np.ceil(len(sig_rx)/k) ) / np.log(2) ) )

        sig_ffo = rx.ffoEstCor(sig_rx, Q)
        msg_rx = rx.fftDizet(sig_ffo, Q)

        ber += rx.ber(msg_rx, msg)
        papr += tx.PAPR(sig_tx)
        # ----   Signal Reconstruction using the same BMOCZTransmitter Class  -----
        sig_recon = tx.coeffCon(msg_rx)
        ch_coeff_hat = chEst.leastSquares(sig_rx, sig_recon)
        chError += np.abs(ch_coeff_hat - ch_coeff)**2
    BER_15[k] = ber / noIter
    PAPR_15[k] = papr / noIter
    chCoeff_15[k] = chError / noIter

plt.figure(1, dpi=400)
plt.plot(BER_15.keys(), BER_15.values(), '-')
plt.grid(True)
plt.xlabel("Block-Length(K)")
plt.ylabel("BER")
plt.title(f"BER vs Block-Length(K) over {noIter} iterations for SNR = {snr}.")
plt.savefig("results/BER_15.jpeg")


plt.figure(2,dpi=400)
plt.plot(PAPR_15.keys(), PAPR_15.values(), '-')
plt.grid(True)
plt.ylabel("Peak to Average Power Ratio (PAPR)")
plt.xlabel("Block-Length(K)")
plt.title(f"PAPR vs Block-Length(K) over {noIter} iterations for SNR = {snr}.")
plt.savefig("results/PAPR_15.jpeg")

plt.figure(3, dpi=400)
plt.plot(chCoeff_15.keys(), chCoeff_15.values(), '-')
plt.grid(True)
plt.xlabel("Block-Length(K)")
plt.ylabel("MSE of channel coefficicent(|h|)")
plt.title(f"MSE of |h| vs Block-Length(K) over {noIter} iterations for SNR = {snr}.")
plt.savefig("results/MSE_h_15.jpeg")

# plt.show()