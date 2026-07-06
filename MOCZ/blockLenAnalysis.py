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

K = np.arange(16, 41)
noIter = 2000
snr = 15
signal_power = 1
chEst = ChannelEstimation()

BER_15 = {}
PAPR_15 = {}
chCoeff_15 ={}
mae_minc = {}   

# to indentify what the MAE is for estimated channel coefficients 
# relative to minimum value from the coefficients
for k in K:
    tx = BMOCZTransmitter(k)
    rx = BMOCZReceiver(k)
    ber, papr, chError = 0, 0, 0
    for i in range(noIter):
        if i % noIter == 0:
            print(f" {i} Iterations Done for K = {k}")
        # msg = [np.random.randint(2) for i in range(k)]
        msg = np.random.randint(0, 2, k, dtype=np.uint8)

        sig_tx = tx.coeffCon(msg)
        sig_power = np.mean(np.abs(sig_tx)**2)
        sig_tx /= np.sqrt(sig_power)
        # print(f"Signal Power: {sig_power}, Signal after normalizing: {np.mean(np.abs(sig_tx)**2)}")
        noise_var = signal_power * 10**(-snr/10)

        ch = SlowFadingChannel(noise_var)
        sig_rx, ch_coeff = ch.transmit(sig_tx)

        Q = int( 2**( np.log( np.ceil(len(sig_rx)/k) ) / np.log(2) ) )

        sig_ffo = rx.ffoEstCor(sig_rx, Q)
        msg_rx = rx.fftDizet(sig_ffo, Q)

        ber += rx.ber(msg_rx, msg)
        papr += tx.PAPR(sig_tx)
        # ----   Signal Reconstruction using the same BMOCZTransmitter Class  -----
        sig_recon = tx.coeffCon(msg_rx)
        sig_power = np.mean(np.abs(sig_recon)**2)
        sig_recon /= np.sqrt(sig_power)

        ch_coeff_hat = chEst.leastSquares(sig_rx, sig_recon)
        # ch_coeff_hat = chEst.modifiedLS(sig_rx, sig_recon)
        if not np.isnan(ch_coeff_hat):
            chError += np.abs(ch_coeff_hat - ch_coeff)
            index = np.floor( np.log10(np.min(np.abs(sig_recon))) ).astype(int)
            if index not in mae_minc:
                arr = np.array([np.abs(ch_coeff_hat - ch_coeff), 1])
                mae_minc[index] = arr
            else:
                mae_minc[index][0] += np.abs(ch_coeff_hat - ch_coeff)
                mae_minc[index][1] += 1
        else:
            index = -100
            if index not in mae_minc:
                arr = np.array([-np.inf, 1])
                mae_minc[index] = arr
            else:
                mae_minc[index][0] += -np.inf
                mae_minc[index][1] += 1
    BER_15[k] = ber / noIter
    PAPR_15[k] = papr / noIter
    chCoeff_15[k] = chError / noIter

plt.figure(1, dpi=800)
plt.plot(BER_15.keys(), BER_15.values(), '-')
plt.grid(True)
plt.xlabel("Block-Length(K)")
plt.ylabel("BER")
plt.title(f"BER vs Block-Length(K) over {noIter} iterations for SNR = {snr}.")
plt.savefig("results/BER_s15.jpeg")


plt.figure(2,dpi=800)
plt.plot(PAPR_15.keys(), PAPR_15.values(), '-')
plt.grid(True)
plt.ylabel("Peak to Average Power Ratio (PAPR)")
plt.xlabel("Block-Length(K)")
plt.title(f"PAPR vs Block-Length(K) over {noIter} iterations for SNR = {snr}.")
plt.savefig("results/PAPR_s15.jpeg")

plt.figure(3, dpi=800)
plt.plot(chCoeff_15.keys(), chCoeff_15.values(), '-')
plt.grid(True)
plt.xlabel("Block-Length(K)")
# plt.ylim(0, 30)
plt.ylabel("MAE of channel coefficicent(|h|)")
plt.title(f"MAE of |h| vs Block-Length(K) over {noIter} iterations for SNR = {snr}.")
plt.savefig("results/MAE_h_s15.jpeg")

# plt.show()
mae_minc = dict(sorted(mae_minc.items(), reverse=False))
for k, v in mae_minc.items():
    print(f"Order of smallest coefficient in transmitted signal: {k}"
            f", MAE wrt to the order: {v[0] / v[1]}, No. of occurances: {v[1]}")