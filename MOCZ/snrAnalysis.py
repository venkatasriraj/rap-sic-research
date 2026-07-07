"""
Monte-carlo simulations for estimated channel over varying SNR.
We will be considering the slow-fading channel with AWGN.
These values are calculate without normalizing the signal coefficicents
in time-domain.
For K = 31 we will be analyzing how
- BER vs SNR
- PAPR vs SNR  # PAPR is not a function of SNR
- MSE of h_est vs SNR
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

SNR_dB = np.arange(-10, 21, 2)
noIter = 10
signal_power = 1
K = 32
tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
chEst = ChannelEstimation()

BER_32 = {}; PER_32 = {}; chCoeff_32 = {}; mae_minc = {}
for snr in SNR_dB:
    ber, chError, pcr = 0, 0, 0
    noise_var = signal_power * 10**(-snr/10)
    ch = SlowFadingChannel(noise_var)
    for i in range(noIter):
        msg = np.random.randint(0, 2, K, dtype=np.uint8)
        
        sig_tx = tx.coeffCon(msg)
        sig_power = np.mean( np.abs(sig_tx)**2 )
        sig_tx /= np.sqrt(sig_power)

        sig_rx, ch_coeff = ch.transmit(sig_tx)

        Q = int( 2**( np.log( np.ceil(len(sig_rx)/K) ) / np.log(2) ) )

        sig_ffo = rx.ffoEstCor(sig_rx, Q)
        msg_rx = rx.fftDizet(sig_ffo, Q)
        # ----   Signal Reconstruction using the same BMOCZTransmitter Class  -----
        sig_recon = tx.coeffCon(msg_rx)
        sig_power = np.mean(np.abs(sig_recon)**2)
        sig_recon /= np.sqrt(sig_power)

        ch_coeff_hat = chEst.leastSquares(sig_rx, sig_recon)
        # ch_coeff_hat = chEst.modifiedLS(sig_rx, sig_recon)
        # if not np.isnan(ch_coeff_hat):
        #     chError += np.abs(ch_coeff_hat - ch_coeff)
        #     index = np.floor( np.log10(np.min(np.abs(sig_recon))) ).astype(int)
        #     if index not in mae_minc:
        #         arr = np.array([np.abs(ch_coeff_hat - ch_coeff), 1])
        #         mae_minc[index] = arr
        #     else:
        #         mae_minc[index][0] += np.abs(ch_coeff_hat - ch_coeff)
        #         mae_minc[index][1] += 1
        # else:
        #     index = -100
        #     if index not in mae_minc:
        #         arr = np.array([-np.inf, 1])
        #         mae_minc[index] = arr
        #     else:
        #         mae_minc[index][0] += -np.inf
        #         mae_minc[index][1] += 1
        chError += np.abs(ch_coeff_hat - ch_coeff)
        ber += rx.ber(msg_rx, msg)
        pcr += rx.per(msg_rx, msg)
    BER_32[snr] = ber / noIter
    PER_32[snr] = 1 - (pcr / noIter)
    chCoeff_32[snr] = chError / noIter
    print(f"SNR: {snr} done")

plt.figure(1, dpi=800)
plt.plot(BER_32.keys(), BER_32.values(), '-')
plt.grid(True)
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.title(f"BER vs SNR over {noIter} iterations for K = {K}.")
plt.savefig(f"results/SNRAnalysis/BER_k{K}.jpeg")

plt.figure(3, dpi=800)
plt.plot(chCoeff_32.keys(), chCoeff_32.values(), '-')
plt.grid(True)
plt.xlabel("SNR(dB)")
plt.ylabel("MAE of channel coefficicent(|h|)")
plt.title(f"MAE of |h| vs SNR over {noIter} iterations for K = {K}.")
plt.savefig(f"results/SNRAnalysis/MAE_h_k{K}.jpeg")

plt.figure(3, dpi=800)
plt.plot(PER_32.keys(), PER_32.values(), '-')
plt.grid(True)
plt.xlabel("SNR(dB)")
plt.ylabel("PER")
plt.title(f"PER vs SNR over {noIter} iterations for K = {K}.")
plt.savefig(f"results/SNRAnalysis/PER_k{K}.jpeg")

# plt.show()
# mae_minc = dict(sorted(mae_minc.items(), reverse=False))
# for k, v in mae_minc.items():
#     print(f"Order of smallest coefficient in transmitted signal: {k}"
#             f", MAE wrt to the order: {v[0] / v[1]}, No. of occurances: {v[1]}")