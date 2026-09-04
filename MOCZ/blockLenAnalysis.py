"""
Monte-carlo simulations for estimated channel over varying block-length(K).
We will be considering the slow-fading channel with AWGN.
These values are calculate without normalizing the signal coefficicents in time-domain.
PAPR analysis should be at Transmitter before PA and DAC.
For SNR =  we will be analyzing how
- BER vs K
- PAPR vs K
- MSE of h_est vs K
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import (
    BMOCZ, PerformanceParameters,
    SlowFadingChannel, ChannelEstimation
)
BER_15 = {}; PAPR_15 = {}; PER_15 = {}; chCoeff_15 ={}; mae_minc = {}   

K = np.arange(6, 41, 1)
Q = 4
noIter = int(1e1)
snr = 10
pathLoss = 1
signal_power = 1
noise_var = signal_power * 10**(-snr/10)
perParam = PerformanceParameters()
ch = SlowFadingChannel(noise_var, pathLoss)
chEst = ChannelEstimation()
for k in K:
    bmoczSystem = BMOCZ(k)
    ber, papr, chError, pcr = 0, 0, 0, 0
    for i in range(noIter):
        msg = np.random.randint(0, 2, k, dtype=np.uint8)

        sig_tx = bmoczSystem.coeffCon(msg)
        sig_power = np.mean(np.abs(sig_tx)**2)
        sig_tx /= np.sqrt(sig_power)
        
        sig_rx, ch_coeff = ch.transmit(sig_tx)

        # Q = int( 2**( np.log( np.ceil(len(sig_rx)/k) ) / np.log(2) ) )  # Q = 2

        sig_ffo = bmoczSystem.ffoEstCor(sig_rx, Q)
        msg_rx = bmoczSystem.fftDizet(sig_ffo, Q)

        # ----   Signal Reconstruction using the same BMOCZTransmitter Class  -----
        sig_recon = bmoczSystem.coeffCon(msg_rx)
        sig_power = np.mean(np.abs(sig_recon)**2)
        sig_recon /= np.sqrt(sig_power)

        ch_coeff_hat = chEst.leastSquares(sig_rx, sig_recon)
        chError += np.abs(ch_coeff_hat - ch_coeff) / np.abs(ch_coeff)
        # --- MODIFIED LS is not required Since we've updated our LS ESTIMATOR for h --------
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
        pcr += perParam.pcr(msg_rx, msg)
        ber += perParam.ber(msg_rx, msg)
        papr += bmoczSystem.PAPR(sig_tx)
    BER_15[k] = ber / noIter
    PER_15[k] =  1 - (pcr / noIter)
    PAPR_15[k] = papr / noIter
    chCoeff_15[k] = chError / noIter
    print(f"Msg-len: {k} done")

plt.figure(1, dpi=800)
plt.plot(BER_15.keys(), BER_15.values(), '-')
plt.grid(True)
plt.xlabel("Msg-Length(K)")
plt.ylabel("BER")
plt.title(f"{noIter} packets per point for SNR = {snr}dB.")
plt.savefig(f"results/BMOCZ/BLAnalysis/BER_s{snr}.jpeg")

plt.figure(2,dpi=800)
plt.plot(PAPR_15.keys(), PAPR_15.values(), '-')
plt.grid(True)
plt.ylabel("Peak to Average Power Ratio (PAPR)")
plt.xlabel("Msg-Length(K)")
plt.title(f"{noIter} packets per point for SNR = {snr}dB.")
plt.savefig(f"results/BMOCZ/BLAnalysis/PAPR_s{snr}.jpeg")

plt.figure(3, dpi=800)
plt.plot(chCoeff_15.keys(), chCoeff_15.values(), '-')
plt.grid(True)
plt.xlabel("Msg-Length(K)")
plt.ylabel("MAE of channel coefficicent(|h|)")
plt.title(f"{noIter} packets per point for SNR = {snr}dB.")
plt.savefig(f"results/BMOCZ/BLAnalysis/MAE_h_s{snr}.jpeg")

plt.figure(4, dpi=800)
plt.plot(PER_15.keys(), PER_15.values(), '-')
plt.grid(True)
plt.xlabel("Msg-Length(K)")
plt.ylabel("PER")
plt.title(f"{noIter} packets per point for SNR = {snr}dB.")
plt.savefig(f"results/BMOCZ/BLAnalysis/PER_s{snr}.jpeg")

# plt.show()
# mae_minc = dict(sorted(mae_minc.items(), reverse=False))
# for k, v in mae_minc.items():
#     print(f"Order of smallest coefficient in transmitted signal: {k}"
#             f", MAE wrt to the order: {v[0] / v[1]}, No. of occurances: {v[1]}")