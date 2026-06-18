"""
Monte-carlo simulations for estimated channel over varying SNR.
We will be considering the slow-fading channel with AWGN.
These values are calculate without normalizing the signal coefficicents
in time-domain.
For K = 31 we will be analyzing how
- BER vs SNR
- PAPR vs SNR
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

SNR_dB = np.arange(0, 31)
noIter = 1000

K = 31

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
chEst = ChannelEstimation()

BER_31 = {}
PAPR_31 = {}
chCoeff_31 = {}
for snr in SNR_dB:
    ber = 0
    papr = 0
    chError = 0
    for i in range(noIter):
        msg = [np.random.randint(2) for i in range(K)]
        
        sig_tx = tx.coeffCon(msg)
        sig_power = np.mean( np.abs(sig_tx)**2 )

        noise_var = sig_power * 10**(-snr/10)

        ch = SlowFadingChannel(noise_var)
        sig_rx, ch_coeff = ch.transmit(sig_tx)

        Q = int( 2**( np.log( np.ceil(len(sig_rx)/K) ) / np.log(2) ) )

        sig_ffo = rx.ffoEstCor(sig_rx, Q)
        msg_rx = rx.fftDizet(sig_ffo, Q)

        ber += rx.ber(msg_rx, msg)
        papr += tx.PAPR(sig_tx)

        # ----   Signal Reconstruction using the same BMOCZTransmitter Class  -----
        sig_recon = tx.coeffCon(msg_rx)
        ch_coeff_hat = chEst.leastSquares(sig_rx, sig_recon)
        chError += np.abs(ch_coeff_hat - ch_coeff)**2
    BER_31[snr] = ber / noIter
    PAPR_31[snr] = papr / noIter
    chCoeff_31[snr] = chError / noIter

plt.figure(1, dpi=400)
plt.plot(BER_31.keys(), BER_31.values(), '-')
plt.grid(True)
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.title(f"BER vs SNR over {noIter} iterations for K = {K}.")
plt.savefig("results/BER_31.jpeg")


plt.figure(2,dpi=400)
plt.plot(PAPR_31.keys(), PAPR_31.values(), '-')
plt.grid(True)
plt.ylabel("Peak to Average Power Ratio (PAPR)")
plt.xlabel("SNR(dB)")
plt.title(f"PAPR vs SNR over {noIter} iterations for K = {K}.")
plt.savefig("results/PAPR_31.jpeg")

plt.figure(3, dpi=400)
plt.plot(chCoeff_31.keys(), chCoeff_31.values(), '-')
plt.grid(True)
plt.xlabel("SNR(dB)")
plt.ylabel("MSE of channel coefficicent(|h|)")
plt.title(f"MSE of |h| vs SNR over {noIter} iterations for K = {K}.")
plt.savefig("results/MSE_h_31.jpeg")

# plt.show()