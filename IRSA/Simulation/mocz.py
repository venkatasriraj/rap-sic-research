"""
Implementation of CRDSA with MOCZ using the designed Packages
"""

import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import (
    BMOCZTransmitter,
    BMOCZReceiver,
    SlowFadingChannel,
    ChannelEstimation
)

K = 32
Q = 4
sig_power = 1 
SNR_dB = np.arange(-10, 21, 2)
noIter = 10000
ch_var = 1

tx = BMOCZTransmitter(K)

rx = BMOCZReceiver(K)
chEst = ChannelEstimation()

per = {}; con_mae = {}; pktDetect = {}; coeff_mae = {}
for snr in SNR_dB:
    pcr = [1e-12, 1e-12]; con_error = 0; err_count = 0; coeff_error = 0
    noise_var = sig_power * 10**(-snr/10)
    ch = SlowFadingChannel(noise_var)
    for i in range(noIter):
        msgU1 = np.random.randint(0, 2, K)
        msgU2 = np.random.randint(0, 2, K)

        sigU1 = tx.coeffCon(msgU1)
        sigU1_power = np.mean(np.abs(sigU1)**2)
        sigU1_norm = sigU1 / np.sqrt(sigU1_power)

        sigU2 = tx.coeffCon(msgU2)
        sigU2_power = np.mean(np.abs(sigU2)**2)
        sigU2_norm = sigU2 / np.sqrt(sigU2_power)

        ch_coeffU1 = np.sqrt(ch_var/2) * ( np.random.randn() + 1j*np.random.randn() )
        ch_coeffU2 = np.sqrt(ch_var/2) * ( np.random.randn() + 1j*np.random.randn() )

        sig_rxS1 = ch.sic_transmit([sigU1_norm], [ch_coeffU1])

        sig_rxS2 = ch.sic_transmit([sigU1_norm, sigU2_norm], [ch_coeffU1, ch_coeffU2])

        msgS1 = rx.fftDizet(sig_rxS1, Q)
        
        sig_reconS1 = tx.coeffCon(msgS1)
        sig_reconS1_power = np.mean(np.abs(sig_reconS1)**2)
        sig_reconS1 /= np.sqrt(sig_reconS1_power)

        chEstS1 = chEst.leastSquares(sig_rxS1, sig_reconS1)
        coeff_error += np.abs(chEstS1 - ch_coeffU1)

        con_error += rx.mae(sig_rxS1, sig_reconS1 * chEstS1) / np.mean(np.abs(sig_rxS1))

        sig_recovS2 = sig_rxS2 - chEstS1 * sig_reconS1
        msgS2 = rx.fftDizet(sig_recovS2, Q)

        if rx.ber(msgS1, msgU1) == 0:
            pcr[0] += 1
        if rx.ber(msgS2, msgU2) == 0:
            pcr[1] += 1
        
    per[snr] = 1 - ( np.array(pcr) / noIter)
    pktDetect[snr] = np.array(pcr) / noIter
    con_mae[snr] = con_error / noIter
    coeff_mae[snr] = coeff_error / noIter
    print(f"SNR: {snr} done")

plt.figure(1, dpi=800)
plt.plot(pktDetect.keys(), pktDetect.values(), '-', linewidth=0.9)
plt.xlabel("SNR(dB)")
plt.ylabel("Throughput(T)")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"Throughput vs SNR")
plt.savefig("results/detectPkts.jpeg")

plt.figure(2, dpi=800)
for i in range(2):
    y_val = [val[i] for val in per.values()]
    plt.plot(per.keys(), y_val, '-', linewidth=0.9, label=f'Slot-{i}')
plt.xlabel("SNR(dB)")
plt.ylabel("PER")
plt.title(f"PER vs SNR over {noIter} frames")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/SICper.jpeg")

plt.figure(3, dpi=800)
plt.plot(con_mae.keys(), con_mae.values(), '-', linewidth=0.9)
plt.grid(True, alpha=0.6, linestyle='--')
plt.xlabel("SNR(dB)")
plt.ylabel("Signal Reconstruction Error")
plt.title("MAE of pkt recon vs SNR")
plt.savefig("results/coeffConErr.jpeg")

plt.figure(4, dpi=800)
plt.plot(coeff_mae.keys(), coeff_mae.values(), '-', linewidth=0.9)
plt.xlabel("SNR(dB)")
plt.ylabel("MAE of h")
plt.title("MAE of h vs SNR")
plt.grid(True, alpha=0.6, linestyle='--')
plt.savefig("results/coeffEstErr.jpeg")