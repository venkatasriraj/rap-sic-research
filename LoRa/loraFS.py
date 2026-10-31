"""
LoRa implementation for different channel models
NOTE: calculate the throughpt by considering the symbol duration
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import LoRa, SlowFadingChannel, PerformanceParameters

FLAG = 'FS'
SF = np.arange(8, 15, 2)
BW, Lh, chVar, noIter = 125e3, 2, 0.95, int(1e4)
SNR_dB = np.arange(-40, 0, 3)
initialFreq = -0.5 * BW
sampRate = BW * 1
perParam = PerformanceParameters()
ber, per, thr = {}, {}, {}
for sf in SF:
    noChirps = 2**sf
    symDuration = noChirps / BW
    chirpRate = BW / symDuration
    sampPerSym = sampRate * symDuration
    loraSys = LoRa(sf, BW, sampRate, initialFreq)
    berSNR, perSNR, thrSNR = {}, {}, {}
    for snr in SNR_dB:
        noiseVar = 10**(-snr/10)
        ch = SlowFadingChannel(noise_var=noiseVar, taps=Lh, chVar=chVar)
        results = loraSys.simulator(noIter, perParam, ch, FLAG=FLAG)

        berSNR[snr] = results['ber']
        perSNR[snr] = 1 - results['pcr']
        thrSNR[snr] = results['pcr'] * sf / noChirps
    print(f'SF: {sf} done.')
    ber[sf] = berSNR
    per[sf] = perSNR
    thr[sf] = thrSNR

plt.figure(1, dpi=800)
for k, v in ber.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'SF-{k}')
plt.ylim(1e-4, 1)
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("BER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"LoRa {FLAG} BER Analysis Lh-{Lh} chVar-{chVar} BW-{BW}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/FS/Lh{Lh}/{FLAG}ber_{chVar}.jpeg")

plt.figure(2, dpi=800)
for k, v in per.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'SF-{k}')
plt.ylim(1e-4, 1)
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("PER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"LoRa {FLAG} PER Analysis Lh-{Lh} chVar-{chVar} BW-{BW}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/FS/Lh{Lh}/{FLAG}per_{chVar}.jpeg")

plt.figure(3, dpi=800)
for k, v in thr.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'SF-{k}')
plt.ylim(1e-4, 1)
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("Throughput normalised wrt symbol duration")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"LoRa {FLAG} Throughput Analysis Lh-{Lh} chVar-{chVar} BW-{BW}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/FS/Lh{Lh}/{FLAG}thr_{chVar}.jpeg")