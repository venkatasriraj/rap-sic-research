"""
(CFO + AWGN) Simulation analysis for binary MOCZ schemes: 
    BMOCZ-ACPC(31, 16), JBMOCZ, SBMOCZ, BMOCZ-PZ
Since the allowed block-length for ACPC codes are [3, 7, 15, 31, 127],
we will be comparing the schems with the above block-lengths.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from wirelessComm import (
    BMOCZ, JBMOCZ, SBMOCZ, ACPC,
    PerformanceParameters, SlowFadingChannel
)

FLAG = "CFO"
m, t, Q = 5, 2, 32
zetaJ, zetaS, PZrad = 1.07, 0.0545, 1.578
Rj, Rs = 1.044, 1.0475
SNR_dB = np.arange(-5, 41, 5)
Lh, chVar, noIter = 1, 0.86, int(1e4)
schemes = ['BMOCZ-ACPC', 'JBMOCZ', 'SBMOCZ', 'BMOCZ-PZ']
K = 2**m-1
BW = math.lcm(K+1, K+2)
pktRate_PZ = BW / (K+2)
pktRate = BW / (K+1)
bmoczSys = BMOCZ(K, PZrad=0)
bmoczPZSys = BMOCZ(K, PZrad=PZrad)
jbmoczSys = JBMOCZ(K, zetaJ, Rj=Rj)
sbmoczSys = SBMOCZ(K, zetaS, Rs=Rs)
acpc = ACPC(m, t)
perParam = PerformanceParameters()
colors = ["tab:red", "tab:green", "tab:blue", 'tab:purple']
linestyles = ["--", "-.", "-", ":"]
markers = ["o", "s", "^", "d"]
ber, per, goodput, papr = {}, {}, {}, {}
for scheme in schemes:
    ber_snr, per_snr, goodput_snr, papr_snr = {}, {}, {}, {} 
    for snr in SNR_dB:
        noiseVar = 10**(-snr/10)
        ch = SlowFadingChannel(noise_var=noiseVar, taps=Lh, chVar=chVar)
        if scheme == 'BMOCZ-ACPC':
            result = bmoczSys.simulatorACPC(noIter, perParam, ch, acpc, Q, FLAG=FLAG)
            goodput_snr[snr] = result['pcr'] * (K - 5*(t+1)) * pktRate
        elif scheme == 'JBMOCZ':
            result = jbmoczSys.simulator(noIter, perParam, ch, FLAG=FLAG)
            goodput_snr[snr] = result['pcr'] * K * pktRate
        elif scheme == 'SBMOCZ':
            result = sbmoczSys.simulator(noIter, perParam, ch, FLAG=FLAG)
            goodput_snr[snr] = result['pcr'] * K * pktRate
        elif scheme == 'BMOCZ-PZ':
            result = bmoczPZSys.simulator(noIter, perParam, ch, Q, FLAG=FLAG)
            goodput_snr[snr] = result['pcr'] * K * pktRate_PZ
        ber_snr[snr], per_snr[snr], papr_snr[snr] = result['ber'], 1 - result['pcr'], result['papr'] 
    print(f"{scheme} Done")
    ber[scheme] = ber_snr
    per[scheme] = per_snr
    papr[scheme] = papr_snr
    goodput[scheme] = goodput_snr

plt.figure(1, dpi=800)
for i, (k, v) in enumerate(ber.items()):
    plt.semilogy(v.keys(), v.values(), linewidth=0.9, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=5, label=k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("BER")
plt.title(f"{FLAG} BER analysis K-{K}, Zs-{zetaS}, Zj-{zetaJ}, Lh-{Lh}, chVar-{chVar}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.ylim(1e-5, 1)
plt.tight_layout()
plt.savefig(f"results/conSim/FS/Lh{Lh}/{FLAG}berK{K}.jpeg")

plt.figure(2, dpi=800)
for i, (k, v) in enumerate(per.items()):
    plt.semilogy(v.keys(), v.values(), linewidth=0.9, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=5, label=k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("PER")
plt.title(f"{FLAG} PER analysis K-{K}, Zs-{zetaS}, Zj-{zetaJ}, Lh-{Lh}, chVar-{chVar}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.ylim(1e-5, 1)
plt.tight_layout()
plt.savefig(f"results/conSim/FS/Lh{Lh}/{FLAG}perK{K}.jpeg")

plt.figure(3, dpi=800)
for i, (k, v) in enumerate(papr.items()):
    plt.plot(v.keys(), v.values(), linewidth=0.9, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=5, label=k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("PAPR (dB)")
plt.title(f"{FLAG} PAPR analysis K-{K}, Zs-{zetaS}, Zj-{zetaJ}, Lh-{Lh}, chVar-{chVar}")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"results/conSim/FS/Lh{Lh}/{FLAG}paprK{K}.jpeg")

plt.figure(4, dpi=800)
for i, (k, v) in enumerate(goodput.items()):
    plt.plot(v.keys(), v.values(), linewidth=0.9, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=5, label=k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("Goodput")
plt.title(f"{FLAG} Goodput analysis K-{K}, Zs-{zetaS}, Zj-{zetaJ}, Lh-{Lh}, chVar-{chVar}")
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f"results/conSim/FS/Lh{Lh}/{FLAG}goodputK{K}.jpeg")