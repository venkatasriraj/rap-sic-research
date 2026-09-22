"""
Simulation analysis for binary MOCZ schemes: 
    BMOCZ-ACPC(31, 16), JBMOCZ, SBMOCZ, BMOCZ-PZ
Since the allowed block-length for ACPC codes are [3, 7, 15, 31, 127],
we will be comparing the schems with the above block-lengths.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from wirelessComm import (
    BMOCZ, JBMOCZ, SBMOCZ, ACPC,
    PerformanceParameters, MultiPathFading
)
m, t, signalPower = 5, 2, 1
zetaJ, zetaS, PZrad = 1.07, 0.0545, 1.3
noIter = int(1e2)
SNR_dB = np.arange(-5, 31, 5)
schemes = ['BMOCZ-ACPC', 'JBMOCZ', 'SBMOCZ', 'BMOCZ-PZ']
K = 2**m-1
Q = 32
BW = math.lcm(K+1, K+2)
pktRate_PZ = BW / (K+2)
pktRate = BW / (K+1)
bmoczSys = BMOCZ(K)
jbmoczSys = JBMOCZ(K, zetaJ)
sbmoczSys = SBMOCZ(K, zetaS)
acpc = ACPC(m, t)
perParam = PerformanceParameters()
colors = ["tab:red", "tab:green", "tab:blue", 'tab:purple']
linestyles = ["--", "-.", "-", ":"]
markers = ["o", "s", "^", "d"]
ber, per, goodput, papr = {}, {}, {}, {}
for scheme in schemes:
    ber_snr, per_snr, goodput_snr, papr_snr = {}, {}, {}, {} 
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = MultiPathFading(noise_var=noiseVar)
        if scheme == 'BMOCZ-ACPC':
            result = bmoczSys.simulatorACPC(noIter, perParam, ch, acpc, Q)
            goodput_snr[snr] = result['pcr'] * (K - 5*(t+1)) * pktRate
        elif scheme == 'JBMOCZ':
            result = jbmoczSys.simulator(noIter, perParam, ch)
            goodput_snr[snr] = result['pcr'] * K * pktRate
        elif scheme == 'SBMOCZ':
            result = sbmoczSys.simulator(noIter, perParam, ch)
            goodput_snr[snr] = result['pcr'] * K * pktRate
        elif scheme == 'BMOCZ-PZ':
            result = bmoczSys.simulator(noIter, perParam, ch, PZrad)
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
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title(f"BER analysis for BlockLen {K}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.ylim(1e-5, 1)
plt.tight_layout()
plt.savefig(f"results/conSim/berK{K}.jpeg")

plt.figure(2, dpi=800)
for i, (k, v) in enumerate(per.items()):
    plt.semilogy(v.keys(), v.values(), linewidth=0.9, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=5, label=k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.title(f"PER analysis for BlockLen {K}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.ylim(1e-5, 1)
plt.tight_layout()
plt.savefig(f"results/conSim/perK{K}.jpeg")

plt.figure(3, dpi=800)
for i, (k, v) in enumerate(papr.items()):
    plt.plot(v.keys(), v.values(), linewidth=0.9, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=5, label=k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR")
plt.title(f"PAPR analysis for BlockLen {K}")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"results/conSim/paprK{K}.jpeg")

plt.figure(4, dpi=800)
for i, (k, v) in enumerate(goodput.items()):
    plt.plot(v.keys(), v.values(), linewidth=0.9, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=5, label=k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR (dB)")
plt.ylabel("Goodput")
plt.title(f"Goodput analysis for BlockLen {K}")
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f"results/conSim/goodputK{K}.jpeg")