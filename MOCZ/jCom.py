"""
Comparision of Heuristic Algorithms of jutted: modified template and IRO against ACPC code
#peaks in modified template = #channel zeros + 2
symbol duration: Ts = (K+1)/W
symbol rate: Rs = W / (K+1)
bit rate: Rb = W * K / (K+1) # non ACPC schemes
             = W * ( K - (t+1)*m) / (K+1) # ACPC scheme
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from wirelessComm import JBMOCZ, SlowFadingChannel, PerformanceParameters, ACPC, BMOCZ, JUTTEDfs

FLAG = "FS_CFO"
m, t, Q = 5, 2, 32
SNR_dB = np.arange(-1, 42, 4)
Lh, chVar = 6, 0.86
zeta, Rj, noIter = 1.15, 1.044, int(1e4)
Peak = [0, 7, 17, 23]
# schemes = ['jIRO']
schemes = ['BMOCZ-ACPC', 'JBMOCZ', 'M-JBMOCZ', 'DiZeT-CFO']
K = 2**m-1
ENERGY = K + Lh
BW = math.lcm(K+1, K+2)
pktRate_PZ = BW / (K+2)
pktRate = BW / (K+1)
bmoczSys = BMOCZ(K, PZrad=0)
perParam = PerformanceParameters()
acpc = ACPC(m, t)
jbmcozSystem = JBMOCZ(K, zeta, Rj=Rj, mPeak=Peak)
jSystem = JUTTEDfs(K, zeta, Rj=Rj, multiplePeak=False)
juttedRef = JBMOCZ(K, zeta, Rj=Rj)
colors = ["black", "tab:green", "tab:blue", "tab:purple", 'tab:orange']
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 2, 4))]
markers = ["", "o", "s", "d", "1", "x", "*", "^", '+']
markerColor = ["None", "None", "tab:blue", "tab:purple"]
ber, per, papr, goodput, rotation = {}, {}, {}, {}, {}
for scheme in schemes:
    berSNR, perSNR, paprSNR, goodputSNR, rotationSNR = {}, {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = 10**(-snr/10)
        ch = SlowFadingChannel(noise_var=noiseVar, taps=Lh, chVar=chVar)
        if scheme == 'BMOCZ-ACPC':
            results = bmoczSys.simulatorACPC(noIter, perParam, ch, acpc, Energy=ENERGY, Q=Q, FLAG=FLAG)
            # goodputSNR[snr] = results['pcr'] * (K - 5*(t+1)) / K # * pktRate
            goodputSNR[snr] = results['pcr'] * (K - 5*(t+1)) / (K + Lh)
        elif scheme == 'M-JBMOCZ':
            results = jbmcozSystem.simulator(noIter, perParam, ch, Energy=ENERGY, FLAG=FLAG)
            # goodputSNR[snr] = results['pcr'] * K / K # * pktRate
            goodputSNR[snr] = results['pcr'] * K / (K + Lh)
        elif scheme == 'DiZeT-CFO':
            results = jSystem.simulator(noIter, perParam, ch, Energy=ENERGY, Q=Q, FLAG=FLAG)
            # goodputSNR[snr] = results['pcr'] * K / K # * pktRate
            goodputSNR[snr] = results['pcr'] * K / (K + Lh)
        elif scheme == 'JBMOCZ':
            results = juttedRef.simulator(noIter, perParam, ch, Energy=ENERGY, FLAG=FLAG)
            # goodputSNR[snr] = results['pcr'] * K / K # * pktRate
            goodputSNR[snr] = results['pcr'] * K / (K + Lh)
        berSNR[snr] = results['ber']
        perSNR[snr] = 1 - results['pcr']
        paprSNR[snr] = results['papr']
        rotationSNR[snr] = results['rotationEst']
    print(f"{scheme} Done")
    ber[scheme] = berSNR
    per[scheme] = perSNR
    papr[scheme] = paprSNR
    goodput[scheme] = goodputSNR
    rotation[scheme] = rotationSNR

plt.rcParams.update({
    "font.family": "serif",
    # "font.size": 26,
    "axes.labelsize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    # "legend.fontsize": 25,
    # "axes.titlesize": 20,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

plt.figure(1, dpi=800)
for i, (k, v) in enumerate(ber.items()):
    plt.semilogy(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=8, label=k, markerfacecolor=markerColor[i])
plt.ylim(1e-4, 1)
plt.xlim(0, 40)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"{FLAG} Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
plt.legend(loc='lower left', framealpha=0.5, fontsize=18)
plt.tight_layout()
plt.savefig(f"results/conSim/heuristicsJ/ber_{Lh}_{Peak}.pdf", bbox_inches="tight")

plt.figure(2, dpi=800)
for i, (k, v) in enumerate(per.items()):
    plt.semilogy(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=8, label=k, markerfacecolor=markerColor[i])
plt.ylim(1e-4, 1)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.xlim(0, 40)
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"{FLAG} Analysis chVar-{chVar} R-{Rj}, Z-{zeta} {Peak}")
plt.legend(loc='lower left', framealpha=0.5, fontsize=18)
plt.tight_layout()
plt.savefig(f"results/conSim/heuristicsJ/per_{Lh}_{Peak}.pdf", bbox_inches="tight")

plt.figure(3, dpi=800)
for i, (k, v) in enumerate(papr.items()):
    plt.plot(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=8, label=k, markerfacecolor=markerColor[i])
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR (dB)")
plt.xlim(0, 40)
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"PAPR Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
plt.legend(loc='lower left', framealpha=0.5, fontsize=18)
plt.tight_layout()
plt.savefig(f"results/conSim/heuristicsJ/papr_{Lh}_{Peak}.pdf", bbox_inches="tight")

plt.figure(4, dpi=800)
for i, (k, v) in enumerate(goodput.items()):
    plt.plot(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=8, label=k, markerfacecolor=markerColor[i])
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.xlabel("SNR (dB)")
plt.ylabel("Throughput (bits/sec)")
plt.ylim(-0.01, 1.01)
plt.xlim(0, 40)
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"Goodput Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
plt.legend(loc='upper left', framealpha=0.5, fontsize=18)
plt.tight_layout()
plt.savefig(f"results/conSim/heuristicsJ/goodput_{Lh}_{Peak}.pdf", bbox_inches="tight")

plt.figure(5, dpi=800)
for i, (k, v) in enumerate(rotation.items()):
    plt.semilogy(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=8, label=k, markerfacecolor=markerColor[i])
plt.ylim(1e-4, 1)
plt.xlim(0, 40)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.xlabel("SNR (dB)")
plt.ylabel("NMAE")
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"{FLAG} Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
plt.legend(loc='lower left', framealpha=0.5, fontsize=18)
plt.tight_layout()
plt.savefig(f"results/conSim/heuristicsJ/rotation_{Lh}_{Peak}.pdf", bbox_inches="tight")