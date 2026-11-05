"""
MJ-BMOCZ simulations
mode-0: SNR
mode-1: Eb/N0
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import (
    JBMOCZ, SlowFadingChannel, PerformanceParameters
)

FLAG = "FS_CFO"
Ns, K = 256, 31
EbN0_dB = np.arange(-1, 42, 2)
mode = 1
Lh, chVar = 6, 0.86
zeta, Rj, noIter = 1.15, 1.044, int(8e4)
peaks = {1: [0], 2: [0, 3, 7], 3: [0, 5, 10, 15], 4: [0, 7, 17, 23], 5: [0, 7, 15, 21, 27]}
perParam = PerformanceParameters()
colors = ["tab:pink", "tab:green", "tab:blue", "black", 'tab:orange']
linestyles = [":", "--", "-.", "-", (0, (3, 1, 2, 4))]
markers = ["1", "s", "+", "", "*", "x", "^", 'd']
# markerColor = ["None", "None", "tab:blue", "tab:purple", ]
ber, per, papr, goodput, rotation = {}, {}, {}, {}, {}
infoBits, ENERGY = K, K + Lh
Rb_BW = infoBits / ENERGY
for PEAK in peaks.keys():
    jbmcozSystem = JBMOCZ(K, zeta, Ns=Ns, Rj=Rj, mPeak=peaks[PEAK])
    berSNR, perSNR, paprSNR, goodputSNR, rotationSNR = {}, {}, {}, {}, {}
    for ebn0 in EbN0_dB:
        if mode == 0:
            snr = ebn0
        else:
            snr = ebn0 + 10 * np.log10(Rb_BW)
        noiseVar = 10**(-snr/10)
        ch = SlowFadingChannel(noise_var=noiseVar, taps=Lh, chVar=chVar)
        results = jbmcozSystem.simulator(noIter, perParam, ch, FLAG=FLAG)
        goodputSNR[snr] = results['pcr'] * Rb_BW
        berSNR[snr] = results['ber']
        perSNR[snr] = 1 - results['pcr']
        paprSNR[snr] = results['papr']
        rotationSNR[snr] = results['rotationEst']
    print(f"{PEAK} Done")
    ber[PEAK] = berSNR
    per[PEAK] = perSNR
    papr[PEAK] = paprSNR
    goodput[PEAK] = goodputSNR
    rotation[PEAK] = rotationSNR

plt.rcParams.update({
    "font.family": "serif",
    # "font.size": 26,
    "axes.labelsize": 26,
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
                color=colors[i], marker=markers[i], markersize=8, label=f'{peaks[k]}')
plt.ylim(1e-2, 1)
plt.xlim(0, 40)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
if mode == 0:
    plt.xlabel("SNR (dB)")
else:
    plt.xlabel(r'$E_b / N_0 [dB]$')
plt.ylabel("BER")
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"{FLAG} Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
plt.legend(loc='best', framealpha=0.5, fontsize=18)
plt.tight_layout()
plt.savefig(f"results/JBMOCZ/MJ/ber_{Lh}.pdf", bbox_inches="tight")

plt.figure(2, dpi=800)
for i, (k, v) in enumerate(per.items()):
    plt.semilogy(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=8, label=f'{peaks[k]}')
plt.ylim(1e-2, 1)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
if mode == 0:
    plt.xlabel("SNR (dB)")
else:
    plt.xlabel(r'$E_b / N_0 [dB]$')
plt.ylabel("BLER")
plt.xlim(0, 40)
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"{FLAG} Analysis chVar-{chVar} R-{Rj}, Z-{zeta} {Peak}")
plt.legend(loc='best', framealpha=0.5, fontsize=18)
plt.tight_layout()
plt.savefig(f"results/JBMOCZ/MJ/per_{Lh}.pdf", bbox_inches="tight")

plt.figure(3, dpi=800)
for i, (k, v) in enumerate(papr.items()):
    plt.plot(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=8, label=f'{peaks[k]}')
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
if mode == 0:
    plt.xlabel("SNR (dB)")
else:
    plt.xlabel(r'$E_b / N_0 [dB]$')
plt.ylabel("PAPR (dB)")
plt.xlim(0, 40)
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"PAPR Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
plt.legend(loc='best', framealpha=0.5, fontsize=18)
plt.tight_layout()
plt.savefig(f"results/JBMOCZ/MJ/papr_{Lh}.pdf", bbox_inches="tight")

plt.figure(4, dpi=800)
for i, (k, v) in enumerate(goodput.items()):
    plt.plot(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=8, label=f'{peaks[k]}')
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
if mode == 0:
    plt.xlabel("SNR (dB)")
else:
    plt.xlabel(r'$E_b / N_0 [dB]$')
plt.ylabel("Throughput (bits/sec)")
plt.ylim(-0.01, 1.01)
plt.xlim(0, 40)
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"Goodput Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
plt.legend(loc='best', framealpha=0.5, fontsize=18)
plt.tight_layout()
plt.savefig(f"results/JBMOCZ/MJ/goodput_{Lh}.pdf", bbox_inches="tight")

plt.figure(5, dpi=800)
for i, (k, v) in enumerate(rotation.items()):
    plt.semilogy(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=8, label=f'{peaks[k]}')
plt.ylim(1e-2, 1)
plt.xlim(0, 40)
if mode == 0:
    plt.xlabel("SNR (dB)")
else:
    plt.xlabel(r'$E_b / N_0 [dB]$')
plt.ylabel("MNAE")
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"{FLAG} Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
plt.legend(loc='best', framealpha=0.5, fontsize=18)
plt.tight_layout()
plt.savefig(f"results/JBMOCZ/MJ/rotation_{Lh}.pdf", bbox_inches="tight")