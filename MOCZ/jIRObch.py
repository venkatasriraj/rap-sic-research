"""
forward error correction (FEC) code BCH for DiZeT-CFO
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import JUTTEDfs, PerformanceParameters, SlowFadingChannel, BCH

FLAG = "FS_CFO"
m, t, Q = 5, 2, 32
SNR_dB = np.arange(-1, 42, 3)
Lh, chVar = 6, 0.86
zeta, Rj, noIter = 1.15, 1.044, int(1e4)
Peak = [0, 7, 17, 23]
# schemes = ['jIRO']
schemes = ['BMOCZ-ACPC', 'JBMOCZ', 'M-JBMOCZ', 'DiZeT-CFO']
K = 2**m-1
bch = BCH(m, t)
jfs = JUTTEDfs(K, zeta, Rj=Rj)
perParam = PerformanceParameters()
colors = ["black", "tab:green", "tab:blue", "tab:purple", 'tab:orange']
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 2, 4))]
markers = ["", "o", "s", "d", "1", "x", "*", "^", '+']
markerColor = ["None", "None", "tab:blue", "tab:purple"]
noiseFlag = True
berSNR, perSNR, paprSNR, goodputSNR, rotationSNR = {}, {}, {}, {}, {}
for snr in SNR_dB:
    noiseVar = 10**(-snr/10)
    ch = SlowFadingChannel(noise_var=noiseVar, taps=Lh, chVar=chVar)
    BER, PCR, PAPR, rotationEst, theta_hat = 0, 0, 0, 0, 0
    for _ in range(noIter):
        msgTx = np.random.randint(0, 2, bch.B)
        msgEnc = bch.msgEncoding(msgTx)

        sigTx = jfs.coeffCon(msgEnc)
        sigPower = np.mean(np.abs(sigTx)**2)
        sigTx /= np.sqrt(sigPower)

        rotation = np.random.uniform(0, 2*np.pi)
        if FLAG == "FS":
            sigRx, _ = ch.frequencySelective(sigTx, noiseFlag)
        elif FLAG == "CFO":
            sigRx = ch.CFO(sigTx, rotation, noiseFlag)
        elif FLAG == "AWGN":
            sigRx = ch.awgn(sigTx)
        elif FLAG == "FS_CFO":
            sigRx, _ = ch.frequencySelective(sigTx, noiseFlag, rotation)

        if FLAG == 'FS_CFO' or FLAG == 'CFO':
            sigRx, rotation_hat = jfs.CFOcorrection(sigRx, Q)
        estCodeword = jfs.fftDizet(sigRx, Q)

        msgRx = bch.msgDecoding(estCodeword)
        mae = np.abs(rotation - rotation_hat) if np.abs(rotation - rotation_hat) < np.pi else 2*np.pi-np.abs(rotation - rotation_hat)
        rotationEst += mae/rotation
        BER += perParam.ber(msgRx, msgTx)
        PCR += perParam.pcr(msgRx, msgTx)
        PAPR += jfs.PAPR(sigTx)
    berSNR[snr] = BER/noIter
    perSNR[snr] = 1 - PCR/noIter
    paprSNR[snr] = PAPR/noIter
    rotationSNR[snr] = rotationEst/noIter
    goodputSNR[snr] = (PCR * (K - m * t)/(K + Lh))/noIter
    print(f"SNR: {snr} done")

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
# for i, (k, v) in enumerate(ber.items()):
#     plt.semilogy(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
#                 color=colors[i], marker=markers[i], markersize=8, label=k, markerfacecolor=markerColor[i])
plt.semilogy(berSNR.keys(), berSNR.values(), linewidth=1.2)
plt.ylim(1e-4, 1)
plt.xlim(0, 40)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"{FLAG} Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
# plt.legend(loc='lower left', framealpha=0.5, fontsize=18)
# plt.tight_layout()
plt.savefig(f"results/JBMOCZ/IRO/BCH/ber_{Lh}_{Peak}.jpeg", bbox_inches="tight")

plt.figure(2, dpi=800)
# for i, (k, v) in enumerate(per.items()):
#     plt.semilogy(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
#                 color=colors[i], marker=markers[i], markersize=8, label=k, markerfacecolor=markerColor[i])
plt.semilogy(perSNR.keys(), perSNR.values(), linewidth=1.2)
plt.ylim(1e-4, 1)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.xlim(0, 40)
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"{FLAG} Analysis chVar-{chVar} R-{Rj}, Z-{zeta} {Peak}")
# plt.legend(loc='lower left', framealpha=0.5, fontsize=18)
# plt.tight_layout()
plt.savefig(f"results/JBMOCZ/IRO/BCH/per_{Lh}_{Peak}.jpeg", bbox_inches="tight")

plt.figure(3, dpi=800)
# for i, (k, v) in enumerate(papr.items()):
#     plt.plot(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
#                 color=colors[i], marker=markers[i], markersize=8, label=k, markerfacecolor=markerColor[i])
plt.plot(paprSNR.keys(), paprSNR.values(), linewidth=1.2)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR (dB)")
plt.xlim(0, 40)
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"PAPR Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
# plt.legend(loc='lower left', framealpha=0.5, fontsize=18)
# plt.tight_layout()
plt.savefig(f"results/JBMOCZ/IRO/BCH/papr_{Lh}_{Peak}.jpeg", bbox_inches="tight")

plt.figure(4, dpi=800)
# for i, (k, v) in enumerate(goodput.items()):
#     plt.plot(v.keys(), v.values(), linewidth=1.2, linestyle=linestyles[i],
#                 color=colors[i], marker=markers[i], markersize=8, label=k, markerfacecolor=markerColor[i])
plt.plot(goodputSNR.keys(), goodputSNR.values(), linewidth=1.2)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.xlabel("SNR (dB)")
plt.ylabel("Throughput (bits/sec)")
plt.ylim(-0.01, 1.01)
plt.xlim(0, 40)
plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"Goodput Analysis chVar-{chVar} R-{Rj}, Z-{zeta}, {Peak}")
# plt.legend(loc='upper left', framealpha=0.5, fontsize=18)
# plt.tight_layout()
plt.savefig(f"results/JBMOCZ/IRO/BCH/goodput_{Lh}_{Peak}.jpeg", bbox_inches="tight")