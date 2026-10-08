"""
BMOCZ system analysis for multitap channel (frequency selective)
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import SlowFadingChannel, BMOCZ, PerformanceParameters

K = np.arange(8, 33, 4)
Q, Lh, chVar = 4, 2, 0.86
FLAG = "BF"
noIter = int(1e4)
SNR_dB = np.arange(-5, 32, 3)
signalPower = 1
perParam = PerformanceParameters()
berK, perK, paprK, rotationEstK = {}, {}, {}, {}
for k in K:
    bmcozSystem = BMOCZ(k, PZrad=0)
    berSNR, perSNR, paprSNR, rotationEstSNR = {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = SlowFadingChannel(noise_var=noiseVar, taps=Lh, chVar=chVar)
        results = bmcozSystem.simulator(noIter, perParam, ch, FLAG=FLAG)

        berSNR[snr] = results['ber']
        perSNR[snr] = 1 - results['pcr']
        paprSNR[snr] = results['papr']
        rotationEstSNR[snr] = results['rotationEst']
    print(f'Block-Len {k} done')
    berK[k] = berSNR
    perK[k] = perSNR
    rotationEstK[k] = rotationEstSNR
    paprK[k] = paprSNR

plt.figure(1, dpi=800)
for k, v in berK.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.ylim(1e-4, 1)
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"BMOCZ {FLAG} BER Analysis Lh-{Lh} chVar-{chVar}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/BMOCZ/FS/{FLAG}berLh{Lh}_{chVar}.jpeg")

plt.figure(2, dpi=800)
for k, v in perK.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.ylim(1e-4, 1)
plt.xlabel("SNR(dB)")
plt.ylabel("PER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"BMOCZ {FLAG} PER Analysis Lh-{Lh} chVar-{chVar}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/BMOCZ/FS/{FLAG}perLh{Lh}_{chVar}.jpeg")

# plt.figure(3, dpi=800)
# for k, v in paprK.items():
#     plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
# plt.xlabel("SNR (dB)")
# plt.ylabel("PAPR (dB)")
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"BMOCZ {FLAG} PAPR Analysis Lh-{Lh} chVar-{chVar}")
# plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig(f"results/BMOCZ/FS/{FLAG}paprLh{Lh}_{chVar}.jpeg")

# plt.figure(4, dpi=800)
# for k, v in rotationEstK.items():
#     plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
# plt.xlabel("SNR(dB)")
# plt.ylabel("Normalised MAE of roation")
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"BMOCZ {FLAG} Rotation Est Analysis Lh-{Lh} chVar-{chVar}")
# plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig(f"results/BMOCZ/FS/{FLAG}rotationEstLh{Lh}.jpeg")