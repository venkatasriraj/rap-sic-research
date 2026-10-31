"""
JBMOCZ Analysis for frequency selective channel (2-tap)
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import JBMOCZ, SlowFadingChannel, PerformanceParameters

K = np.arange(8, 33, 4)
FLAG = "FS_CFO"
SNR_dB = np.arange(-5, 32, 3)
Lh, chVar = 2, 0.86
zeta, Rj, noIter = 1.15, 1.044, int(1e4)
perParam = PerformanceParameters()
berK, perK, paprK, rotationEstK = {}, {}, {}, {}
for k in K:
    jbmcozSystem = JBMOCZ(k, zeta, Rj=Rj)
    mPeak = [0]
    berSNR, perSNR, paprSNR, rotationEstSNR = {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = 10**(-snr/10)
        ch = SlowFadingChannel(noise_var=noiseVar, taps=Lh, chVar=chVar)
        results = jbmcozSystem.simulator(noIter, perParam, ch, FLAG=FLAG)

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
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("BER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"JBMOCZ {FLAG} BER Analysis Lh-{Lh} chVar-{chVar} R-{Rj}, Z-{zeta}, [0]")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
if len(mPeak) == 1:
    plt.savefig(f"results/JBMOCZ/FS/Lh{Lh}/{FLAG}berZ{zeta}_{chVar}.jpeg")
else:
    plt.savefig(f"results/JBMOCZ/FS_M/Lh{Lh}/{FLAG}berZ{zeta}_{chVar}.jpeg")

plt.figure(2, dpi=800)
for k, v in perK.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.ylim(1e-4, 1)
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("PER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"JBMOCZ {FLAG} PER Analysis Lh-{Lh} chVar-{chVar} R-{Rj}, Z-{zeta}, [0]")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
if len(mPeak) == 1:
    plt.savefig(f"results/JBMOCZ/FS/Lh{Lh}/{FLAG}perZ{zeta}_{chVar}.jpeg")
else:
    plt.savefig(f"results/JBMOCZ/FS_M/Lh{Lh}/{FLAG}perZ{zeta}_{chVar}.jpeg")
    
# plt.figure(3, dpi=800)
# for k, v in paprK.items():
#     plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
# plt.ylabel("PAPR (dB)")
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"JBMOCZ {FLAG} PAPR Analysis Lh-{Lh} chVar-{chVar} R-{Rj}, Z-{zeta}, [0]")
# plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
# plt.tight_layout()
# if len(mPeak) == 1:
#     plt.savefig(f"results/JBMOCZ/FS/Lh{Lh}/{FLAG}paprZ{zeta}_{chVar}.jpeg")
# else:
#     plt.savefig(f"results/JBMOCZ/FS_M/Lh{Lh}/{FLAG}paprZ{zeta}_{chVar}.jpeg")