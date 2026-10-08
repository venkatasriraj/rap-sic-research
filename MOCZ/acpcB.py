"""
Implementation of BMOCZ-ACPC for various channel models
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import BMOCZ, PerformanceParameters, SlowFadingChannel, ACPC

m, t, Q, noIter = 7, 2, 128, int(1e4)
chVar = 0.95
K = 2**m-1
Lh = [1, 10, 23, 31, 63, 127]
FLAG = 'FS'
SNR_dB = np.arange(-5, 41, 3)
bmoczSystem = BMOCZ(K, PZrad=0)
perParam = PerformanceParameters()
acpc = ACPC(m, t)
berLh, perLh, paprLh, rotationEstLh = {}, {}, {}, {}
for l in Lh:
    berSNR, perSNR, paprSNR, rotationEstSNR = {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = 10**(-snr/10)
        ch = SlowFadingChannel(noise_var=noiseVar, taps=l, chVar=chVar)
        results = bmoczSystem.simulatorACPC(noIter, perParam, ch, acpc, FLAG=FLAG)

        berSNR[snr] = results['ber']
        perSNR[snr] = 1 - results['pcr']
        paprSNR[snr] = results['papr']
        rotationEstSNR[snr] = results['rotationEst']
    print(f"{l}-tap channel done.")
    berLh[l] = berSNR
    perLh[l] = perSNR
    rotationEstLh[l] = rotationEstSNR
    paprLh[l] = paprSNR

plt.figure(1, dpi=800)
for k, v in berLh.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'Lh-{k}')
plt.ylim(1e-5, 1)
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"BMOCZ-ACPC({K}, {K-(t+1)*m}) {FLAG} BER Analysis chVar-{chVar}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/BMOCZ/ACPC/{FLAG}ber{chVar}_{K}.jpeg")

plt.figure(2, dpi=800)
for k, v in perLh.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.ylim(1e-5, 1)
plt.xlabel("SNR(dB)")
plt.ylabel("PER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"BMOCZ-ACPC({K}, {K-(t+1)*m}) {FLAG} PER Analysis chVar-{chVar}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/BMOCZ/ACPC/{FLAG}per{chVar}_{K}.jpeg")

# plt.figure(3, dpi=800)
# for k, v in paprLh.items():
#     plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
# plt.xlabel("SNR(dB)")
# plt.ylabel("PAPR (dB)")
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"BMOCZ-ACPC({K}, {K-(t+1)*m}) {FLAG} PAPR Analysis chVar-{chVar}")
# plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig(f"results/BMOCZ/ACPC/{FLAG}papr{chVar}_{K}.jpeg")

# plt.figure(4, dpi=800)
# for k, v in rotationEstLh.items():
#     plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
# # plt.ylim(1e-5, 1)
# plt.xlabel("SNR(dB)")
# plt.ylabel("MAE of CFO")
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"BMOCZ-ACPC({K}, {K-(t+1)*m}) {FLAG} CFO Analysis chVar-{chVar}")
# plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig(f"results/BMOCZ/ACPC/{FLAG}cfo{chVar}_{K}.jpeg")