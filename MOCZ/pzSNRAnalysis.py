"""
BMOCZ-PilotZero Analysis over SNR for a block-len will be done
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import BMOCZ, MultiPathFading, PerformanceParameters
K = np.array([8, 15, 16, 31])
Q = 32
SNR_dB = np.arange(-10, 46, 5)
signalPower = 1
noIter = int(1e4)
perParam = PerformanceParameters()
berK, perK, rotationEstK = {}, {}, {}
for k in K:
    bmoczSys = BMOCZ(k)
    berSNR, perSNR, rotationEstSNR = {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = MultiPathFading(noiseVar)

        results = bmoczSys.simulator(noIter, perParam, ch, PZrad=1.25)

        berSNR[snr] = results['ber']
        perSNR[snr] = 1 -results['pcr']
        rotationEstSNR[snr] = results['rotationEst']
    print(f'Block-Len {k} done')
    berK[k] = berSNR
    perK[k] = perSNR
    rotationEstK[k] = rotationEstSNR

plt.figure(1, dpi=800)
for k, v in berK.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.ylim(1e-4, 1)
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title("BMOCZ-PZ BER Analysis")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/SNRAnalysis/ber.jpeg")

plt.figure(2, dpi=800)
for k, v in perK.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.ylim(1e-4, 1)
plt.xlabel("SNR(dB)")
plt.ylabel("PER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title("BMOCZ-PZ PER Analysis")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/SNRAnalysis/per.jpeg")

plt.figure(3, dpi=800)
for k, v in rotationEstK.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.xlabel("SNR(dB)")
plt.ylabel("Normalised MAE of roation")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title("BMOCZ-PZ BER Analysis")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/SNRAnalysis/rotationEst.jpeg")