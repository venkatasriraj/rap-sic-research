"""
We investigate the system performance for various values of Q for a fixed K, M
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import (
    UidMOCZ, PerformanceParameters, MultiPathFading
)
K = 16
M = 4
Q = 2**np.arange(1, 8)
SNR_dB = np.arange(-5, 46, 5)
noIter = int(1e4)
signalPower = 1
perParam = PerformanceParameters()
berQ, perQ, uIdEstQ, rotationEstQ ={}, {}, {}, {}
for q in Q:
    uidMOCZ_System = UidMOCZ(K, M)
    blockLen = uidMOCZ_System.UIDbits + K
    berSNR, perSNR, uIdEstSNR, rotationEstSNR = {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = MultiPathFading(noiseVar)
        BER, PCR, uIdEst, rotationEst = 0, 0, 0, 0
        for i in range(noIter):
            rotation = np.random.uniform(0, 2*np.pi)
            msgTx = np.random.randint(0, 2, blockLen)

            sigTx = uidMOCZ_System.coeffCon(msgTx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            sigRx = ch.transmit(sigTx, rotation)

            msg_rx, userId_est, rotation_hat = uidMOCZ_System.uIdDecoder(sigRx, q)
            BER += perParam.ber(msg_rx, msgTx)
            # uIdEst += 0 if userId_est == userId else 1
            rotationEst += np.abs(rotation_hat - rotation) / rotation
            # if userId_est == userId and perParam.pcr(msg_rx, msgTx) == 1:
            #     PCR += 1
            PCR += perParam.pcr(msg_rx, msgTx)
        berSNR[snr] = BER / noIter
        perSNR[snr] = 1 - (PCR / noIter)
        # uIdEstSNR[snr] = uIdEst / noIter
        rotationEstSNR[snr] = rotationEst / noIter
    print(f"Over-sampling factor: {q} done")
    berQ[q] = berSNR
    perQ[q] = perSNR
    # uIdEstQ[q] = uIdEstSNR
    rotationEstQ[q] = rotationEstSNR

plt.figure(1, dpi=800)
for k, v in berQ.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f"Q={k}")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Over-Sampling Factor Analysis for K={K} and M={M}")
plt.ylim(1e-3, 1)
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/QAnalysis/berM{M}_{FLAG}.jpeg")

plt.figure(2, dpi=800)
for k, v in perQ.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f"Q={k}")
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Over-Sampling Factor Analysis for K={K} and M={M}")
plt.ylim(1e-3, 1)
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/QAnalysis/perM{M}_{FLAG}.jpeg")

# plt.figure(3, dpi=800)
# for k, v in uIdEstQ.items():
#     plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f"Q={k}")
# plt.xlabel("SNR (dB)")
# plt.ylabel("UserId Estimation Error")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.ylim(0, 1.05)
# plt.title(f"Over-Sampling Factor Analysis for K={K} and M={M}")
# plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig(f"results/uidMOCZ/QAnalysis/uidEstM{M}_{FLAG}.jpeg")

plt.figure(4, dpi=800)
for k, v in rotationEstQ.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f"Q={k}")
plt.xlabel("SNR (dB)")
plt.ylabel("Normalised MAE of Rotation ")
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Over-Sampling Factor Analysis for K={K} and M={M}")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/QAnalysis/rotationEstM{M}_{FLAG}.jpeg")