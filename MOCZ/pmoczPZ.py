"""
Insertion of Pilot-Zero for PMOCZ to estimate the roation caused by CFO
and analyse the system performance
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from wirelessComm import (
    PMOCZ, PerformanceParameters, MultiPathFading
)
K = 16
M = 2**np.arange(6)
Q = 64
noIter = int(1e1)
SNR_dB = np.arange(-10, 31, 3)
signalPower = 1
perParam = PerformanceParameters()
paprM, perM, goodputM, berM, maeRotationM = {}, {}, {}, {}, {}
for m in M:
    pmoczSystem = PMOCZ(K, m)
    singlePZ = [-1.25 * pmoczSystem.R]
    totalBits = int(K * (1 + math.log2(m)))
    paprSNR, perSNR, goodputSNR, berSNR, maeRotationSNR = {}, {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = MultiPathFading(noiseVar)
        PAPR, PCR, BER, maeRotation = 0, 0, 0, 0
        for i in range(noIter):
            rotation = np.random.uniform(0, 2*np.pi)
            msgTx = np.random.randint(0, 2, totalBits)

            sigTx = pmoczSystem.coeffCon(msgTx, singlePZ)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            sigRx = ch.transmit(sigTx, rotation)

            msgRx, rotation_hat = pmoczSystem.singlePZDecodedMsg(sigRx, Q, singlePZ) 
            maeRotation += (rotation - rotation_hat) / rotation
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += pmoczSystem.PAPR(sigTx)
        paprSNR[snr] = PAPR / noIter
        perSNR[snr] = 1 - (PCR / noIter)
        goodputSNR[snr] = (PCR / noIter) * totalBits
        berSNR[snr] = BER / noIter
        maeRotationSNR[snr] = maeRotation / noIter
    print(f"Sub-Sector Sampling: {m} done")
    paprM[m] = paprSNR
    perM[m] = perSNR
    goodputM[m] = goodputSNR
    berM[m] = berSNR
    maeRotationM[m] = maeRotationSNR

plt.figure(1, dpi=800)
for k, v in paprM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("Peak to Average Power Ratio (PAPR)")
plt.title(f"PMOCZ Pilot-Zero {np.round(singlePZ, 4)} Analysis")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PMOCZ/PilotZero/papr{k}.jpeg")

plt.figure(2, dpi=800)
for k, v in perM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.title(f"PMOCZ Pilot-Zero {np.round(singlePZ, 4)} Analysis")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PMOCZ/PilotZero/per{k}.jpeg")

plt.figure(3, dpi=800)
for k, v in berM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title(f"PMOCZ Pilot-Zero {np.round(singlePZ, 4)} Analysis")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PMOCZ/PilotZero/berr{k}.jpeg")

plt.figure(4, dpi=800)
for k, v in maeRotationM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("Normalised MAE of Rotation")
plt.title(f"PMOCZ Pilot-Zero {np.round(singlePZ, 4)} Analysis")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PMOCZ/PilotZero/maeRotation{k}.jpeg")

plt.figure(5, dpi=800)
for k, v in goodputM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("Goodput")
plt.title(f"PMOCZ Pilot-Zero {np.round(singlePZ, 4)} Analysis")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PMOCZ/PilotZero/goodput{k}.jpeg")