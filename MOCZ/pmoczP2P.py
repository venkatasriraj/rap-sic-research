"""
Testing of P-MOCZ for point-to-point communication link
- 4 subsectors within a sector
Q: over-sampling factor
nFFT: K*M*Q
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from wirelessComm import (
    PMOCZ, PerformanceParameters, SlowFadingChannel, ChannelEstimation
)

K = 16
M = 2**np.arange(6)
Q = 8
noIter = int(1e1)
SNR_dB = np.arange(-10, 21, 2)
signalPower = 1
perParam = PerformanceParameters()
chEst = ChannelEstimation()
paprM, perM, goodputM, berM, maeHM = {}, {}, {}, {}, {}
for m in M:
    pmoczSystem = PMOCZ(K, m)
    totalBits = int(K * (1 + math.log2(m)))
    paprSNR, perSNR, goodputSNR, berSNR, maeHSNR = {}, {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = SlowFadingChannel(noiseVar)
        PAPR, PCR, BER, maeH = 0, 0, 0, 0
        for i in range(noIter):
            msg_tx = np.random.randint(0, 2, totalBits)
            sigTx = pmoczSystem.coeffCon(msg_tx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            sigRx, h = ch.transmit(sigTx)

            msg_est = pmoczSystem.fftDiZet(sigRx, Q)
            # -- Signal reconstruction bu the obtained message
            sigRecon = pmoczSystem.coeffCon(msg_est)
            sigPower = np.mean(np.abs(sigRecon)**2)
            sigRecon /= np.sqrt(sigPower)

            h_est = chEst.leastSquares(sigRx, sigRecon)
            maeH += abs(h - h_est) / abs(h)
            PCR += perParam.pcr(msg_est, msg_tx)
            BER += perParam.ber(msg_est, msg_tx)
            PAPR += pmoczSystem.PAPR(sigTx)
        paprSNR[snr] = PAPR / noIter
        perSNR[snr] = 1 - (PCR / noIter)
        berSNR[snr] = BER / noIter
        maeHSNR[snr] = maeH / noIter
        goodputSNR[snr] = (PCR / noIter) * totalBits
    print(f"Sub-Sector Sampling: {m} done.")
    paprM[m] = paprSNR
    perM[m] = perSNR
    berM[m] = berSNR
    maeHM[m] = maeHSNR
    goodputM[m] = goodputSNR

plt.figure(1, dpi=800)
for k, v in paprM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR(dB)")
plt.ylabel("PAPR")
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Sub-Sector Sampling Analysis")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PMOCZ/conSim/papr{K}.jpeg")

plt.figure(2, dpi=800)
for k, v in perM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR(dB)")
plt.ylabel("PER")
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Sub-Sector Sampling Analysis")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PMOCZ/conSim/per{K}.jpeg")

plt.figure(3, dpi=800)
for k, v in berM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Sub-Sector Sampling Analysis")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PMOCZ/conSim/ber{K}.jpeg")

plt.figure(4, dpi=800)
for k, v in maeHM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR(dB)")
plt.ylabel("Normalised MAE of h")
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Sub-Sector Sampling Analysis")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PMOCZ/conSim/maeh{K}.jpeg")

plt.figure(5, dpi=800)
for k, v in goodputM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR(dB)")
plt.ylabel("GoodPut")
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f"Sub-Sector Sampling Analysis")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PMOCZ/conSim/goodput{K}.jpeg")