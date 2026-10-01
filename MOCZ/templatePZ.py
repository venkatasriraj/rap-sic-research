"""
Simulation analysis of JBMOCZ point-to-point communication link.

"""
import numpy as np
from wirelessComm import BMOCZ, SlowFadingChannel, PerformanceParameters
import matplotlib.pyplot as plt

K = np.arange(8, 33, 4)
Q = 4
noIter = int(1e4)
SNR_dB = np.arange(-10, 41, 5)
signalPower = 1
perParam = PerformanceParameters()
PZrad = 1.637
ber_K, per_K, papr_K, rotationEst_K = {}, {}, {}, {}
for k in K:
    bmcozSystem = BMOCZ(k, PZrad=PZrad)
    singlePZ = [-PZrad * bmcozSystem.R]
    ber_snr, per_snr, papr_snr, rotationEst_snr = {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = SlowFadingChannel(noise_var=noiseVar)
        BER, PCR, PAPR, rotationEst = 0, 0, 0, 0
        for i in range(noIter):
            msgTx = np.random.randint(0, 2, k)
            sigTx = bmcozSystem.coeffCon(msgTx, singlePZ)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            rotation = np.random.uniform(0, 2*np.pi)
            sigRx = ch.CFO(sigTx, rotation)

            sigRxCorrected, rotation_hat = bmcozSystem.rotationEstTemplate(sigRx)
            msgRx = bmcozSystem.fftDizet(sigRxCorrected)
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += bmcozSystem.PAPR(sigTx)
            mae = np.abs(rotation - rotation_hat) if np.abs(rotation - rotation_hat) < np.pi else 2*np.pi-np.abs(rotation - rotation_hat)
            rotationEst += mae/rotation
        ber_snr[snr] = BER / noIter
        per_snr[snr] = 1 - (PCR/noIter)
        papr_snr[snr] = PAPR / noIter
        rotationEst_snr[snr] = rotationEst / noIter
    print(f"Block-Length {k} done")
    ber_K[k] = ber_snr
    per_K[k] = per_snr
    papr_K[k] = papr_snr
    rotationEst_K[k] = rotationEst_snr

plt.figure(1, dpi=800)
for k, v in ber_K.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title(f"BMOCZ BER Analysis PZ- {-PZrad}")
plt.ylim(1e-5, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/template/ber{PZrad}.jpeg")

plt.figure(2, dpi=800)
for k, v in per_K.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.ylim(1e-5, 1)
plt.title(f"BMOCZ PER Analysis PZ- {-PZrad}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/template/per{PZrad}.jpeg")

plt.figure(3, dpi=800)
for k, v in papr_K.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR (dB)")
plt.title(f"BMOCZ PAPR Analysis PZ- {-PZrad}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/template/papr{PZrad}.jpeg")

plt.figure(4, dpi=800)
for k, v in rotationEst_K.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.xlabel("SNR (dB)")
plt.ylabel("Normalised RotationEst")
plt.title(f"BMOCZ Normalised RotationEst Analysis PZ- {-PZrad}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/template/rotationEst{PZrad}.jpeg")