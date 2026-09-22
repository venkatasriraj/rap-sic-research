"""
Simulation analysis of JBMOCZ point-to-point communication link.

"""
import numpy as np
from wirelessComm import JBMOCZ, MultiPathFading, PerformanceParameters, BMOCZ
import matplotlib.pyplot as plt

zeta = 1.07
K = np.arange(8, 21)
noIter = int(1e2)
SNR_dB = np.arange(-10, 21, 3)
signalPower = 1
perParam = PerformanceParameters()
ber_K, per_K, papr_K, rotationEst_K = {}, {}, {}, {}
for k in K:
    jbmcozSystem = JBMOCZ(k, zeta)
    ber_snr, per_snr, papr_snr, rotationEst_snr = {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = MultiPathFading(noise_var=noiseVar)
        BER, PCR, PAPR, rotationEst = 0, 0, 0, 0
        for i in range(noIter):
            msgTx = np.random.randint(0, 2, k)
            sigTx = jbmcozSystem.coeffCon(msgTx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            rotation = np.random.uniform(0, 2*np.pi)
            sigRx = ch.transmit(sigTx, rotation)

            sigRxCorrected, rotation_hat = jbmcozSystem.rotationEst(sigRx)
            msgRx = jbmcozSystem.fftDizet(sigRxCorrected)
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += jbmcozSystem.PAPR(sigTx)
            rotationEst += np.abs(rotation_hat - rotation)/rotation
        ber_snr[snr] = BER / noIter
        per_snr[snr] = 1 - (PCR/noIter)
        papr_snr[snr] = PAPR / noIter
        rotationEst_snr[snr] = rotationEst / noIter
        # result = jbmcozSystem.simulator(noIter, perParam, ch)
        # ber_snr[snr], per_snr[snr], papr_snr[snr], rotationEst_snr[snr] = result['ber'], 1 - result['pcr'], result['papr'], result['rotationEst']
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
plt.title(f"JBMOCZ BER Analysis zeta-{zeta}")
plt.ylim(1e-5, 1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/JBMOCZ/berz{zeta}.jpeg")

plt.figure(2, dpi=800)
for k, v in per_K.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.ylim(1e-5, 1)
plt.title(f"JBMOCZ PER Analysis zeta-{zeta}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/JBMOCZ/perz{zeta}.jpeg")

plt.figure(3, dpi=800)
for k, v in papr_K.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR")
plt.title(f"JBMOCZ PAPR Analysis zeta-{zeta}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/JBMOCZ/paprz{zeta}.jpeg")

plt.figure(4, dpi=800)
for k, v in rotationEst_K.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.xlabel("SNR (dB)")
plt.ylabel("Normalised RotationEst")
plt.title(f"JBMOCZ Normalised RotationEst Analysis zeta-{zeta}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/JBMOCZ/rotationEstz{zeta}.jpeg")