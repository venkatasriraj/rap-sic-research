"""
Simulation analysis of SBMOCZ point-to-point communication link.

"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import SBMOCZ, MultiPathFading, PerformanceParameters

zeta = 0.0545
K = np.arange(8, 21)
SNR_dB = np.arange(-10, 21, 3)
perParam = PerformanceParameters()
noIter = int(1e4)
ber_K, per_K, papr_K, rotationEst_K = {}, {}, {}, {}
for k in K:
    sbmoczSystem = SBMOCZ(k, zeta)
    ber_snr, per_snr, papr_snr, rotationEst_snr = {}, {}, {}, {}
    for snr in SNR_dB:
        ch = MultiPathFading(noise_var=0.01)
        BER, PCR, PAPR, rotationEst = 0, 0, 0, 0
        for i in range(noIter):
            msgTx = np.random.randint(0, 2, k)
            sigTx = sbmoczSystem.coeffCon(msgTx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            rotation = np.random.uniform(0, 2*np.pi)
            sigRx = ch.transmit(sigTx, rotation)

            sigRx_corrected, rotation_hat = sbmoczSystem.rotationEst(sigRx, Ns=256)
            msgRx = sbmoczSystem.msgDecoder(sigRx_corrected)
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += sbmoczSystem.PAPR(sigTx)
            rotationEst += np.abs(rotation - rotation_hat)/rotation
        ber_snr[snr] = BER/noIter
        per_snr[snr] = 1 - (PCR/noIter)
        papr_snr[snr] = PAPR/noIter
        rotationEst_snr[snr] = rotationEst/noIter
    print(f"Block-Length(K): {k} Done")
    ber_K[k] = ber_snr
    per_K[k] = per_snr
    papr_K[k] = papr_snr
    rotationEst_K[k] = rotationEst_snr

plt.figure(1, dpi=800)
for k, v in ber_K.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(1e-6, 1)
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("SBMOCZ BER Analysis")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/BMOCZ/smooshed/berz{zeta}.jpeg")

plt.figure(2, dpi=800)
for k, v in per_K.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.ylim(1e-6, 1)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/BMOCZ/smooshed/perz{zeta}.jpeg")

plt.figure(3, dpi=800)
for k, v in papr_K.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR")
plt.title("SBMOCZ PAPR Analysis")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/BMOCZ/smooshed/paprz{zeta}.jpeg")

plt.figure(4, dpi=800)
for k, v in rotationEst_K.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'BL-{k}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR (dB)")
plt.ylabel("Normalised RotationEst")
plt.title("SBMOCZ Normalised RotationEst Analysis")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/BMOCZ/smooshed/rotationEstz{zeta}.jpeg")