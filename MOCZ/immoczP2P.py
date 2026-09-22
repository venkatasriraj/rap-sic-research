"""
IM-MOCZ with Pilot-Zero analysis (FLAG = 0)
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import IMMOCZ, PerformanceParameters, MultiPathFading
M = 2**np.arange(6)
K, Q = 16, 32
noIter = int(1e4)
signalPower = 1
SNR_dB = np.arange(-5, 51, 5)
perParam = PerformanceParameters()
berM, perM, paprM, rotationEstM, sectorEstM, msgEstM = {}, {}, {}, {}, {}, {}
for m in M:
    immoczSys = IMMOCZ(K, m)
    blockLen = K + immoczSys.addBitsLen    
    ber_snr, per_snr, papr_snr, rotationEst_snr, sectorEst_snr, msgEst_snr = {}, {}, {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = MultiPathFading(noiseVar)
        BER, PCR, PAPR, rotationEst, sectorEst, msgEst = 0, 0, 0, 0, 0, 0
        for _ in range(noIter):
            rotation = np.random.uniform(0, 2*np.pi)
            msgTx = np.random.randint(0, 2, blockLen)
            sector = immoczSys.bin2dec(msgTx[:immoczSys.addBitsLen])

            sigTx = immoczSys.coeffCon(msgTx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            sigRx = ch.transmit(sigTx, rotation)
            msgRx, sector_hat, rotation_hat = immoczSys.imDecoder(sigRx, Q)
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            PAPR += immoczSys.PAPR(sigTx)
            rotationEst += abs(rotation - rotation_hat) / rotation
            sectorEst += 1 if sector_hat != sector else 0
            msgEst += perParam.ber(msgRx[immoczSys.addBitsLen:], msgTx[immoczSys.addBitsLen:])
        ber_snr[snr] = BER / noIter
        per_snr[snr] = 1 - PCR/noIter
        papr_snr[snr] = PAPR / noIter
        rotationEst_snr[snr] = rotationEst / noIter
        sectorEst_snr[snr] = sectorEst / noIter
        msgEst_snr[snr] = msgEst / noIter
    print(f"Sub-sectors: {m} done")
    berM[m] = ber_snr
    perM[m] = per_snr
    paprM[m] = papr_snr
    rotationEstM[m] = rotationEst_snr
    sectorEstM[m] = sectorEst_snr
    msgEstM[m] = msgEst_snr

plt.figure(1, dpi=800)
for k, v in berM.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.ylim(1e-5, 1)
plt.title("IM-MOCZ BER Analysis")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/IMMOCZ/ber{K}.jpeg")

plt.figure(2, dpi=800)
for k, v in perM.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.ylim(1e-5, 1)
plt.title("IM-MOCZ PER Analysis")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/IMMOCZ/per{K}.jpeg")

plt.figure(3, dpi=800)
for k, v in paprM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR")
plt.title("IM-MOCZ PAPR Analysis")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/IMMOCZ/papr{K}.jpeg")

plt.figure(4, dpi=800)
for k, v in rotationEstM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("Normalised Rotation MAE")
plt.title("IM-MOCZ Rotation Analysis")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/IMMOCZ/rotationEst{K}.jpeg")

plt.figure(5, dpi=800)
for k, v in sectorEstM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("Sector Error Estimate")
plt.title("IM-MOCZ Sector Est. Analysis")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/IMMOCZ/sectorEst{K}.jpeg")

plt.figure(6, dpi=800)
for k, v in msgEstM.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'M = {k}')
plt.xlabel("SNR (dB)")
plt.ylabel("Payload BER")
plt.ylim(1e-5, 1)
plt.title("IM-MOCZ Payload BER Analysis")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/IMMOCZ/payload_ber{K}.jpeg")