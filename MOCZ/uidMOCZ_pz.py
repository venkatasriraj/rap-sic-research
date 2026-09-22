"""
UserId based pilot-zero positioning and transmission of generated signal
UserId, Pq = K*M :
    - uId1 = UserId % Pq 
    - subSector = UserId / Pq (default zero location like MOCZ will not be selected)
        one of the sub-sectors will be selected to transmit the message
    - j = uId % M
    - i = uId // M
pilotZero-1 will be placed at -1.25*Router
pilotZero-2 will be placed at 1.75*Router*e**(1j* (2*pi/(K*M)) * (i*M + j))
    UserId = subSector * Pq + i*M + j
subSector is estimated using "ffo_est" method
i, j are estimated using FFT based pilo-zero detection
We will be applying FFT 3 times:
    1) rotation estimation using pilotZero-1
    2) s

- we will only consider possibleUid - 1, since the errors are happening for userId = possibleUid
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import (
    UidMOCZ, MultiPathFading, PerformanceParameters
)
K = 16
M = 2**np.arange(6)
Q = 32
SNR_dB = np.arange(-5, 51, 5)
noIter = int(1e4)
signalPower = 1
perParam = PerformanceParameters()
berM, perM, uIdEstM, paprM, payloadEstM, rotationEstM = {}, {}, {}, {}, {}, {}
for m in M:
    # possibleUid = int(K * m**2) if FLAG == 1 else m
    uidMOCZ_System = UidMOCZ(K, m)
    blockLen = K + uidMOCZ_System.UIDbits    
    berSNR, perSNR, uIdEstSNR, paprSNR, payloadEstSNR, rotationEstSNR = {}, {}, {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = MultiPathFading(noise_var=noiseVar)   
        BER, PCR, uIdEst, PAPR, rotationEst, payloadEst = 0, 0, 0, 0, 0, 0
        for _ in range(noIter):
            # userId = np.random.randint(0, possibleUid)
            # print(f"UserId: {userId}, {uidMOCZ_System.locations}, {possibleUid}")
            rotation = np.random.uniform(0, 2*np.pi)
            msgTx = np.random.randint(0, 2, blockLen)
            userId = uidMOCZ_System.bin2dec(msgTx[:uidMOCZ_System.UIDbits])

            sigTx = uidMOCZ_System.coeffCon(msgTx)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            sigRx = ch.transmit(sigTx, rotation)
            msg_rx, userId_est, rotation_hat = uidMOCZ_System.uIdDecoder(sigRx, Q)
            BER += perParam.ber(msg_rx, msgTx)
            uIdEst += 0 if userId == userId_est else 1
            PCR += perParam.pcr(msg_rx, msgTx)
            PAPR += uidMOCZ_System.PAPR(sigTx)
            rotationEst += abs(rotation_hat - rotation) / rotation
            payloadEst += perParam.ber(msg_rx[uidMOCZ_System.UIDbits:], msgTx[uidMOCZ_System.UIDbits:])
        berSNR[snr] = BER / noIter
        perSNR[snr] = 1 - (PCR / noIter)
        uIdEstSNR[snr] = uIdEst / noIter
        paprSNR[snr] = PAPR / noIter
        rotationEstSNR[snr] = rotationEst/noIter
        payloadEstSNR[snr] = payloadEst/noIter
    print(f"Sub-Sector: {m} done.")
    berM[m] = berSNR
    perM[m] = perSNR
    uIdEstM[m] = uIdEstSNR
    paprM[m] = paprSNR
    rotationEstM[m] = rotationEstSNR
    payloadEstM[m] = payloadEstSNR

plt.figure(1, dpi=800)
for k, v in berM.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.ylim(1e-5, 1)
plt.title(f"UserId Encoded Pilot MOCZ BER Analysis BL-{K}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/subSectorAnalysis/berK{K}.jpeg")

plt.figure(2, dpi=800)
for k, v in perM.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.ylim(1e-5, 1)
plt.title(f"UserId Encoded Pilot MOCZ PER Analysis BL-{K}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/subSectorAnalysis/perK{K}.jpeg")

plt.figure(3, dpi=800)
for k, v in uIdEstM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR (dB)")
plt.ylabel("UserId Error Estimate")
plt.title(f"UserId Encoded Pilot MOCZ UserId Est. Analysis BL-{K}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/subSectorAnalysis/uidEstK{K}.jpeg")

plt.figure(4, dpi=800)
for k, v in paprM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR")
plt.title(f"UserId Encoded Pilot MOCZ PAPR Analysis BL-{K}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/subSectorAnalysis/paprK{K}.jpeg")

plt.figure(5, dpi=800)
for k, v in rotationEstM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR (dB)")
plt.ylabel("Normalised MAE of rotation")
plt.title(f"UserId Encoded Pilot MOCZ Rotation Est Analysis BL-{K}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/subSectorAnalysis/rotationEstK{K}.jpeg")

plt.figure(6, dpi=800)
for k, v in payloadEstM.items():
    plt.semilogy(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR (dB)")
plt.ylabel("Payload BER")
plt.ylim(1e-5, 1)
plt.title(f"UserId Encoded Pilot MOCZ Payload BER Analysis BL-{K}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/subSectorAnalysis/payload_berK{K}.jpeg")