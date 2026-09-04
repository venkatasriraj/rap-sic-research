"""
UserId based pilot-zero positioning and transmission of generated signal
UserId, Pq = K*M / 2:
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
M = 2**np.arange(1, 8)
Q = 16
SNR_dB = np.arange(-10, 31, 3)
noIter = int(1e2)
signalPower = 1
perParam = PerformanceParameters()
berM, perM, uIdEstM, paprM = {}, {}, {}, {}
for m in M:
    possibleUid = int(K * m**2 / 2)
    uidMOCZ_System = UidMOCZ(K, m)
    berSNR, perSNR, uIdEstSNR, paprSNR = {}, {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = MultiPathFading(noise_var=noiseVar)   
        BER, PCR, uIdEst, PAPR = 0, 0, 0, 0
        for i in range(noIter):
            userId = np.random.randint(1, possibleUid)
            rotation = np.random.uniform(0, 2*np.pi)
            msgTx = np.random.randint(0, 2, K)

            sigTx = uidMOCZ_System.coeffCon(msgTx, userId)
            sigRx = ch.transmit(sigTx, rotation)

            msg_rx, userId_est = uidMOCZ_System.uIdDecoder(sigRx, Q)
            BER += perParam.ber(msg_rx, msgTx)
            uIdEst += 0 if userId == userId_est else 1
            if perParam.pcr(msg_rx, msgTx) == 1 and userId_est == userId:
                PCR += 1
            PAPR += uidMOCZ_System.PAPR(sigTx)
        berSNR[snr] = BER / noIter
        perSNR[snr] = 1 - (PCR / noIter)
        uIdEstSNR[snr] = uIdEst / noIter
        paprSNR[snr] = PAPR / noIter
    print(f"Sub-Sector: {m} done.")
    berM[m] = berSNR
    perM[m] = perSNR
    uIdEstM[m] = uIdEstSNR
    paprM[m] = paprSNR

plt.figure(1, dpi=800)
for k, v in berM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("UserId Encoded Pilot MOCZ")
plt.grid(True, linestyle='--', alpha=0.9)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/subSectorAnalysis/berK{K}.jpeg")

plt.figure(2, dpi=800)
for k, v in perM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.title("UserId Encoded Pilot MOCZ")
plt.grid(True, linestyle='--', alpha=0.9)
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/subSectorAnalysis/perK{K}.jpeg")

plt.figure(3, dpi=800)
for k, v in uIdEstM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR (dB)")
plt.ylabel("UserId Error Estimate")
plt.title("UserId Encoded Pilot MOCZ")
plt.grid(True, linestyle='--', alpha=0.9)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/subSectorAnalysis/uidEstK{K}.jpeg")

plt.figure(4, dpi=800)
for k, v in paprM.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f'M={k}')
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR")
plt.title("UserId Encoded Pilot MOCZ")
plt.grid(True, linestyle='--', alpha=0.9)
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/uidMOCZ/subSectorAnalysis/paprK{K}.jpeg")