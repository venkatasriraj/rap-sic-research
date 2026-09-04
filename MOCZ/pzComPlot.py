"""
Comparision of designed Pilot-zero MOCZ with Tradtional MOCZ with ACPC to counter the CFO
Using ACPC(31, 16) will be correcting 2 bit errors wrt to the received message
while the pilot-zero code doesn't have any inherent error-correction capability
Pilot-Zero will be sending K bits so both ACPC and PZ have different symbol duration
FRO: Fractional Rotation Offset
Goodput (bits/sec) = (1-PER) * bits_per_pkt * pkts_per_sec
For same bandwidth for both schemes, the paket rate is different, that is 
    R_acpc = 33 pkts/sec; R_pz = 32 pkts/sec
    bits_per_sec_ACPC = R_acpc * 16 = 528 bps (this scheme corrects upto 2 errors)
    bits_per_sec_PZ   = R_px * 31 = 992 bps
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from wirelessComm import(
    BMOCZ, ACPC, MultiPathFading, PerformanceParameters
)
m = 5
t = 2
K = 2**m - 1 
Q = 256
BW = 1
subSym_ACPC = (K + 1) 
subSym_PZ = (K + 2)
baseline = math.lcm(subSym_ACPC, subSym_PZ)
pktRate_ACPC = baseline * BW / subSym_ACPC
pktRate_PZ = baseline * BW / subSym_PZ
signal_power = 1
noIter = int(1e2)
SNR_dB = np.arange(-10, 31, 3)

bmoczSystem = BMOCZ(K)
perParam = PerformanceParameters()
acpc = ACPC(m, t)
singlePZ = [-1.25 * bmoczSystem.R]
berACPC = {}; paprACPC = {}; perACPC = {}; lACPC = {}; goodputACPC = {}
berPZ = {}; paprPZ = {}; perPZ = {}; rotationPZ = {}; goodputPZ = {}
for snr in SNR_dB:
    noiseVar = signal_power * 10**(-snr/10)
    ch = MultiPathFading(noiseVar)
    BERacpc, PAPRacpc, PCRacpc, lacpc = 0, 0, 0, 0
    BERpz, PAPRpz, PCRpz, rotationpz = 0, 0, 0, 0
    for i in range(noIter):
        rotation = np.random.uniform(0, 2*np.pi)
        l = np.floor(rotation / (2 * np.pi / K))

        acpcMsg = np.random.randint(0, 2, acpc.B)
        acpcMsgEn = acpc.msg_encoding(acpcMsg)
        acpcSigTx = bmoczSystem.coeffCon(acpcMsgEn)
        acpcSigTx /= np.sqrt( np.mean(np.abs(acpcSigTx)**2) )
        acpcSigRx = ch.transmit(acpcSigTx, rotation)
        acpcSigFRO = bmoczSystem.ffoEstCor(acpcSigRx, Q)
        estCodeword = bmoczSystem.fftDizet(acpcSigFRO, Q)
        acpcEstMsg, acpcEst_l = acpc.codeword_decoding(estCodeword)
        BERacpc += perParam.ber(acpcEstMsg, acpcMsg)
        PCRacpc += perParam.pcr(acpcEstMsg, acpcMsg)
        PAPRacpc += bmoczSystem.PAPR(acpcSigTx)
        lacpc += abs(acpcEst_l - l)/(l+1)

        pzMsg = np.random.randint(0, 2, K)
        pzSigTx = bmoczSystem.coeffCon(pzMsg, singlePZ)
        pzSigTx /= np.sqrt( np.mean(np.abs(pzSigTx)**2) )
        pzSigRx = ch.transmit(pzSigTx, rotation)
        msg_hat, rotate_hat = bmoczSystem.singlePZDecodedMsg(pzSigRx, Q, singlePZ)
        BERpz += perParam.ber(msg_hat, pzMsg)
        PCRpz += perParam.pcr(msg_hat, pzMsg)
        PAPRpz += bmoczSystem.PAPR(pzSigTx)
        deviationR = np.abs(rotation - rotate_hat) 
        maeRotate = deviationR if deviationR <= np.pi else 2*np.pi - deviationR
        rotationpz += maeRotate / rotation
    berACPC[snr] = BERacpc / noIter
    paprACPC[snr] = PAPRacpc / noIter
    perACPC[snr] = 1 - (PCRacpc / noIter)
    goodputACPC[snr] = (PCRacpc / noIter) * (K - 5*(t+1)) * pktRate_ACPC
    lACPC[snr] = lacpc / noIter
    print(f"SNR: {snr} done.")
    berPZ[snr] = BERpz / noIter
    paprPZ[snr] = PAPRpz / noIter
    perPZ[snr] = 1 - (PCRpz / noIter)
    rotationPZ[snr] = rotationpz / noIter
    goodputPZ[snr] = (PCRpz / noIter) * K * pktRate_PZ

plt.figure(1, dpi=800)
plt.plot(berACPC.keys(), berACPC.values(), '-', color='tab:purple', linewidth=0.9, label=f"ACPC ({K}, {K-(t+1)*5})")    
plt.plot(berPZ.keys(), berPZ.values(), '--', color='tab:green', linewidth=0.9, label=f"Pilot-Zero {np.round(singlePZ, 4)}")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.ylim(0, 1.05)
plt.title(f"{noIter} runs per point")
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PilotZero/ACPC/ber_{t}.jpeg")

plt.figure(2, dpi=800)
plt.plot(perACPC.keys(), perACPC.values(), '-', color='tab:purple', linewidth=0.9, label=f"ACPC ({K}, {K-(t+1)*5})")    
plt.plot(perPZ.keys(), perPZ.values(), '--', color='tab:green', linewidth=0.9, label=f"Pilot-Zero {np.round(singlePZ, 4)}")
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.ylim(0, 1.05)
plt.title(f"{noIter} runs per point")
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PilotZero/ACPC/per_{t}.jpeg")

plt.figure(3, dpi=800)
plt.plot(paprACPC.keys(), paprACPC.values(), '-', color='tab:purple', linewidth=0.9, label=f"ACPC ({K}, {K-(t+1)*5})")    
plt.plot(paprPZ.keys(), paprPZ.values(), '--', color='tab:green', linewidth=0.9, label=f"Pilot-Zero {np.round(singlePZ, 4)}")
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR")
plt.title(f"{noIter} runs per point")
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PilotZero/ACPC/papr_{t}.jpeg")

plt.figure(4, dpi=800)
plt.plot(lACPC.keys(), lACPC.values(), '-', color='tab:purple', linewidth=0.9) 
plt.xlabel("SNR (dB)")
plt.ylabel("Normalised MAE of Integer Ration Offset")
plt.title(f"{noIter} runs per point")
plt.savefig(f"results/PilotZero/ACPC/lACPC_{t}.jpeg")

plt.figure(5, dpi=800)
plt.plot(rotationPZ.keys(), rotationPZ.values(), '--', color='tab:green', linewidth=0.9)
plt.xlabel("SNR (dB)")
plt.ylabel("MAE of estimated rotation")
plt.title(f"{noIter} runs per point")
plt.savefig(f"results/PilotZero/ACPC/rotationPZ_{t}.jpeg")

plt.figure(6, dpi=800)
plt.plot(goodputACPC.keys(), goodputACPC.values(), '-', color='tab:purple', linewidth=0.9, label=f'ACPC({K}, {K-(t+1)*5})', marker='*')
for k, v in goodputACPC.items():
    plt.annotate(f'{np.round(v, 4)}', (k,v), textcoords="offset points", xytext=(0, 8), ha='center')
plt.plot(goodputPZ.keys(), goodputPZ.values(), '--', color='tab:green', linewidth=0.9, label=f'Pilot-Zero {np.round(singlePZ, 4)}', marker='d')
for k, v in goodputPZ.items():
    plt.annotate(f'{np.round(v, 4)}', (k, v), textcoords='offset points', xytext=(0, 8), ha='center')
plt.xlabel('SNR(dB)')
plt.ylabel('Goodput')
plt.title(f"{noIter} runs per point")
plt.legend(loc='upper left', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PilotZero/ACPC/goodput_{t}.jpeg")