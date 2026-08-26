"""
Comparision of designed Pilot-zero MOCZ with Tradtional MOCZ with ACPC to counter the CFO
Using ACPC(31, 16) will be correcting 2 bit errors wrt to the received message
while the pilot-zero code doesn't have any inherent error-correction capability
Pilot-Zero will be sending K bits so both ACPC and PZ have different symbol duration
FRO: Fractional Rotation Offset
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import(
    BMOCZTransmitter, BMOCZReceiver, ACPC,
    MultiPathFading, PerformanceParameters
)
m = 5
t = 2
K = 2**m - 1 
Q = 32
signal_power = 1
noIter = int(1e4)
SNR_dB = np.arange(-10, 21, 5)

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
perParam = PerformanceParameters()
acpc = ACPC(m, t)
singlePZ = [-1.25 * tx.R]
berACPC = {}; paprACPC = {}; perACPC = {}; lACPC = {}
berPZ = {}; paprPZ = {}; perPZ = {}; rotationPZ = {}
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
        acpcSigTx = tx.coeffCon(acpcMsgEn)
        acpcSigTx /= np.sqrt( np.mean(np.abs(acpcSigTx)**2) )
        acpcSigRx = ch.transmit(acpcSigTx, rotation)
        acpcSigFRO = rx.ffoEstCor(acpcSigRx, Q)
        estCodeword = rx.fftDizet(acpcSigFRO, Q)
        acpcEstMsg, acpcEst_l = acpc.codeword_decoding(estCodeword)
        BERacpc += perParam.ber(acpcEstMsg, acpcMsg)
        PCRacpc += perParam.pcr(acpcEstMsg, acpcMsg)
        PAPRacpc += tx.PAPR(acpcSigTx)
        lacpc += abs(acpcEst_l - l)/(l+1)

        pzMsg = np.random.randint(0, 2, K)
        pzSigTx = tx.coeffConSinglePZ(pzMsg, singlePZ)
        pzSigTx /= np.sqrt( np.mean(np.abs(pzSigTx)**2) )
        pzSigRx = ch.transmit(pzSigTx, rotation)
        msg_hat, rotate_hat = rx.singlePZDecodedMsg(pzSigRx, Q, singlePZ)
        BERpz += perParam.ber(msg_hat, pzMsg)
        PCRpz += perParam.pcr(msg_hat, pzMsg)
        PAPRpz += tx.PAPR(pzSigTx)
        deviationR = np.abs(rotation - rotate_hat) 
        maeRotate = deviationR if deviationR <= np.pi else 2*np.pi - deviationR
        rotationpz += maeRotate / rotation
    berACPC[snr] = BERacpc / noIter
    paprACPC[snr] = PAPRacpc / noIter
    perACPC[snr] = 1 - (PCRacpc / noIter)
    lACPC[snr] = lacpc / noIter
    print(f"SNR: {snr} done.")
    berPZ[snr] = BERpz / noIter
    paprPZ[snr] = PAPRpz / noIter
    perPZ[snr] = 1 - (PCRpz / noIter)
    rotationPZ[snr] = rotationpz / noIter

plt.figure(1, dpi=800)
plt.plot(berACPC.keys(), berACPC.values(), '-', color='tab:purple', linewidth=0.9, label=f"ACPC ({K}, {K-5-2*(2*t+1)})")    
plt.plot(berPZ.keys(), berPZ.values(), '--', color='tab:green', linewidth=0.9, label="Pilot-Zero")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.ylim(0, 1.05)
plt.title(f"{noIter} runs per point")
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/pzACPC/ber_{K}.jpeg")

plt.figure(2, dpi=800)
plt.plot(perACPC.keys(), perACPC.values(), '-', color='tab:purple', linewidth=0.9, label=f"ACPC ({K}, {K-5-2*(2*t+1)})")    
plt.plot(perPZ.keys(), perPZ.values(), '--', color='tab:green', linewidth=0.9, label="Pilot-Zero")
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.ylim(0, 1.05)
plt.title(f"{noIter} runs per point")
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/pzACPC/per_{K}.jpeg")

plt.figure(3, dpi=800)
plt.plot(paprACPC.keys(), paprACPC.values(), '-', color='tab:purple', linewidth=0.9, label=f"ACPC ({K}, {K-5-2*(2*t+1)})")    
plt.plot(paprPZ.keys(), paprPZ.values(), '--', color='tab:green', linewidth=0.9, label="Pilot-Zero")
plt.xlabel("SNR (dB)")
plt.ylabel("PAPR")
plt.title(f"{noIter} runs per point")
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/pzACPC/papr_{K}.jpeg")

plt.figure(4, dpi=800)
plt.plot(lACPC.keys(), lACPC.values(), '-', color='tab:purple', linewidth=0.9) 
plt.xlabel("SNR (dB)")
plt.ylabel("Normalised MAE of Integer Ration Offset")
plt.title(f"{noIter} runs per point")
plt.savefig(f"results/pzACPC/lACPC_{K}.jpeg")

plt.figure(5, dpi=800)
plt.plot(rotationPZ.keys(), rotationPZ.values(), '--', color='tab:green', linewidth=0.9)
plt.xlabel("SNR (dB)")
plt.ylabel("MAE of estimated rotation")
plt.title(f"{noIter} runs per point")
plt.savefig(f"results/pzACPC/rotationPZ_{K}.jpeg")