"""
Comparisions of BMOCZ-PZ decoders to select the best among them
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import BMOCZ, MultiPathFading, PerformanceParameters
K = 16
Q = 32
SNR_dB = np.arange(-5, 41, 3)
signalPower = 1
noIter = int(1e3)
perParam = PerformanceParameters()
berK, perK, rotationEstK = {}, {}, {}
schemes = ['singlePZ', 'majorityVote', 'FRO', 'FRO_PZ']
colors = ["tab:red", "tab:green", "tab:blue", 'tab:purple']
linestyles = ["--", "-.", "-", ":"]
markers = ["o", "s", "^", "d"]
bmoczSys = BMOCZ(K)
ber, per, rotation_est = {}, {}, {}
for scheme in schemes:    
    singlePZ = [-1.25*bmoczSys.R]
    berSNR, perSNR, rotationEstSNR = {}, {}, {}
    for snr in SNR_dB:
        noiseVar = signalPower * 10**(-snr/10)
        ch = MultiPathFading(noiseVar)
        BER, PCR, rotationEst = 0, 0, 0
        for _ in range(noIter):
            msgTx = np.random.randint(0, 2, K)
            sigTx = bmoczSys.coeffCon(msgTx, singlePZ)
            sigPower = np.mean(np.abs(sigTx)**2)
            sigTx /= np.sqrt(sigPower)

            rotation = np.random.uniform(0, 2*np.pi)
            sigRx = ch.transmit(sigTx, rotation)
            if scheme == 'singlePZ':
                msgRx, rotation_hat = bmoczSys.singlePZDecodedMsg(sigRx, Q, singlePZ)
            elif scheme == 'majorityVote':
                msgRx, rotation_hat = bmoczSys.pilotDecoder_MV(sigRx, Q, singlePZ)
            elif scheme == 'FRO':
                msgRx, rotation_hat = bmoczSys.pilotDecoder_FRO(sigRx, Q, singlePZ)
            elif scheme == 'FRO_PZ':
                msgRx, rotation_hat = bmoczSys.pilotDecoder_FracInt(sigRx, Q, singlePZ)
            BER += perParam.ber(msgRx, msgTx)
            PCR += perParam.pcr(msgRx, msgTx)
            rotationEst += abs(rotation_hat - rotation) / rotation
        berSNR[snr] = BER / noIter
        perSNR[snr] = 1 - PCR/noIter
        rotationEstSNR[snr] = rotationEst / noIter
    print(f"{scheme} done")
    ber[scheme] = berSNR
    per[scheme] = perSNR
    rotation_est[scheme] = rotationEstSNR
    
plt.figure(1, dpi=800)
for i, (k, v) in enumerate(ber.items()):
    plt.semilogy(v.keys(), v.values(), linewidth=0.9, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=5, label=k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title(f"BER analysis for BlockLen {K}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.ylim(1e-5, 1)
plt.tight_layout()
plt.savefig(f"results/BMOCZ/berK{K}.jpeg")

plt.figure(2, dpi=800)
for i, (k, v) in enumerate(per.items()):
    plt.semilogy(v.keys(), v.values(), linewidth=0.9, linestyle=linestyles[i],
                color=colors[i], marker=markers[i], markersize=5, label=k)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.title(f"PER analysis for BlockLen {K}")
plt.legend(loc='lower left', framealpha=0.6, fontsize=7)
plt.ylim(1e-5, 1)
plt.tight_layout()
plt.savefig(f"results/BMOCZ/perK{K}.jpeg")