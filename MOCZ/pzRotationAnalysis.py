"""
we will be considering the pilot-zero at angle 0 on circle of radius 2*R
and estimate the rotation and plot the MAE of it
We will be studying
- does it improve throughput
- can it applied for all block-lengths
- will there be a constraint on the over-sampling factor(Q) in estimating the fractional rotation
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import (
    BMOCZ, MultiPathFading, PerformanceParameters
)
K = np.arange(11, 31)
Q = 32
noIter = int(1e2)
SNR_dB = np.arange(-10, 21, 5)
signal_power = 1
perParam = PerformanceParameters()
ber_snr = {}; papr_snr = {}; thr_snr = {}; rotation_snr = {}
for snr in SNR_dB:
    noise_var = signal_power * 10**(-snr/10)
    ch = MultiPathFading(noise_var, pathLoss=1)
    ber = {}; papr = {}; throughput = {}; rotationMAE = {}
    for k in K:
        bmoczSystem = BMOCZ(k)
        singlePZ = [-1.25 * bmoczSystem.R]
        BER, PCR, PAPR, ROTATION = 0, 0, 0, 0
        for i in range(noIter):
            msg = np.random.randint(0, 2, k)

            sig_tx = bmoczSystem.coeffCon(msg, singlePZ)
            sig_power = np.mean( np.abs(sig_tx)**2 )
            sig_tx /= np.sqrt(sig_power) 

            rotation = np.random.uniform(0, np.pi*2)
            sig_rx = ch.transmit(sig_tx, rotation)
            
            msg_hat, rotate_hat = bmoczSystem.singlePZDecodedMsg(sig_rx, Q, singlePZ)
            
            BER += perParam.ber(msg_hat, msg)
            PCR += perParam.pcr(msg_hat, msg)
            PAPR += bmoczSystem.PAPR(sig_tx)
            # maeRotate = np.abs(rotation - rotation_hat) if rotation < np.pi else np.abs(rotation - rotation_hat - 2*np.pi) 
            # if rotation < np.angle(singlePZ[0]):
            #     maeRotate = np.abs(rotation + np.pi - rotate_hat)
            # else:
            #     maeRotate = np.abs(rotation - np.pi - rotate_hat)
            deviationR = np.abs(rotation - rotate_hat)
            maeRotate = deviationR if deviationR <= np.pi else 2*np.pi - deviationR
            # print(f"rotation: {rotation}, est rotation: {rotation_hat}, {maeRotate}, {maeRotate/rotation}")
            ROTATION += maeRotate / rotation
        ber[k] = BER / noIter
        papr[k] = PAPR / noIter
        throughput[k] = PCR / noIter
        rotationMAE[k] = ROTATION / noIter
    print(f"SNR: {snr} done")
    ber_snr[snr] = ber
    papr_snr[snr] = papr
    thr_snr[snr] = throughput
    rotation_snr[snr] = rotationMAE

plt.figure(1, dpi=800)
for k, v in ber_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f'SNR = {k} dB')
plt.grid(True, alpha=0.6, linestyle='--')
plt.xlabel("Msg-len(K)")
plt.ylabel("BER")
plt.ylim(0,1.05)
plt.title(f"{noIter} packets per point")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7) 
plt.tight_layout()
plt.savefig(f"results/PilotZero/RotationAnalysis/berQ{Q}")   

plt.figure(2, dpi=800)
for k, v in papr_snr.items():
    plt.plot(list(v.keys()), list(v.values()), linestyle='-', linewidth=0.9, label=f"SNR = {k} dB")
plt.xlabel("Msg-len(K)")
plt.ylabel("PAPR")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"{noIter} packets per point")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/RotationAnalysis/paprQ{Q}")

plt.figure(3, dpi=800)
for k, v in thr_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"SNR = {k} dB")
plt.xlabel("Msg-len(K)")
plt.grid(True, alpha=0.6, linestyle='--')
plt.ylabel("Throughput")
plt.title(f"{noIter} packets per point")
plt.legend(loc='upper right', fontsize=7, framealpha=0.5)
plt.tight_layout()
plt.savefig(f"results/PilotZero/RotationAnalysis/thrQ{Q}")

plt.figure(4, dpi=800)
for k, v in rotation_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"SNR = {k} dB")
plt.xlabel("Msg-len(K)")
plt.ylabel("MAE of estimated rotation")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"{noIter} packets per point")
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PilotZero/RotationAnalysis/rotationQ{Q}")