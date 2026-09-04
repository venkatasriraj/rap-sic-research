"""
PAPR analysis for pilot zero inserted in the BMOCZ message zeros
using updated method for Pilot-Zero
And the pilot is placed at three locations [-1, 1j, -1j]
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import (
    BMOCZ, MultiPathFading, PerformanceParameters
)

K = np.arange(6, 41, 1)
Q = 4
noIter = int(1e2)
SNR_dB = np.arange(-10, 21, 5)
signal_power = 1 
singlePZ = [-1]
perParam = PerformanceParameters()
ber_snr = {}; papr_snr = {}; per_snr = {}; thr_snr = {}
for snr in SNR_dB:
    noise_var = signal_power * 10**(-snr/10)
    ch = MultiPathFading(noise_var, pathLoss=1)
    ber = {}; papr = {}; per = {}; throughput = {}
    for k in K:
        bmoczSystem = BMOCZ(k)
        BER, PAPR, PCR = 0, 0, 0
        for i in range(noIter):
            msg = np.random.randint(0, 2, k, dtype=np.uint8)

            sig_tx = bmoczSystem.coeffCon(msg, singlePZ)
            sig_power = np.mean(np.abs(sig_tx)**2)
            sig_tx /= np.sqrt(sig_power)

            rotation = np.random.uniform(0, 2*np.pi)
            sig_rx = ch.transmit(sig_tx, rotation)

            msg_hat, _ = bmoczSystem.singlePZDecodedMsg(sig_rx, Q, singlePZ)

            BER += perParam.ber(msg_hat, msg)
            PAPR += bmoczSystem.PAPR(sig_tx)
            PCR += perParam.pcr(msg_hat, msg)
        ber[k] = BER / noIter
        papr[k] = PAPR / noIter
        per[k] = 1 - (PCR / noIter)
        throughput[k] = PCR / noIter
    print(f"SNR - {snr} done")
    ber_snr[snr] = ber
    papr_snr[snr] = papr
    per_snr[snr] = per
    thr_snr[snr] = throughput

plt.figure(1, dpi=800)
for k, v in ber_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f'SNR = {k}dB')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Msg-Length(K)")
plt.ylabel("BER")
plt.title(f"{noIter} packets per point")
plt.ylim(0, 1.05)
plt.legend(loc='upper left', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/PilotZero/PAPR/BERQ{Q}.jpeg")

plt.figure(2, dpi=800)
for k, v in papr_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"SNR = {k}dB")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Msg-Length(K)")
plt.ylabel("Peak to Average Power Ratio (PAPR)")
plt.title(f"{noIter} packets per point")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/PAPR/PAPRQ{Q}.jpeg")

plt.figure(3, dpi=800)
for k, v in per_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f'SNR = {k}dB')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Block-length(K)")
plt.ylabel("Msg Error Rate (PER)")
plt.title(f"{noIter} packets per point")
plt.ylim(0, 1.05)
plt.legend(loc='lower right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/PAPR/PERQ{Q}.jpeg")

plt.figure(4, dpi=800)
for k, v in thr_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"SNR = {k}dB")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Msg-length(K)")
plt.ylabel("Throughpyt(T)")
plt.title(f"{noIter} packets per point")
plt.ylim(0, 1.05)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/PilotZero/PAPR/ThroughputQ{Q}.jpeg")