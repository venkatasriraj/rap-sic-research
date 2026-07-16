"""
Monte-carlo simulations for Zero-Marker in identifying and correcing the 
phase rotation due to CFO.
We will be analyzing the system performance over a given block-length.
For a block-length(K) we will be analyzing how
- BER vs K
- PAPR vs K
NOTE: Channel estimation is halted in the case of multipath fading 
and will be persued later. 
- MSE of h_est vs K for SNR = 15dB.

- The choice of Q for DiZeT decoder is also something to be looked at.
"""
import numpy as np
import matplotlib.pyplot as plt
from BMOCZ import (
    BMOCZReceiver,
    BMOCZTransmitter
)
from CHANNEL import MultiPathFading

K = np.arange(6, 41, 1)
Q = 8
noIter = 10
SNR_dB = np.arange(-10, 21, 5)
signal_power = 1 

ber_snr = {}; papr_snr = {}; per_snr = {}; thr_snr = {}
for snr in SNR_dB:
    noise_var = signal_power * 10**(-snr/10)
    ch = MultiPathFading(noise_var=noise_var, pathLoss=1)
    ber = {}; papr = {}; per = {}; throughput = {}
    for k in K:
        tx = BMOCZTransmitter(k)
        rx = BMOCZReceiver(k)
        BER, PAPR, PCR = 0, 0, 0
        for i in range(noIter):
            msg = np.random.randint(0, 2, k, dtype=np.uint8)

            sig_tx = tx.coeffConZM(msg)
            sig_power = np.mean(np.abs(sig_tx)**2)
            sig_norm = sig_tx / np.sqrt(sig_power)

            rotation = np.random.uniform(0, 2*np.pi)
            sig_rx = ch.transmit(sig_norm, rotation)

            sig_ffo = rx.ffoEstCor(sig_rx, Q)
            msg_rx = rx.fftDizet(sig_ffo, Q)

            if k % 4 == 0:
                int_rotation_est = rx.fftConPZ(sig_ffo)
            else:
                int_rotation_est = rx.BLodd(sig_ffo)
            # int_rotation_est = rx.intRotationEst(sig_ffo)
            msg_hat = np.roll(msg_rx, int_rotation_est)

            BER += rx.ber(msg_hat, msg)
            PAPR += tx.PAPR(sig_tx)
            PCR += rx.per(msg_hat, msg)
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
plt.title(f"BER vs K over {noIter} packets")
plt.ylim(0, 1.05)
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/IRO/zmBERQ{Q}.jpeg")

plt.figure(2, dpi=800)
for k, v in papr_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"SNR = {k}dB")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Msg-Length(K)")
plt.ylabel("Peak to Average Power Ratio (PAPR)")
plt.title(f"PAPR vs K over {noIter} packets")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/IRO/zmPAPRQ{Q}.jpeg")

plt.figure(3, dpi=800)
for k, v in per_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f'SNR = {k}dB')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Msg-length(K)")
plt.ylabel("Packet Error Rate (PER)")
plt.title(f"PER vs K for {noIter} packets")
plt.ylim(0, 1.05)
plt.legend(loc='lower right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/IRO/zmPERQ{Q}.jpeg")

plt.figure(4, dpi=800)
for k, v in thr_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"SNR = {k}dB")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Msg-length(K)")
plt.ylabel("Throughpyt(T)")
plt.title(f"Throughput vs K for {noIter} packets")
plt.ylim(0, 1.05)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/IRO/zmThroughputQ{Q}.jpeg")

# plt.show()