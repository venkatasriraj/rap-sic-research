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
from BMOCZ import (
    BMOCZReceiver,
    BMOCZTransmitter
)
from CHANNEL import MultiPathFading

K = np.arange(6, 41)
Q = 16
noIter = 10
SNR_dB = np.arange(-10, 21, 10)
signal_power = 1

ber_snr = {}; papr_snr = {}; thr_snr = {}; rotation_snr = {}
for snr in SNR_dB:
    noise_var = signal_power * 10**(-snr/10)
    ch = MultiPathFading(noise_var, pathLoss=1)
    ber = {}; papr = {}; throughput = {}; rotationMAE = {}
    for k in K:
        tx = BMOCZTransmitter(k)
        rx = BMOCZReceiver(k)
        singlePZ = [2 * tx.R]
        BER, PCR, PAPR, ROTATION = 0, 0, 0, 0
        for i in range(noIter):
            msg = np.random.randint(0, 2, k)

            sig_tx = tx.coeffConSinglePZ(msg, singlePZ)
            sig_power = np.mean( np.abs(sig_tx)**2 )
            sig_tx /= np.sqrt(sig_power) 

            rotation = np.random.uniform(0, np.pi*2)
            sig_rx = ch.transmit(sig_tx, rotation)
            
            msg_hat, rotation_hat = rx.singlePZDecodedMsg(sig_rx, Q, singlePZ)
            
            BER += rx.ber(msg_hat, msg)
            PCR += rx.per(msg_hat, msg)
            PAPR += tx.PAPR(sig_tx)
            ROTATION += np.abs(rotation - rotation_hat) / rotation
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
plt.title(f"BER vs K for {noIter} iters")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7) 
plt.tight_layout()
plt.savefig(f"results/singlePZ/berQ{Q}")   

plt.figure(2, dpi=800)
for k, v in papr_snr.items():
    plt.plot(list(v.keys()), list(v.values()), linestyle='-', linewidth=0.9, label=f"SNR = {k} dB")
plt.xlabel("Msg-len(K)")
plt.ylabel("PAPR")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"PAPR vs K for {noIter} iters")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/singlePZ/paprQ{Q}")

plt.figure(3, dpi=800)
for k, v in thr_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"SNR = {k} dB")
plt.xlabel("Msg-len(K)")
plt.grid(True, alpha=0.6, linestyle='--')
plt.ylabel("Throughput")
plt.title(f"Throughput vs K for {noIter} iters")
plt.legend(loc='upper right', fontsize=7, framealpha=0.5)
plt.tight_layout()
plt.savefig(f"results/singlePZ/thrQ{Q}")

plt.figure(4, dpi=800)
for k, v in rotation_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"SNR = {k} dB")
plt.xlabel("Msg-len(K)")
plt.ylabel("MAE of estimated rotation")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"MAE of rotation vs K for {noIter} iters")
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/singlePZ/rotationQ{Q}")