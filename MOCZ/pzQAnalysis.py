"""
Here we will be studying the impact of Q over-sampling factor 
in estimation of rotation and MAE of it
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

Q = np.arange(4, 33, 4)
K = np.arange(13, 21)
noIter = 1000
snr = 20
signal_power = 1

noise_var = signal_power * 10**(-snr/10)
ch = MultiPathFading(noise_var)
ber_K = {}; thr_K = {}; rotation_K = {}
for k in K:
    tx = BMOCZTransmitter(k)
    rx = BMOCZReceiver(k)
    singlePZ = [2*tx.R]
    ber = {}; throughput = {}; rotationMAE = {}
    for q in Q:
        BER, PCR, ROTATION = 0, 0, 0
        for i in range(noIter):
            msg = np.random.randint(0, 2, k)

            sig_tx = tx.coeffConSinglePZ(msg, singlePZ)
            sig_power = np.mean(np.abs(sig_tx)**2)
            sig_tx /= np.sqrt(sig_power)

            rotation = np.random.uniform(0, 2*np.pi)
            sig_rx = ch.transmit(sig_tx, rotation)

            msg_hat, rotation_hat = rx.singlePZDecodedMsg(sig_rx, q, singlePZ)

            BER += rx.ber(msg_hat, msg)
            PCR += rx.per(msg_hat, msg)
            ROTATION += np.abs(rotation - rotation_hat) / rotation
        ber[q] = BER / noIter
        throughput[q] = PCR / noIter
        rotationMAE[q] = (ROTATION / noIter).astype(float)
    ber_K[k] = ber
    thr_K[k] = throughput
    rotation_K[k] = rotationMAE
    print(f"Msg-len: {k} done")

plt.figure(1, dpi=800)
for k, v in ber_K.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"Msg-Len: {k}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Over-Sampling factor(Q)")
plt.ylabel("BER")
plt.title(f"BER vs Q for {noIter} iters")
plt.tight_layout()
plt.savefig(f"results/singlePZ/Q/ber{K}.jpeg")

plt.figure(2, dpi=800)
for k, v in thr_K.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"Msg-Len: {k}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Over-Sampling factor(Q)")
plt.ylabel("Throughput")
plt.title(f"Throughput vs Q for {noIter} iters")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/singlePZ/Q/thr{K}.jpeg")

plt.figure(3, dpi=800)
for k, v in rotation_K.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"Msg-Len: {k}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Over-Sampling factor(Q)")
plt.ylabel("MAE of rotation")
plt.title(f"MAE of rotation vs Q for {noIter} iters")
plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/singlePZ/Q/rotation{K}.jpeg")