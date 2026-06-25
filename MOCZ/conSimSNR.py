"""
Throughput analuysis of BMOCZ + IPSA for fixed load and varying SNR
System parameters:
- Total number of users(n) = 20
- Number of slots per frame(m) = 20
- Channel: Rayleigh Block-fading 
- Payload (packet size): 32 bits
- Slot distribution: CRDSA (x**2)
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from BMOCZ import (
    BMOCZReceiver,
    BMOCZTransmitter
)
from CHANNEL import (
    SlowFadingChannel,
    ChannelEstimation
)
from simulation import Simulation

degree = 2 
m = 20
n = m
noIter = 100
K = 32
SNR_dB = np.arange(-10, 21, 5)
G = np.linspace(0.1, 1, 10)
signal_power = 1

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
chEst = ChannelEstimation()

thr_load = {}; per_load = {}; ber_load = {}
for load in G:
    throughput = {}; per = {}; ber = {}
    for snr in SNR_dB:
        THROUGHPUT, BER, PER = 0, 0, 0
        noise_var = signal_power * 10**(-snr/10)
        ch = SlowFadingChannel(noise_var)
        sim = Simulation(tx, rx, ch, chEst, m, n, degree, K, Q=4)
        userSlotsGen = sim.userSlots
        for i in range(noIter):
            FRAME = {}
            slot = set()
            activeUsers = sorted( random.sample( range(1, n+1), int(load * n) ) )
            for userId in activeUsers:
                userSlot = userSlotsGen[userId]
                for s in userSlot:
                    if s not in slot:
                        FRAME[s] = [userId]
                        slot.add(s)
                    else:
                        FRAME[s] += [userId]
            FRAME = dict( sorted( FRAME.items(), reverse=False ) )
            frame, h = sim.frameBuild(FRAME)
            frameBAPM = sim.genBAPM(activeUsers)

            msg_hat, h_hat = sim.frameParse(frame, frameBAPM)

            pcr, bcr_frame = sim.per(msg_hat)
            PER += ( 1 - ( pcr / len(activeUsers) ) )
            BER += ( 1 - (bcr_frame / (K * len(activeUsers))) )
            THROUGHPUT += (bcr_frame / (K * len(activeUsers)))
        throughput[snr] = THROUGHPUT / noIter
        per[snr] = PER / noIter
        ber[snr] = (BER / noIter).astype(float)
    print(f"Load - {load} done")
    thr_load[load] = throughput
    per_load[load] = per
    ber_load[load] = ber

plt.figure(figsize=(8,6), dpi=800)
for k, v in thr_load.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"Load = {k}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR(dB)")
plt.ylabel("Throughput (T)")
plt.ylim(0, 1.05)
plt.title(f"Throughput vs SNR over {noIter} iterations")
plt.legend(loc='lower left', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig("results/ConSim/mthrSNR.jpeg")