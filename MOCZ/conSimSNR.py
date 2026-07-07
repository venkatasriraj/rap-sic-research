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
noIter = 10
K = 32
SNR_dB = np.arange(-10, 21, 5)
G = np.linspace(0.1, 1, 10)
signal_power = 1
uId = 1

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
chEst = ChannelEstimation()

thr_load = {}; per_load = {}; ber_load = {}; mae_hEst = {}
for load in G:
    throughput = {}; per = {}; ber = {}; mae_herr = {}
    for snr in SNR_dB:
        THROUGHPUT, BER, PER, MAE, MAE_count = 0, 0, 0, 0, 1e-10
        noise_var = signal_power * 10**(-snr/10)
        ch = SlowFadingChannel(noise_var)
        sim = Simulation(tx, rx, ch, chEst, m, n, degree, K, Q=4)
        userSlotsGen = sim.userSlots
        for i in range(noIter):
            random.seed(int(load*n) + i)
            FRAME = {}
            slot = set()
            activeUsers = sorted( random.sample( range(1, n+1), int(load * n) ) )
            # if snr == 20:
            #     print(f"Active Users: {activeUsers}")
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
            if uId in activeUsers:
                mae_temp, count = sim.maeh(h, h_hat, uId)
                MAE += mae_temp
                MAE_count += count
            pcr, bcr_frame = sim.per(msg_hat)
            PER += ( 1 - ( pcr / len(activeUsers) ) )
            BER += ( 1 - (bcr_frame / (K * len(activeUsers))) )
            THROUGHPUT += ( pcr /  len(activeUsers) )
        throughput[snr] = THROUGHPUT / noIter
        per[snr] = PER / noIter
        ber[snr] = (BER / noIter).astype(float)
        mae_herr[snr] = MAE / MAE_count
    print(f"Load - {load} done")
    thr_load[load] = throughput
    per_load[load] = per
    ber_load[load] = ber
    mae_hEst[load] = mae_herr

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

plt.figure(figsize=(8,6), dpi=800)
for k, v in mae_hEst.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f'Load = {k}')
plt.grid(True, alpha=0.6, linestyle='--')
plt.xlabel("SNR(dB)")
plt.ylabel(f"MAE of h_est for user-{uId}")
plt.title(f"MAE of h_est vs SNR over {noIter} frames")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/ConSim/mhSNR.jpeg")