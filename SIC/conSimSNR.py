"""
Throughput analysis of DBPSK + IRSA for fixed load and varying SNR
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
from BPSK import BPSKBase
from CHANNEL import (
    SlowFadingChannel,
    ChannelEstimation
)
from simulation import Simulation

accessCode = [1,0,1,0,1,0,1,0]
lenAC = 8
degree = 2
m = 20
n = m  # number of users
noIter = 1000

pktSize = 32   # in bits (4B)
LOAD = np.linspace(0.1, 1, 10)
SNR_dB = np.arange(-10, 21, 5)
signal_power = 1 
uId = 1

base = BPSKBase()
chEst = ChannelEstimation()

thr_load = {}; per_load = {}; ber_load = {}; maeh_load = {}
for load in LOAD:
    throughput = {}; per = {}; ber = {}; maeh = {}
    for snr in SNR_dB:
        PER, BER, THROUGHPUT, MAE, MAE_count = 0, 0, 0, 0, 1e-10
        noise_var = signal_power * 10**(-snr/10)
        ch = SlowFadingChannel(noise_var)
        if lenAC == 0:
            pilot = []
        else:
            pilot = accessCode[:lenAC]
        sim = Simulation(base, ch, chEst, m, n, degree, pktSize, pilot)
        userSlotsGen = sim.userSlots
        for i in range(noIter):
            random.seed(int(load*n) + i)
            FRAME = {}
            slot = set()
            activeUsers = sorted( random.sample( range(1, n+1), int(load*n) ))
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
            # we wil be receiving the packet (pilot + payload) for ber, per analysis
            # for throughput we will only be considering the payload
            pkt_hat, h_hat = sim.frameParse(frame, frameBAPM)
            if uId in activeUsers:
                mae_temp, count = sim.mae(h, h_hat, uId)
                MAE += mae_temp
                MAE_count += count
            pcr, bcr_frame = sim.per(pkt_hat)
            PER += ( 1 - (pcr/len(activeUsers)) )
            BER += ( 1 - ( bcr_frame / ( pktSize * len(activeUsers) ) ) )
            THROUGHPUT += pcr / len(activeUsers) 
        per[snr] = PER / noIter
        ber[snr] = (BER / noIter).astype(float)
        throughput[snr] = THROUGHPUT / noIter
        maeh[snr] = MAE / MAE_count
    print(f"Load - {load} done")
    thr_load[load] = throughput
    per_load[load] = per
    ber_load[load] = ber
    maeh_load[load] = maeh

plt.figure(figsize=(8,6), dpi=800)
for k, v in thr_load.items():
    plt.plot(v.keys(), v.values(), linewidth=0.9, linestyle='-', label=f"Load - {k}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR(dB)")
plt.ylabel("Throughput (T)")
plt.ylim(0, 1.05)
plt.title(f"Throughput vs SNR for {noIter} Iters with PilotLen {lenAC}")
plt.legend(loc='lower left', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/ConSim/PilotLen/d{lenAC}thrSNR.jpeg")

plt.figure(2, dpi=800)
for k, v in maeh_load.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"Load = {k}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR(dB)")
plt.ylabel(f"MAE of h_est for User-{uId}")
plt.title(f"MAE of h_est vs SNR for {noIter} iters")
plt.legend(loc="upper right", framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig(f"results/ConSim/PilotLen/d{lenAC}hSNR.jpeg")