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
noIter = 10

pktSize = 32   # in bits (4B)
LOAD = np.linspace(0.1, 1, 10)
SNR_dB = np.arange(-10, 21, 5)
signal_power = 1 
uId = 1
pathLoss = 1

base = BPSKBase()
chEst = ChannelEstimation()

thr_load = {}; per_load = {}; ber_load = {}; maeh_load = {}
# userCount = {}; slotCount = {}
for load in LOAD:
    throughput = {}; per = {}; ber = {}; maeh = {}
    for snr in SNR_dB:
        PER, BER, THROUGHPUT, MAE, MAE_count = 0, 0, 0, 0, 1e-10
        noise_var = signal_power * 10**(-snr/10)
        ch = SlowFadingChannel(noise_var, pathLoss)
        if lenAC == 0:
            pilot = []
        else:
            pilot = accessCode[:lenAC]
        seedNo = abs(int(load*n*3 + snr) )
        sim = Simulation(base, ch, chEst, m, n, degree, pktSize, pilot, seedNo)
        for i in range(noIter):
            userSlotsGen = sim.userSlotGen()
            FRAME = {}
            slot = set()
            activeUsers = sorted( sim.rng.sample( range(1, n+1), int(load*n) ))
            # print(f"userSlots: {userSlotsGen}")
            # print(f"    active users: {activeUsers}")
            for userId in activeUsers:
                userSlot = userSlotsGen[userId]
                # if userId not in userCount.keys():
                #     userCount[userId] = 1
                # else:
                #     userCount[userId] += 1
                for s in userSlot:
                    if s not in slot:
                        FRAME[s] = [userId]
                        slot.add(s)
                    else:
                        FRAME[s] += [userId]
                    # if s not in slotCount.keys():
                    #     slotCount[s] = 1
                    # else:
                    #     slotCount[s] += 1
            FRAME = dict( sorted( FRAME.items(), reverse=False ) )
            frame, h = sim.frameBuild(FRAME)
            frameBAPM = sim.genBAPM(activeUsers, userSlotsGen)
            # print(frameBAPM)
            # we wil be receiving the packet (pilot + payload) for ber, per analysis
            # for throughput we will only be considering the payload
            pkt_hat, h_hat = sim.frameParse(frame, frameBAPM, userSlotsGen)
            if uId in activeUsers:
                mae_temp, count = sim.mae(h, h_hat, uId)
                MAE += mae_temp
                MAE_count += count
            pcr, bcr_frame = sim.per(pkt_hat)
            PER += ( 1 - (pcr/len(activeUsers)) )
            BER += ( 1 - ( bcr_frame / ( pktSize * len(activeUsers) ) ) )
            THROUGHPUT += pcr / len(activeUsers) 
            # print(f"Thr: {THROUGHPUT}, MAE: {MAE}, MAE count: {MAE_count}")
        per[snr] = PER / noIter
        ber[snr] = (BER / noIter).astype(float)
        throughput[snr] = THROUGHPUT / noIter
        maeh[snr] = MAE / MAE_count
    print(f"Load - {load} done")
    thr_load[load] = throughput
    per_load[load] = per
    ber_load[load] = ber
    maeh_load[load] = maeh
# slotCount = dict( sorted( slotCount.items(), reverse=False ) )
# userCount = dict( sorted( userCount.items(), reverse=False ) )
# print(f"slot count for load-0.1: {slotCount}")
# print(f"user-1for load 0.1: {userCount}")
plt.figure(figsize=(8,6), dpi=800)
for k, v in thr_load.items():
    plt.plot(v.keys(), v.values(), linewidth=0.9, linestyle='-', label=f"Load - {k}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR(dB)")
plt.ylabel("Throughput (T)")
plt.ylim(0, 1.05)
plt.title(f"Throughput vs SNR for {noIter} Iters with PilotLen {lenAC}")
plt.legend(loc='upper left', fontsize=7, framealpha=0.6)
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