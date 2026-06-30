"""
Throughput analysis of DBPSK + IRSA for fixed SNR and varying Load
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

accessCode = [1, 0] * 4
lenAc = 8
degree = 2 
m = 20; n = m
noIter = 1000

pktSize = 32
SNR_dB = np.arange(-10, 21, 5)
LOAD = np.linspace(0.1, 1, 10)
signal_power = 1

base = BPSKBase()
chEst = ChannelEstimation()

thr_snr = {}; per_snr = {}; ber_snr = {}
for snr in SNR_dB:
    throughput = {}; per = {}; ber = {}
    noise_var = signal_power * 10**(-snr/10)
    ch = SlowFadingChannel(noise_var)
    if lenAc == 0:
        pilot = []
    else:
        pilot = accessCode[:lenAc]
    sim = Simulation(base, ch, chEst, m, n, degree, pktSize, pilot)
    userSlotsGen = sim.userSlots
    for load in LOAD:
        PER, THROUGHPUT, BER = 0, 0, 0
        for i in range(noIter):
            FRAME = {}
            slot = set()
            activeUsers = sorted(random.sample( range(1, n+1), int(load*n) ))
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

            pkt_hat, h_hat = sim.frameParse(frame, frameBAPM)

            pcr, bcr_frame = sim.per(pkt_hat)
            PER += ( 1 - (pcr/(len(activeUsers))) )
            BER += ( 1 - ( bcr_frame / (pktSize * len(activeUsers)) ) )
            THROUGHPUT += pcr /  len(activeUsers) # here we have not considered rate = (pktsize - aclen) / pktsize
        throughput[load] = THROUGHPUT / noIter
        ber[load] = (BER / noIter).astype(float)
        per[load] = PER / noIter
    print(f"SNR - {snr} done")
    thr_snr[snr] = throughput
    per_snr[snr] = per
    ber_snr[snr] = ber

plt.figure(figsize=(8,6), dpi=800)
for k, v in thr_snr.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"SNR = {k}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Load(g)")
plt.ylim(0, 1.05)
plt.ylabel("Throughpt (T)")
plt.title(f"Throughput vs Load for {noIter} iters with PilotLen {lenAc}")
plt.legend(loc='lower left', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig(f"results/ConSim/PilotLen/d{lenAc}thrLoad.jpeg")