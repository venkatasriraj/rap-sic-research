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
import matplotlib.pyplot as plt
from BPSK import BPSKBase
from CHANNEL import (
    SlowFadingChannel,
    ChannelEstimation
)
from simulation import Simulation

accessCode = [1,0,1,0,1,0,1,0]
lenAC = 4
degree = 2
m = 20
n = m  # number of users
noIter = 100

pktSize = 32   # in bits (4B)
LOAD = np.linspace(0.125, 1, 8)
SNR_dB = np.arange(-12, 41, 4)
signal_power = 1 

base = BPSKBase()
chEst = ChannelEstimation()

thr_load = {}; per_load = {}; ber_load = {}
for load in LOAD:
    throughput = {}; per = {}; ber = {}
    for snr in SNR_dB:
        PER, BER, THROUGHPUT = 0, 0, 0
        noise_var = signal_power * 10**(-snr/10)
        ch = SlowFadingChannel(noise_var)
        sim = Simulation(base, ch, chEst, m, n, degree, pktSize, accessCode[:lenAC])
        userSlotsGen = sim.userSlots
        for i in range(noIter):
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

            pcr, bcr_frame = sim.per(pkt_hat)
            PER += ( 1 - (pcr/len(activeUsers)) )
            BER += ( 1 - ( bcr_frame / ( pktSize * len(activeUsers) ) ) )
            THROUGHPUT += bcr_frame / ( pktSize * len(activeUsers) )
        per[snr] = PER / noIter
        ber[snr] = (BER / noIter).astype(float)
        throughput[snr] = THROUGHPUT / noIter
    print(f"Load - {load} done")
    thr_load[load] = throughput
    per_load[load] = per
    ber_load[load] = ber

plt.figure(figsize=(8,6), dpi=400)
for k, v in thr_load.items():
    plt.plot(v.keys(), v.values(), linewidth=0.9, linestyle='-', label=f"Load - {k}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("SNR(dB)")
plt.ylabel("Throughput (T)")
plt.ylim(0, 1.05)
plt.title(f"Throughput vs SNR over varied load for {noIter} Iterations")
plt.legend(loc='lower left', fontsize=3, framealpha=0.6)
plt.tight_layout()
plt.savefig("results/ConSim/dloadSNR.jpeg")
# plt.show()