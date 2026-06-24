"""
Throughput analysis of BMOCZ + IRSA for fixed SNR over varying LOAD.
System parameters:
- Total number of users(n) = 20
- Number of slots per frame(m) = 20
- Channel: Rayleigh Block-fading 
- Payload (packet size): 32 bits
- Slot distribution: CRDSA (x**2)
We will also be genrating a BAPM (Binary Access Pattern Matrix) which defines the slots 
in which users will be transmitting.
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

degree = 2  # CRDSA 
m = 20
n = m
noIter = 100  # gives the simulation over 1000 frames
G = np.linspace(0.125, 1, 8)

K = 32   # 4B
SNR_dB = np.arange(10, 41, 5)
signal_power = 1

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
chEst = ChannelEstimation()

thr_snr = {}; per_snr = {}; ber_snr = {}
for snr in SNR_dB:
    noise_var = signal_power * 10**(-snr/10)
    ch = SlowFadingChannel(var=noise_var)
    sim = Simulation(tx, rx, ch, chEst, m, n, degree, K, Q=4)
    userSlotsGen = sim.userSlots
    throughput = {}; per = {}; ber = {}
    for g in G:
        PER, BER, THROUGHPUT = 0, 0, 0
        for i in range(noIter):
            FRAME = {}
            slot = set()
            activeUsers = sorted( random.sample(range(1, n+1), int(g*m)) )
            for userId in activeUsers:
                userSlot = userSlotsGen[userId]
                for s in userSlot:
                    if s not in slot:
                        FRAME[s] = [userId]
                        slot.add(s)
                    else:
                        FRAME[s] += [userId] 
            FRAME = dict(sorted(FRAME.items(), reverse=False))
            frame, h = sim.frameBuild(FRAME)
            frameBAPM = sim.genBAPM(activeUsers)
            msg_hat, h_hat = sim.frameParse(frame, frameBAPM)
            # print(msg_hat)
            pcr, bcr_frame = sim.per(msg_hat)
            PER += ( 1 - (pcr/(len(activeUsers))) )
            BER += ( 1 - ( bcr_frame / ( K * len(activeUsers) ) ) )
            THROUGHPUT += bcr_frame / ( K * len(activeUsers) )
        per[g] = PER / noIter
        ber[g] = (BER / noIter).astype(float)
        throughput[g] = THROUGHPUT / noIter
    print(f"SNR - {snr} done")
    thr_snr[snr] = throughput
    per_snr[snr] = per
    ber_snr[snr] = ber


# plt.figure(1, dpi=400)
# plt.plot(per.keys(), per.values(), '-')
# plt.grid(True)
# plt.xlabel("Load(g)")
# plt.ylabel("Packet Error Rate")
# plt.ylim((0,1))
# plt.title(f"PER vs Load over {noIter} Iterations at SNR {snr}")
# plt.savefig("results/ConSim/Mper.jpeg")

# plt.figure(2, dpi=400)
# plt.plot(ber.keys(), ber.values(), '-')
# plt.grid(True)
# plt.xlabel("Load(g)")
# plt.ylabel("Bit Error Rate(BER)")
# plt.ylim((0,1))
# plt.title(f"BER vs Load over {noIter} Iterations at SNR {snr}")
# plt.savefig("results/ConSim/Mber.jpeg")

# # rate = 1 bit / sec
# plt.figure(3, dpi=400)
# plt.plot(throughput.keys(), throughput.values(), '-')
# plt.grid(True)
# plt.xlabel("Load(g)")
# plt.ylabel("Throughput(T)")
# plt.ylim((0,1))
# plt.title(f"Throughput vs Load over {noIter} Iterations at SNR {snr}")
# plt.savefig("results/ConSim/Mthr.jpeg")

plt.figure(figsize=(8,6), dpi=400)
# markers = ['o', 's', '^', 'D', 'v', 'p', '*']
for i, (k, v) in enumerate(thr_snr.items()):
    # marker = markers[i % len(markers)]
    plt.plot(list(v.keys()), list(v.values()), 
            linestyle='-', linewidth=0.9,
            label=f"SNR = {k}dB"
            ) # marker=marker,  markersize=6,
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Load(g)", fontsize=7, fontweight='bold')
plt.ylabel("Throughput(T)", fontsize=7) # , fontweight='bold'
plt.ylim(0, 1.05)
plt.title(f"Throughput vs Load over {noIter} Iterations", fontsize=7, pad = 4)
plt.legend(loc='upper right', fontsize=4, framealpha=0.9)
plt.tight_layout()
plt.savefig("results/ConSim/thrLoad.jpeg")
# plt.show()