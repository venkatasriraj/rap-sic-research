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
noIter = 10  # gives the simulation over 1000 frames
G = np.linspace(0.1, 1, 10)

K = 32   # 4B
SNR_dB = np.arange(-10, 20, 5)
signal_power = 1
uId = 1
pathLoss = 1

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
chEst = ChannelEstimation()

thr_snr = {}; per_snr = {}; ber_snr = {}; mae_hEst = {}
for snr in SNR_dB:
    noise_var = signal_power * 10**(-snr/10)
    ch = SlowFadingChannel(noise_var)
    throughput = {}; per = {}; ber = {}; mae_herr = {}
    for g in G:
        PER, BER, THROUGHPUT, MAE, MAE_count = 0, 0, 0, 0, 1e-10
        seedNo = abs(int(g*n*3 + snr) )
        sim = Simulation(tx, rx, ch, chEst, m, n, degree, K, Q=4, seed=seedNo)
        for i in range(noIter):
            userSlotsGen = sim.userSlotGen()
            FRAME = {}
            slot = set()
            activeUsers = sorted( sim.rng.sample(range(1, n+1), int(g*m)) )
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
            frameBAPM = sim.genBAPM(activeUsers, userSlotsGen)
            msg_hat, h_hat = sim.frameParse(frame, frameBAPM, userSlotsGen)
            if uId in activeUsers:
                mae_temp, count = sim.maeh(h, h_hat, uId)
                MAE += mae_temp
                MAE_count += count
            pcr, bcr_frame = sim.per(msg_hat)
            PER += ( 1 - (pcr/(len(activeUsers))) )
            BER += ( 1 - ( bcr_frame / ( K * len(activeUsers) ) ) )
            THROUGHPUT += pcr / len(activeUsers) 
        per[g] = PER / noIter
        ber[g] = (BER / noIter).astype(float)
        throughput[g] = THROUGHPUT / noIter
        # print(f"load is {g} - MAE: {MAE}; MAE Count: {MAE_count}; Avg {MAE / MAE_count}, Active Users: {activeUsers}")
        mae_herr[g] = MAE / MAE_count
    print(f"SNR - {snr} done")
    thr_snr[snr] = throughput
    per_snr[snr] = per
    ber_snr[snr] = ber
    mae_hEst[snr] = mae_herr

plt.figure(figsize=(8,6), dpi=800)
# markers = ['o', 's', '^', 'D', 'v', 'p', '*']
for i, (k, v) in enumerate(mae_hEst.items()):
    # marker = markers[i % len(markers)]
    plt.plot(list(v.keys()), list(v.values()), 
            linestyle='-', linewidth=0.9,
            label=f"SNR = {k}dB"
            ) # marker=marker,  markersize=6,
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel("Load(g)", fontsize=7, fontweight='bold')
plt.ylabel(f"MAE of h_est for user-{uId}", fontsize=7) # , fontweight='bold'
plt.title(f"MAE of h vs Load over {noIter} Iterations", fontsize=7, pad = 4)
plt.legend(loc='upper left', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig("results/ConSim/mhLoad.jpeg")

plt.figure(figsize=(8,6), dpi=800)
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
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.savefig("results/ConSim/mthrLoad.jpeg")