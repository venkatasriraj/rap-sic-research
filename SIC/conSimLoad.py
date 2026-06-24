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
import matplotlib.pyplot as plt
from BPSK import BPSKBase
from CHANNEL import (
    SlowFadingChannel,
    ChannelEstimation
)
from simulation import Simulation

accessCode = [1, 0] * 4
lenAc = 4
degree = 2 
m = 20; n = m
noIter = 1

pktSize = 32
SNR_dB = np.arange(-12, 41, 4)
LOAD = np.linspace(0.125, 1, 8)
signal_power = 1

base = BPSKBase()
chEst = ChannelEstimation()

thr_snr = {}; per_snr = {}; ber_snr = {}
for snr in SNR_dB:
    throughput = {}; per = {}; ber = {}
    noise_var = signal_power * 10**(-snr/10)
    ch = SlowFadingChannel(noise_var)
    sim = Simulation(base, ch, chEst, m, n, degree, pktSize, accessCode)
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
                        slot.add(userId)
                    else:
                        FRAME[s] += [userId]
            FRAME = dict( sorted( FRAME.items(), reverse=False ) )
            frame, h = sim.frameBuild(FRAME),
            frameBAPM = sim.genBAPM(activeUsers)

            pkt_hat, h_hat = sim.frameParse(frame, frameBAPM)

            pcr, bcr_frame = sim.per(pkt_hat)
            PER += ( 1 - (pcr/(len(activeUsers))) )
            BER += ( 1 - ( bcr_frame / (pktSize * len(activeUsers)) ) )
            THROUGHPUT += bcr_frame / (pktSize * len(activeUsers))
        throughput[load] = THROUGHPUT / noIter
        ber[load] = (BER / noIter).astype(float)
        per[load] = PER / noIter
    print(f"SNR - {snr} done")
    thr_snr[snr] = throughput
    per_snr[snr] = per
    ber_snr[snr] = ber

