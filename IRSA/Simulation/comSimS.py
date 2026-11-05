"""
Combined plots for BMOCZ and DBPSK with and without pilot
with baseline for system performance without SIC over varied Load
n: number of users
m: number of slots
"""
import numpy as np
import os
from wirelessComm import (
    BMOCZ, DPSK, SlowFadingChannel, ChannelEstimation, SIMULATOR    
)
m, n = 20, 20
save_dir = f'results/simulation/U{m}_S{n}'
os.makedirs(save_dir, exist_ok=True)
degree = 2 # SLOT-DISTRIBUTION
peakLoad = n/m
SNR_dB = np.arange(-5, 41, 5)
LOAD, noIter = 0.65, int(1e4)
packetSize, uId = 31, 1  # K = packetSize
Q = 32
bmoczSystem = BMOCZ(packetSize, PZrad=0, Q=2)
dpsk = DPSK()
chEst = ChannelEstimation()
schemes = ['BMOCZ', 'DPSK']
scheme = 'DPSK'
modes = ['SIC', 'noSIC', 'CSI']

for mode in modes:
    throughput, per, ber, maeH = [], [], [], []
    for snr in SNR_dB:
        noiseVar = 10**(-snr/10)
        ch = SlowFadingChannel(noiseVar)
        if scheme == 'BMOCZ':
            sim = SIMULATOR(bmoczSystem, ch, chEst, m, n, degree, packetSize, Q, scheme=scheme)
        elif scheme == 'DPSK':
            sim = SIMULATOR(dpsk, ch, chEst, m, n, degree, packetSize, Q, scheme=scheme)
    
        if mode == 'SIC':
            PER, BER, THROUGHPUT, MAE, MAE_count = sim.simulator(LOAD, noIter, sicMode=mode, uId=uId)
            maeH.append( MAE / MAE_count )
        else:
            PER, BER, THROUGHPUT = sim.simulator(LOAD, noIter, sicMode=mode, uId=uId)   
        throughput.append(THROUGHPUT/noIter)
        per.append(PER/noIter)
        ber.append(BER/noIter)
    print(f"mode {mode} done")
    np.savez_compressed(

        os.path.join(save_dir, f'{scheme}_{mode}snr.npz'),

        MODE = mode,
        xlabelVal = SNR_dB,
        xlabel = 'SNR (dB)',

        THROUGHPUT = throughput,
        BER = ber,
        PER = per,
        MAEH = maeH,

        pktSize = packetSize,
        Q = Q,
        users = n,
        slots = m,
        degree = degree,
        noIter = noIter,
        uId = uId,
        load = LOAD
    )