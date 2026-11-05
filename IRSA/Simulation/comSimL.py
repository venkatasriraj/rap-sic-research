"""
Combined plots for BMOCZ and DBPSK with and without pilot
with baseline for system performance without SIC over varied Load
n: number of users
m: number of slots

SIMULATOR arguments:
( system, ch, chEst, slots=20, users=20, degree=2, pktSize=31, Q=32, 
maxIter='None', energy='None', scheme='DPSK', chMode='BF', cfo=False)
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
LOAD = np.arange(0.1, peakLoad+0.1, 0.1)
SNR_dB, noIter = 15, int(1e1)
packetSize, uId = 31, 1  # K = packetSize
Lh, chVar = 1, 0.86
# BMOCZ Parameters
Q = 32
# DBPSK Parameters
accessCode = [1, 0] * 4
lenAC = 8
ch = SlowFadingChannel(noise_var=10**(-SNR_dB/10), taps=Lh, chVar=chVar)
bmoczSystem = BMOCZ(packetSize, PZrad=0, Q=2)
dpsk = DPSK()
chEst = ChannelEstimation()

schemes = ['BMOCZ', 'DPSK']
scheme = 'BMOCZ'
modes = ['SIC', 'noSIC', 'CSI']
cfoModes = ['CFO', 'noCFO']
cfo = 'noCFO'
cfoMode = True if cfo == 'CFO' else False 

if scheme == 'BMOCZ':
    sim = SIMULATOR(bmoczSystem, ch, chEst, m, n, degree, packetSize, Q, scheme=scheme, cfo=cfoMode)
elif scheme == 'DPSK':
    sim = SIMULATOR(dpsk, ch, chEst, m, n, degree, packetSize, Q, scheme=scheme, cfo=cfoMode)
    
for mode in modes:
    throughput, per, ber, maeH = [], [], [], []
    for load in LOAD:
        if mode == 'SIC':
            PER, BER, THROUGHPUT, MAE, MAE_count = sim.simulator(load, noIter, sicMode=mode, uId=uId)
            maeH.append( MAE / MAE_count )
        else:
            PER, BER, THROUGHPUT = sim.simulator(load, noIter, sicMode=mode, uId=uId)   
        throughput.append(THROUGHPUT/noIter)
        per.append(PER/noIter)
        ber.append(BER/noIter)
    print(f"mode {mode} done")
    np.savez_compressed(

        os.path.join(save_dir, f'{scheme}_{mode}Load_{cfo}{Lh}.npz'),

        MODE = mode,
        xlabelVal = LOAD,
        xlabel = 'LOAD',

        THROUGHPUT = throughput,
        BER = ber,
        PER = per,
        MAEH = maeH,

        pktSize = packetSize,
        Q = Q,
        lenAC = lenAC,
        accessCode = accessCode,
        users = n,
        slots = m,
        degree = degree,
        noIter = noIter,
        uId = uId,
        snr = SNR_dB
    )