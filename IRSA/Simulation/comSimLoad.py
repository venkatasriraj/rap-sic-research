"""
Combined plots for BMOCZ and DBPSK with and without pilot
with baseline for system performance without SIC over varied Load
n: number of users
m: number of slots
"""
import numpy as np
import os
from wirelessComm import (
    BMOCZ, moczSIMULATION, DPSK, dbpskSIMULATION, PerformanceParameters,
    SlowFadingChannel, ChannelEstimation, Simulator    
)
m, n = 20, 20
save_dir = f'results/simulation/U{m}_S{n}'
os.makedirs(save_dir, exist_ok=True)
degree = 2 # SLOT-DISTRIBUTION
peakLoad = n/m
LOAD = np.arange(0.1, peakLoad+0.1, 0.1)
SNR_dB, noIter = 15, int(1e4)
packetSize, uId = 31, 1  # K = packetSize
# BMOCZ Parameters
Q = 2
# DBPSK Parameters
accessCode = [1, 0] * 4
lenAC = 8
ch = SlowFadingChannel(noise_var=10**(-SNR_dB/10))
bmoczSystem = BMOCZ(packetSize, PZrad=0)
dpsk = DPSK()
chEst = ChannelEstimation()
perParam = PerformanceParameters()

labels = ["MOCZ", "MOCZ-noSIC", "MOCZ-CSI", "DBPSK-pilot", "DBPSK-pilot-noSIC", "DBPSK-pilot-CSI", "DBPSK-decDir", "DBPSK-decDir-noSIC", "DBPSK-decDir-CSI"]
scheme = 'DPSK'
modes = ['SIC', 'noSIC', 'CSI']

for mode in modes:
    throughput, per, ber, maeH = [], [], [], []
    for load in LOAD:
        seedNo = abs(int(SNR_dB * n * 3 + load))
        if scheme == 'BMOCZ':
            sim = moczSIMULATION(bmoczSystem, ch, chEst, m, n, degree, packetSize, Q, seedNo)
        elif scheme == 'DPSK-Pilot':
            pilot = accessCode
            sim = dbpskSIMULATION(dpsk, ch, chEst, m, n, degree, packetSize, pilot, seedNo)
        elif scheme == 'DPSK':
            pilot = []
            sim = dbpskSIMULATION(dpsk, ch, chEst, m, n, degree, packetSize, pilot, seedNo)

        if mode == 'SIC':
            PER, BER, THROUGHPUT, MAE, MAE_count = Simulator().simulator(sim, load, noIter, sicMode=mode, uId=uId)
            maeH.append( MAE / MAE_count )
        else:
            PER, BER, THROUGHPUT = Simulator().simulator(sim, load, noIter, sicMode=mode, uId=uId)   
        throughput.append(THROUGHPUT/noIter)
        per.append(PER/noIter)
        ber.append(BER/noIter)
    print(f"mode {mode} done")
    np.savez_compressed(

        os.path.join(save_dir, f'{scheme}_{mode}Load.npz'),

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

# throughput_load = {}; per_load = {}; ber_load = {}; maeh_load = {}
# for mode in range(len(labels)):
#     sicMode = "normal" if mode % 3 == 0 else ( "noSIC" if mode % 3 == 1 else "CSI" )
#     throughput = {}; per = {}; ber = {}; maeh = {}
#     for load in LOAD:
#         seedNo = abs(int(SNR_dB * n * 3 + load))
#         if mode // 3 == 2: # no-pilot DBPSK
#             pilot = []
#             sim = dbpskSIMULATION(dpsk, ch, chEst, m, n, degree, packetSize, pilot, seedNo)
#         elif mode // 3 == 1: # 8-bit pilot
#             pilot = accessCode
#             sim = dbpskSIMULATION(dpsk, ch, chEst, m, n, degree, packetSize, pilot, seedNo)
#         else: # MOCZ
#             sim = moczSIMULATION(bmoczSystem, ch, chEst, m, n, degree, packetSize, Q, seedNo)

#         if sicMode == "normal":
#             PER, BER, THROUGHPUT, MAE, MAE_count = Simulator().simulator(sim, load, noIter, sicMode=sicMode, uId=uId)
#             maeh[load] = MAE / MAE_count
#         else:
#             PER, BER, THROUGHPUT = Simulator().simulator(sim, load, noIter, sicMode=sicMode, uId=uId)   
#         per[load] = PER / noIter
#         ber[load] = (BER / noIter).astype(float)
#         throughput[load] = THROUGHPUT / noIter
#     print(f"MODE: {labels[mode]} done")
#     throughput_load[mode] = throughput
#     per_load[mode] = per
#     ber_load[mode] = ber
#     if sicMode == "normal":
#         maeh_load[mode] = maeh

# plt.figure(figsize=(8,6), dpi=800)
# for i, (k, v) in enumerate(throughput_load.items()):
#     plt.plot(v.keys(), v.values(), color=colors[i//3], linestyle=linestyles[i%3],
#                 marker=markers[i%3], linewidth=1.2, markersize=0, label=labels[i])
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xlabel("Channel Load")
# plt.ylabel("Throughput")
# plt.ylim(0, 1.05)
# plt.title(f"{noIter} frames per point for SNR = {SNR_dB}dB")
# plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig(f"results/comPlot/thrLoad{SNR_dB}.jpeg")

# plt.figure(figsize=(8,6), dpi=800)
# for i, (k,v) in enumerate(per_load.items()):
#     plt.plot(v.keys(), v.values(), color=colors[i//3], linestyle=linestyles[i%3],
#                 marker=markers[i%3], markersize=0, linewidth=1.2, label=labels[i])
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xlabel("Channel Load")
# plt.ylabel("PER")
# plt.ylim(0, 1.05)
# plt.title(f"{noIter} frames per point at SNR = {SNR_dB}dB")
# plt.legend(loc="upper left", framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig(f"results/comPlot/perLoad{SNR_dB}.jpeg")

# plt.figure(figsize=(8,6), dpi=800)
# for i, (k, v) in enumerate(ber_load.items()):
#     plt.plot(v.keys(), v.values(), color=colors[i//3], linestyle=linestyles[i%3],
#                 marker=markers[i%3], markersize=0, linewidth=1.2, label=labels[i])
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.xlabel("Channel Load")
# plt.ylabel("BER")
# plt.ylim(0, 1.05)
# plt.title(f"{noIter} frames per point at SNR = {SNR_dB}dB")
# plt.legend(loc='upper left', framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig(f"results/comPlot/berLoad{SNR_dB}.jpeg")

# plt.figure(figsize=(8,6), dpi=800)
# for i, (k,v) in enumerate(maeh_load.items()):
#     plt.plot(v.keys(), v.values(), color=colors[i], linestyle=linestyles[i],
#                 marker=markers[i], markersize=0, linewidth=1.2, label=labels[3*i])
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.xlabel("Channel Load")
# plt.ylim(0, 1.05)
# plt.ylabel(f"User-{uId} Normalized MAE of h")
# plt.title(f"{noIter} frames per point at SNR = {SNR_dB}dB")
# plt.legend(loc="upper right", framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig(f"results/comPlot/maehLoad{SNR_dB}.jpeg")