"""
SNR = Eb * Rb / ( N0 * BW )
Rb = K * Rs = K / Ts = K * BW / (K+1)
"""
import numpy as np
import os 
from wirelessComm import (
    JBMOCZ, SlowFadingChannel, PerformanceParameters, 
    ACPC, BMOCZ, JUTTEDfs, BCH
)

save_dir = 'results/conSim/heuristicsJ/simulation'
os.makedirs(save_dir, exist_ok=True)

FLAG = "FS_CFO"
m, t, Q, Ns = 5, 2, 32, 256
EbN0_dB = np.arange(-1, 42, 2)
Lh, chVar = 1, 0.86
zeta, Rj, noIter = 1.15, 1.044, int(8e4)
Peak = [0, 7, 17, 23]
schemes = ['BMOCZ-ACPC', 'J-BMOCZ', 'MJ-BMOCZ', 'DiZeT-CFO', 'DiZeT-CFO-BCH']
# schemes = ['BMOCZ-ACPC']
K = 2**m-1
ENERGY, mode = K + Lh, 1
bmoczSys = BMOCZ(K, PZrad=0)
perParam = PerformanceParameters()
jbmcozSystem = JBMOCZ(K, zeta, Ns=Ns, Rj=Rj, mPeak=Peak)
jSystem = JUTTEDfs(K, zeta, Rj=Rj, multiplePeak=False)
juttedRef = JBMOCZ(K, zeta, Ns=Ns, Rj=Rj)
acpc = ACPC(m, t)
bch = BCH(m, t)

for scheme in schemes:
    berSNR, perSNR, paprSNR, goodputSNR, rotationSNR = [], [], [], [], []
    for ebn0 in EbN0_dB:
        if scheme == 'BMOCZ-ACPC':
            infoBits = K - m*(t+1)
        elif scheme == 'DiZeT-CFO-BCH':
            infoBits = K - m*t
        else:
            infoBits = K
        Rb_BW = infoBits / ENERGY
        if mode == 0:
            snr = ebn0
        else:
            snr = ebn0 + 10 * np.log10(Rb_BW)
        # print(snr)
        noiseVar = 10**(-snr/10)
        ch = SlowFadingChannel(noise_var=noiseVar, taps=Lh, chVar=chVar)
        if scheme == 'BMOCZ-ACPC':
            results = bmoczSys.simulatorACPC(noIter, perParam, ch, acpc, Energy=ENERGY, Q=Q, FLAG=FLAG)
            # goodputSNR[snr] = results['pcr'] * (K - 5*(t+1)) / K # * pktRate
            gp = results['pcr'] * Rb_BW
        elif scheme == 'MJ-BMOCZ':
            results = jbmcozSystem.simulator(noIter, perParam, ch, Energy=ENERGY, FLAG=FLAG)
            # goodputSNR[snr] = results['pcr'] * K / K # * pktRate
            gp = results['pcr'] * Rb_BW
        elif scheme == 'DiZeT-CFO':
            results = jSystem.simulator(noIter, perParam, ch, Energy=ENERGY, Q=Q, FLAG=FLAG)
            # goodputSNR[snr] = results['pcr'] * K / K # * pktRate
            gp = results['pcr'] * Rb_BW
        elif scheme == 'J-BMOCZ':
            results = juttedRef.simulator(noIter, perParam, ch, Energy=ENERGY, FLAG=FLAG)
            # goodputSNR[snr] = results['pcr'] * K / K # * pktRate
            gp = results['pcr'] * Rb_BW
        elif scheme == 'DiZeT-CFO-BCH':
            results = jSystem.simulatorBCH(noIter, perParam, ch, bch, Energy=ENERGY, Q=Q, FLAG=FLAG)
            gp = results['pcr'] * Rb_BW
        berSNR.append(results['ber'])
        perSNR.append(1 - results['pcr'])
        paprSNR.append(results['papr'])
        rotationSNR.append(results['rotationEst'])
        goodputSNR.append(gp)
    print(f"{scheme} Done")
    # Save individual scheme results
    np.savez_compressed(
        os.path.join(save_dir, f"{scheme}_Lh{Lh}_{FLAG}.npz"),
        
        SNR_dB=EbN0_dB,
        mode = mode, 

        BER=np.asarray(berSNR),
        PER=np.asarray(perSNR),
        PAPR=np.asarray(paprSNR),
        Goodput=np.asarray(goodputSNR),
        Rotation=np.asarray(rotationSNR),

        # Simulation configuration
        K=K,
        Lh=Lh,
        chVar=chVar,
        zeta=zeta,
        Rj=Rj,
        m=m,
        t=t,
        noIter=noIter
    )