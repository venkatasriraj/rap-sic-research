"""
Packet Error Analysis for BMOCZ in FS channel under AWGN and zero rotation
"""
import numpy as np
import os
from wirelessComm import (
    JBMOCZ, SlowFadingChannel, PerformanceParameters, 
    ACPC, BMOCZ, JUTTEDfs, BCH
)
save_dir = 'results/ErrorAnalysis/Simulation'
os.makedirs(save_dir, exist_ok=True)

FLAG = 'FS_CFO'
# Lh = np.concatenate(([1, 2, 3, 4, 5, 6], np.arange(10, 51, 5)), axis=None)
Lh = [1, 4, 6]
chVar, Q = 0.86, 32
m, t, Ns = 5, 2, 256
K = 31
Peak = [0, 7, 17, 23]
zeta, Rj, noIter = 1.15, 1.044, int(1e4)
SNR_dB = np.arange(0, 31, 5)
perParam = PerformanceParameters()

schemes = ['BMOCZ-ACPC', 'J-BMOCZ', 'MJ-BMOCZ', 'DiZeT-CFO', 'DiZeT-CFO-BCH', 'BMOCZ']
scheme = 'DiZeT-CFO-BCH'

if scheme == 'BMOCZ-ACPC':
    system = BMOCZ(K, PZrad=0)
    acpc = ACPC(m, t)
elif scheme == 'BMOCZ':
    system = BMOCZ(K, PZrad=0) 
elif scheme == 'MJ-BMOCZ':
    system = JBMOCZ(K, zeta, Ns=Ns, Rj=Rj, mPeak=Peak)
elif scheme == 'J-BMOCZ':
    system = JBMOCZ(K, zeta, Ns=Ns, Rj=Rj)
elif scheme == 'DiZeT-CFO':
    system = JUTTEDfs(K, zeta, Rj=Rj, multiplePeak=False)
elif scheme == 'DiZeT-CFO-BCH':
    system = JUTTEDfs(K, zeta, Rj=Rj, multiplePeak=False)
    bch = BCH(m, t)

for l in Lh:
    berSNR, perSNR, errorDistSNR, errorPositionSNR = [], [], [], []
    Energy = K + l
    for snr in SNR_dB:
        errorDist, errorPosition = np.zeros(K+2, dtype=int), np.zeros(K+1, dtype=int)
        noiseVar = 10**(-snr/10)
        ch = SlowFadingChannel(noise_var=noiseVar, taps=l, chVar=chVar)

        if scheme == 'BMOCZ-ACPC':
            results = system.simulatorACPC(noIter, perParam, ch, acpc, Energy, Q, FLAG, errorAnalysis=True)
        elif scheme == 'BMOCZ':
            results = system.simulator(noIter, perParam, ch, Energy, Q, FLAG=FLAG, errorAnalysis=True)
        elif scheme == 'MJ-BMOCZ':
            results = system.simulator(noIter, perParam, ch, Energy, FLAG=FLAG, errorAnalysis=True)
        elif scheme == 'J-BMOCZ':
            results = system.simulator(noIter, perParam, ch, Energy, FLAG=FLAG, errorAnalysis=True)
        elif scheme == 'DiZeT-CFO':
            results = system.simulator(noIter, perParam, ch, Energy, Q, FLAG, errorAnalysis=True)
        elif scheme == 'DiZeT-CFO-BCH':
            results = system.simulatorBCH(noIter, perParam, ch, bch, Energy, Q, FLAG, errorAnalysis=True)
        
        berSNR.append(results['ber']/noIter)
        perSNR.append(1 - (results['pcr']/noIter))
        errorDistSNR.append(results['errorDist'])
        errorPositionSNR.append(results['errorPosition'])
        # print(results['errorDist'])
    print(f"Lh - {l} done")
    np.savez_compressed(
        os.path.join(save_dir, f"{scheme}_Lh{l}_{FLAG}.npz"),

        SNR_dB = SNR_dB,
        BER = np.asarray(berSNR),
        PER = np.asarray(perSNR),
        ErrorDist = np.asarray(errorDistSNR),
        ErrorPosition = np.asarray(errorPositionSNR),

        K=K,
        Lh = l,
        chVar = chVar,
        noIter = noIter
    )

# for _ in range(noIter):
#     msgTx =  np.random.randint(0, 2, K)
#     sigTx = bmoczSystem.coeffCon(msgTx)
#     l2_norm = np.linalg.norm(sigTx, ord=2) 
#     sigTx = (sigTx / l2_norm) * np.sqrt(Energy)      

#     sigRx, _ = ch.frequencySelective(sigTx, noiseFlag=True)
    
#     msgRx = bmoczSystem.fftDizet(sigRx, Q)

#     BER += perParam.ber(msgRx, msgTx)
#     PCR += perParam.pcr(msgRx, msgTx)
#     errorDist[perParam.bitError(msgRx, msgTx)] += 1
#     errorPosition[perParam.errorPos(msgRx, msgTx)] += 1