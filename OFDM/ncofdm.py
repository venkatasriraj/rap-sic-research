"""
Implementation of Noncoherent OFDM-IM
N(subcarriers per cluster) = 16, G(clusters) = 2, 
Q(active subcarriers per cluster) = 8
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import NoncoherentOFDMIM, PerformanceParameters, SlowFadingChannel

delay = [0, 3]
N, G, Q = 8, 1, 4
chVar, Lh, noIter = 0.95, 2, int(1e2)
cpLen = Lh + 1 #max(delay) +1
power = chVar**np.arange(len(delay))
SNR_dB = np.arange(-5, 32, 3)
ncOFDMim = NoncoherentOFDMIM(N, cpLen, G, Q, threshold=0.25, A=2)
perParam = PerformanceParameters()
bitsEnc = ncOFDMim.bits_cluster * G
ch = SlowFadingChannel(noise_var=0.01, chVar=chVar, taps=Lh, power=power, delay=delay)

msgTx = np.random.randint(0, 2, bitsEnc)
sigTx = ncOFDMim.transmitModule(msgTx)

sigRx, _ = ch.frequencySelective(sigTx, noiseFlag=False)
# sigRx = ch.PDP_LTI(sigTx, noiseFlag=False)

msgRx = ncOFDMim.receiveModule(sigRx)
print(f"Tx: {msgTx}\nRx: {msgRx}")
print(f"BER: {perParam.ber(msgRx, msgTx)}; bitsPerCluster: {bitsEnc}")

# ber, per = {}, {}
# for snr in SNR_dB:
#     ch = SlowFadingChannel(noise_var=10**(-snr/10), chVar=chVar, taps=Lh)
#     BER, PCR = 0, 0
#     for _ in range(noIter):
#         msgTx = np.random.randint(0, 2, bitsEnc)
#         sigTx = ncOFDMim.transmitModule(msgTx)

#         sigRx, _ = ch.frequencySelective(sigTx, noiseFlag=False)
#         # sigRx = ch.awgn(sigTx)
#         msgRx = ncOFDMim.receiveModule(sigRx)
#         BER += perParam.ber(msgRx, msgTx)
#         PCR += perParam.pcr(msgRx, msgTx)
#     ber[snr] = BER / noIter
#     per[snr] = 1 - PCR / noIter

# plt.figure(1, dpi=800)
# plt.semilogy(ber.keys(), ber.values(), linewidth=0.9)
# plt.ylim(1e-4, 1)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
# plt.ylabel("BER")
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"Noncoherent OFDM-IM BER Analysis chVar-{chVar} G{G}, Q{Q}, N{N}")
# plt.savefig(f"results/nonCoh/ber.jpeg")

# plt.figure(2, dpi=800)
# plt.semilogy(per.keys(), per.values(), linewidth=0.9)
# plt.ylim(1e-4, 1)
# plt.xlabel(f"SNR(dB): {noIter} Runs per point")
# plt.ylabel("PER")
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.title(f"Noncoherent OFDM-IM PER Analysis chVar-{chVar} G{G}, Q{Q}, N{N}")
# plt.savefig(f"results/nonCoh/per.jpeg")