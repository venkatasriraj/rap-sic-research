"""
Implementation of OFDM with PSK modulation in frequency selective channels
maxDelay (D) = cpLen-1; practical applications have cpLen = 4 * D

"""
import numpy as np
import math
import matplotlib.pyplot as plt
from wirelessComm import OFDM, PSK, SlowFadingChannel, PerformanceParameters

delay = [0, 3, 6]
N, M, cpLen = 64, 2, max(delay)+1
pilotSpacing = N // (max(delay) + 1) 
chVar, Lh, noIter = 0.63, 1, int(1e4)
SNR_dB = np.arange(-5, 32, 3)
power = chVar**np.arange(len(delay))
perParam = PerformanceParameters()
ofdm = OFDM(N, cpLen, pilotSpacing)
bpsk = PSK(M)
ch = SlowFadingChannel(noise_var=0.01, chVar=chVar, taps=Lh)
msgBits = int( math.log2(M) * (N - np.ceil(N / pilotSpacing)) ) 
# print(msgBits)
# msgTx = np.random.randint(0, 2, msgBits)
# msgTx = [1, 0, 1, 1]
# symbols = bpsk.bpskModulation(msgTx)
# sigTx = ofdm.cohTxModule(symbols)
# print(f"MsgTx: {msgTx}\nBPSK: {symbols}\nOFDM Sym: {sigTx}\n")

# sigRx, h = ch.frequencySelective(sigTx, noiseFlag=False)
# sigRx = ch.PDP_LTI(delay, power, sigTx, noiseFlag=True)
# print(f"channel: {h}, {len(sigRx)}\n")

# symbolsHat = ofdm.cohRxModule(sigRx)
# msgRx = bpsk.bpskDemodulation(symbolsHat)
# ber = perParam.ber(msgRx, msgTx)
# print(f"OFDM Sym: {sigRx}\nBPSK: {symbolsHat}\nMsgRx: {msgRx}")

# print(f"BER: {ber}")

ber, per = {}, {}
for snr in SNR_dB:
    ch = SlowFadingChannel(noise_var=10**(-snr/10), chVar=chVar, taps=Lh, delay=delay, power=power)
    BER, PCR = 0, 0
    for _ in range(noIter):
        msgTx = np.random.randint(0, 2, msgBits)
        symbols = bpsk.bpskModulation(msgTx)
        sigTx = ofdm.cohTxModule(symbols)

        # sigRx, _ = ch.frequencySelective(sigTx, noiseFlag=False)
        sigRx = ch.PDP_LTI(sigTx, noiseFlag=True)
        # sigRx = ch.awgn(sigTx)

        symbolsHat = ofdm.cohRxModule(sigRx)
        msgRx = bpsk.bpskDemodulation(symbolsHat)

        BER += perParam.ber(msgRx, msgTx)
        PCR += perParam.pcr(msgRx, msgTx)
    ber[snr] = BER / noIter
    per[snr] = 1 - PCR / noIter

plt.figure(1, dpi=800)
plt.semilogy(ber.keys(), ber.values(), linewidth=0.9)
plt.ylim(1e-4, 1)
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("BER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"OFDM BER Analysis chVar-{chVar}, N{N}")
plt.savefig(f"results/coh/ber.jpeg")

plt.figure(2, dpi=800)
plt.semilogy(per.keys(), per.values(), linewidth=0.9)
plt.ylim(1e-4, 1)
plt.xlabel(f"SNR(dB): {noIter} Runs per point")
plt.ylabel("PER")
plt.grid(True, alpha=0.6, linestyle='--')
plt.title(f"OFDM PER Analysis chVar-{chVar}, N{N}")
plt.savefig(f"results/coh/per.jpeg")