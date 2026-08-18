"""
channel model: frequency selective channel
channel power(variance) = 1
"""
import numpy as np
import matplotlib.pyplot as plt
from lora import LoRa
from wirelessComm import (
    MultiPathFading, PerformanceParameters
)
SF = np.arange(7, 16, 1)
SNR_dB = np.arange(-20, 21, 5)
noIter = int(1e1)
taps = 2; chVar = 1
BW = 125e3
initialFreq = -0.5 * BW
sampRate = BW * 1
ber_SF, per_SF, thr_SF = {}, {}, {}
perParam = PerformanceParameters()
for sf in SF:
    noChirps = 2**sf
    symDuration = noChirps / BW
    chirpRate = BW / symDuration
    sampPerSym = sampRate * symDuration
    msg_len = 1 * sf
    loraSys = LoRa(sf, BW, sampRate, initialFreq)
    BER, PER, THROUGHPUT = {}, {}, {}
    for snr in SNR_dB:
        ber, pcr = 0, 0
        noiseVar = 10**(-snr/10)
        ch = MultiPathFading(noiseVar, chVar=chVar, taps=taps)
        for i in range(noIter):
            msg = np.random.randint(0, 2, msg_len)

            sig_tx = loraSys.modulation(msg)
            sig_rx, h = ch.multipathTransmit(sig_tx)
            msg_hat = loraSys.demodulation(sig_rx)[:msg_len]
            ber += perParam.ber(msg_hat, msg)
            pcr += perParam.pcr(msg_hat, msg)
        BER[snr] = (ber / noIter).astype(float)
        PER[snr] = 1 - pcr/noIter
        THROUGHPUT[snr] = pcr/noIter
    print(f"Spreading Factor(SF): {sf} done")
    ber_SF[sf] = BER
    per_SF[sf] = PER
    thr_SF[sf] = THROUGHPUT

plt.figure(1, dpi=800)
for k, v in thr_SF.items():
    plt.plot(v.keys(), v.values(), linestyle='-', linewidth=0.9, label=f"SF = {k}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.xlabel("SNR (dB)")
plt.ylabel("Throughput")
plt.ylim(0, 1.05)
plt.legend(loc='lower right', framealpha=0.6, fontsize=7)
plt.title(f"{noIter} runs per point")
plt.tight_layout()
plt.savefig("results/multitap/thr.jpeg")

plt.figure(2, dpi=800)
for k, v in ber_SF.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f"SF = {k}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate")
plt.title(f"{noIter} runs per point")
plt.ylim(0, 1.05)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.savefig("results/multitap/ber.jpeg")

plt.figure(3, dpi=800)
for k, v in per_SF.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f"SF = {k}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.xlabel("SNR (dB)")
plt.ylabel("PER")
plt.title(f"{noIter} runs per point")
plt.ylim(0, 1.05)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/multitap/per.jpeg")