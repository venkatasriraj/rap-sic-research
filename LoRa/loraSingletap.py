"""
Implementation of up-chirp spread spectrum modulation for 
block-fading quasi-static channel
chirp rate = max(hop rate, symbol rate)
chirp is defined as the smallest duration of frequency tone
spreading factor(SF): number of bits per sym can be transmitted 
number of chirps(N) : number of chirps per symbol
preocessing gain(PG): obtained due to spreading of symbol over the bandwidth
"""
import numpy as np
import matplotlib.pyplot as plt
from lora import LoRa
from wirelessComm import (
    SlowFadingChannel, PerformanceParameters
)
SF = np.arange(7, 16, 1)
SNR_dB = np.arange(-20, 21, 5)
noIter = int(1e4)
BW = 125e3
initialFreq = -0.5 * BW
# symDuration = 1 / BW
# chirpDuration = symDuration / noChirps
# chirpRate = 1 / chirpDuration  # Rc = N * Rs ensures orthogonality
# sampRate = noChirps * BW
# sampPerSym = sampRate * symDuration
sampRate = BW * 1 
ber_SF, per_SF, thr_SF = {}, {}, {}
for sf in SF:
    noChirps = 2**sf 
    symDuration = noChirps / BW
    chirpRate = BW / symDuration
    sampPerSym = sampRate * symDuration
    msg_len = 1 * sf # in bits
    loraSys = LoRa(sf, BW, sampRate, initialFreq)
    perParam = PerformanceParameters()
    # print(f"Samples per Symbol: {sampPerSym}")
    # print(f"Symbol Duration {symDuration}")
    # print(f"Chirp Duration {chirpDuration}")
    # print(f"Chirp rate {chirpRate}")
    # print(f"Bandwidth: {BW}")
    BER, THROUGHPUT, PER = {}, {}, {}
    for snr in SNR_dB:
        ber, pcr = 0, 0
        ch = SlowFadingChannel(noise_var=10**(-snr/10))
        for i in range(noIter):
            msg = np.random.randint(0, 2, msg_len)

            sig_tx = loraSys.modulation(msg)
            sig_rx, h = ch.transmit(sig_tx)
            msg_hat = loraSys.demodulation(sig_rx)

            ber += perParam.ber(msg_hat, msg)
            pcr += perParam.pcr(msg_hat, msg)
        # print(f"BER: {ber/noIter} \nPER: {1 - (pcr/noIter)}")
        BER[snr] = (ber / noIter).astype(float)
        PER[snr] = 1 - pcr/noIter
        THROUGHPUT[snr] = pcr/(noIter * symDuration)
    print(f"SF: {sf} done.")
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
plt.savefig("results/singletap/thr.jpeg")

plt.figure(2, dpi=800)
for k, v in ber_SF.items():
    plt.plot(v.keys(), v.values(), '-', linewidth=0.9, label=f"SF = {k}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.xlabel("SNR (dB)")
plt.ylabel("Bit Error Rate")
plt.title(f"{noIter} runs per point")
plt.ylim(0, 1.05)
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.savefig("results/singletap/ber.jpeg")

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
plt.savefig("results/singletap/per.jpeg")