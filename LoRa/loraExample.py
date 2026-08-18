"""
Implementation of LoRa class for user
if each chirp requires BW = 125KHz, 
then for a symbol we need total BW = BW * 2**SF = 32MHz
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from lora import LoRa
from wirelessComm import (
    MultiPathFading, SlowFadingChannel, PerformanceParameters
)

SF = 8
noChirps = 2**SF
BW = 125e3  # 125 kHz
sampRate = BW * 1 #
initalfreq = -BW/2 # -BW/2
symDuration = 2**SF / BW
chirpRate = BW / symDuration
sampPerSym = sampRate * symDuration
lora = LoRa(SF, BW, sampRate, initalfreq)
SNR = 20
msg_len = 8
msg = np.random.randint(0, 2, msg_len)
print(f"messsage: {msg}")
print(f"Spreading Factor: {SF} \n Symbol Duration: {symDuration}\n "
        f"Chirp Rate: {chirpRate}")

time = np.arange(0, symDuration, 1/sampRate)
phi_t = 2*np.pi * ( initalfreq*time + 0.5*chirpRate * time**2)
refChirp = np.exp( 1j * phi_t)

## ------ PLOT OF SPECTROGRAM FOR SSM SIGNAL
symbol = lora.bin2dec(msg)
# % we need to rotate the reference signal in left direction
symChirp = np.roll(refChirp, int(-symbol * len(refChirp) / 2**SF))
# f, tt, Sxx = spectrogram(symChirp, sampRate, nperseg=64, noverlap=56)
# plt.figure()
# plt.pcolormesh(tt, np.fft.fftshift(f), 
#         10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-12), shading='gouraud')
# plt.savefig("results/spectrogramSym.jpeg")
# # plt.show()

# plt.figure(2, dpi=800)
# plt.plot(time, np.abs(refChirp), '-', linewidth=0.9, color='r')
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("Reference Chirp")
# # plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
# # plt.tight_layout()
# plt.savefig("results/refChirp.jpeg")

##  ----- ENERGY OF EACH SYMBOL
# for i in range(noChirps):
#     symChirp = np.roll(refChirp, int(i * len(refChirp) / noChirps))
#     energy = np.sum(np.abs(symChirp)**2)
#     print(f"Symbol: {i}, Energy of the symbol: {energy}")

## --- Dechirping and Symbol detection using FFT
# deChirp = symChirp * np.conjugate(refChirp)
# fftTone = np.fft.fft(deChirp)
# symHat = np.argmax(np.abs(fftTone)) 
# msgDecoded = lora.dec2bin(symHat)

# print(f"Symbol Est: {symHat}, Symbol Transmitted: {symbol}")


## ---- Spread Spectrum Modulation and Demodulation using LoRa Class
signalTx = lora.modulation(msg)
# msg_hat = lora.demodulation(signalTx)
# print(f"Decoded message: {msg_hat}")

## ----- Multipath channel analysis for LoRa
ch = MultiPathFading(noise_var=10**(-SNR/10), chVar=1, taps=2)
sig_rx, h = ch.multipathTransmit(signalTx)
# plt.figure(1, dpi=800)
# plt.plot(np.arange(len(signalTx))/sampRate, signalTx, label='transmitted signal')
# plt.plot(np.arange(len(sig_rx))/sampRate, sig_rx, label="received signal")
# plt.xlabel("time")
# plt.ylabel("amplitude")
# plt.legend()
# plt.tight_layout()
# plt.savefig("results/multipath.jpeg")
#----------- spectrogram analysis for Rx signal
# f, tt, Sxx = spectrogram(sig_rx, sampRate, nperseg=64, noverlap=56)
# plt.figure()
# plt.pcolormesh(tt, np.fft.fftshift(f), 
#         10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-12), shading='gouraud')
# plt.savefig("results/specRxSym.jpeg")
# # -- spectrogram analysis for Tx signal
# f, tt, Sxx = spectrogram(signalTx, sampRate, nperseg=64, noverlap=56)
# plt.figure()
# plt.pcolormesh(tt, np.fft.fftshift(f), 
#         10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-12), shading='gouraud')
# plt.savefig("results/specTxSym.jpeg")
msg_hat = lora.demodulation(sig_rx)
ber = PerformanceParameters.ber(msg_hat[:msg_len], msg)
print(f"BER: {ber}")
print(f"MSG tx: {msg}")
print(f"Msg rx: {msg_hat}")