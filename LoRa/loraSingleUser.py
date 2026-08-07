"""
Implementation of LoRa class for user
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from lora import LoRa

SF = 8
BW = 125e3  # 125 kHz
sampRate = BW * 2 #
initalfreq = -BW/2 # -BW/2
symDuration = 2**SF / BW
chirpRate = BW / symDuration

msg_len = 8
msg = np.random.randint(0, 2, msg_len)
print(f"messsage: {msg}")
print(f"Spreading Factor: {SF} \n Symbol Duration: {symDuration}\n "
        f"Chirp Rate: {chirpRate}")

time = np.arange(0, symDuration, 1/sampRate)
phi_t = 2*np.pi * ( initalfreq*time + 0.5*chirpRate * time**2)
refChirp = np.exp( 1j * phi_t)

symChirp = np.roll(refChirp, int(56 * len(refChirp) / 2**SF))
f, tt, Sxx = spectrogram(symChirp, sampRate, nperseg=64, noverlap=56)
plt.figure()
plt.pcolormesh(tt, np.fft.fftshift(f), 
        10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-12), shading='gouraud')
plt.savefig("results/spectrogramSym.jpeg")
# plt.show()

plt.figure(2, dpi=800)
plt.plot(time, np.abs(refChirp), '-', linewidth=0.9, color='r')
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Reference Chirp")
# plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
# plt.tight_layout()
plt.savefig("results/refChirp.jpeg")