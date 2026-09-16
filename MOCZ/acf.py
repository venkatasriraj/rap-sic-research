"""
Testing of auto-correlation functions(ACF) for MOCZ modulation schemes

"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import BMOCZ, SBMOCZ, JBMOCZ, MultiPathFading

zetaS = 0.1
zetaJ = 1.1
K = 16
bmoczSys = BMOCZ(K)
singlePZ = [-1.25*bmoczSys.R]
sbmoczSys = SBMOCZ(K, zetaS)
# jbmoczSys = JBMOCZ(K, zetaJ)
ch = MultiPathFading(noise_var=0.01) # operating in 20dB SNR region
msgTx = np.random.randint(0, 2, K)
bTx = bmoczSys.coeffCon(msgTx)
sigPower = np.mean(np.abs(bTx)**2)
bTx /= np.sqrt(sigPower)

bTxPZ = bmoczSys.coeffCon(msgTx, singlePZ)
sigPower = np.mean(np.abs(bTxPZ)**2)
bTxPZ /= np.sqrt(sigPower)

sbTx = sbmoczSys.coeffCon(msgTx)
sigPower = np.mean(np.abs(sbTx)**2)
sbTx /= np.sqrt(sigPower)

# jbTx = jbmoczSys.coeffCon(msgTx)

# PSD / Magnitude spectrum of Transmitted Signal
bTx_dtft, omega_dtft = bmoczSys.DTFT(bTx)
bTxPZ_dtft, omega_dtft = bmoczSys.DTFT(bTxPZ)
sbTx_dtft, omega_dtft = sbmoczSys.DTFT(sbTx)
# jbTx_dtft, omega_dtft = jbmoczSys.DTFT(jbTx)

# ACF of Transmitted Signal
bTx_ACF = bmoczSys.AACF(bTx)
bTxPZ_ACF = bmoczSys.AACF(bTxPZ)
sbTx_ACF = sbmoczSys.AACF(sbTx)

# Now we apply rotation to the transmitted zeros and observe how the
# peak in the PSD shifts

rotation = np.random.uniform(0, 2*np.pi)
bRx = ch.transmit(bTx, rotation)
bRxPZ = ch.transmit(bTxPZ, rotation)
sbRx = ch.transmit(sbTx, rotation)
# PSD for the received signal to observe the rotation
bRx_dtft, omega_dtft = bmoczSys.DTFT(bRx)
bRxPZ_dtft, omega_dtft = bmoczSys.DTFT(bRxPZ)
sbRx_dtft, omega_dtft = sbmoczSys.DTFT(sbRx)

sbRx_corrected, rotationEst = sbmoczSys.rotationEst(sbRx)
# PSD for the rotation corrected signal
sbRxC_dtft, omega_dtft = sbmoczSys.DTFT(sbRx_corrected)
print(f"Rotation Appled: {rotation}, Rotation Est: {rotationEst}, Error: {abs(rotationEst-rotation)}")

plt.figure(1, dpi=800)
plt.plot(omega_dtft, np.abs(bTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(bRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.title("Magnitude Spectrum / Power Spectral Density(PSD) of BMCOZ")
plt.savefig("results/BMOCZ/ACF/bPSD.jpeg")

plt.figure(2, dpi=800)
plt.plot(omega_dtft, np.abs(sbTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(sbRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.plot(omega_dtft, np.abs(sbRxC_dtft)**2, '-.', linewidth=0.9, label='Rx Corrected')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.title("Magnitude Spectrum / Power Spectral Density(PSD) of SBMCOZ")
plt.savefig("results/BMOCZ/ACF/sbPSD.jpeg")

# plt.figure(3, dpi=800)
# plt.plot(omega_dtft, np.abs(jbTx_dtft)**2, '-', linewidth=0.9)
# plt.xlabel("Angular Frequency (w)")
# plt.ylabel("Magnitude")
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.title("Magnitude Spectrum / Power Spectral Density(PSD)of JBMOCZ")
# plt.savefig("results/BMOCZ/ACF/jbPSD.jpeg")

plt.figure(4, dpi=800)
plt.plot(omega_dtft, np.abs(bTx_dtft)**2, '-', linewidth=0.9, label='BMOCZ')
plt.plot(omega_dtft, np.abs(sbTx_dtft)**2, '--', linewidth=0.9, label='SBMOCZ')
plt.plot(omega_dtft, np.abs(bTxPZ_dtft)**2, '-.', linewidth=0.9, label="BMCOZ-PZ")
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.title("PSD Analysis for MOCZ schemes")
plt.savefig("results/BMOCZ/ACF/PSD.jpeg")

plt.figure(5, dpi=800)
plt.plot(omega_dtft, np.abs(bTxPZ_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(bRxPZ_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.title("Power Spectral Density(PSD) of BMCOZ with Pilot-Zero")
plt.savefig("results/BMOCZ/ACF/bPZ_PSD.jpeg")