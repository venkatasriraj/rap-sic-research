"""
Testing of auto-correlation functions(ACF) for MOCZ modulation schemes

"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import BMOCZ, SBMOCZ, JBMOCZ, MultiPathFading

zetaS = 0.0545
zetaJ = 1.07
K = 31
bmoczSys = BMOCZ(K)
singlePZ = [-1.3*bmoczSys.R]
sbmoczSys = SBMOCZ(K, zetaS)
jbmoczSys = JBMOCZ(K, zetaJ)
ch = MultiPathFading(noise_var=0.01) # operating in 20dB SNR region
# msgTx = np.ones(K, dtype=np.uint8)
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

jbTx = jbmoczSys.coeffCon(msgTx)
sigPower = np.mean(np.abs(jbTx)**2)
jbTx /= np.sqrt(sigPower)

# PSD / Magnitude spectrum of Transmitted Signal
bTx_dtft, omega_dtft = bmoczSys.DTFT(bTx)
bTxPZ_dtft, omega_dtft = bmoczSys.DTFT(bTxPZ)
sbTx_dtft, omega_dtft = sbmoczSys.DTFT(sbTx)
jbTx_dtft, omega_dtft = jbmoczSys.DTFT(jbTx)

# ACF of Transmitted Signal
bTx_ACF = bmoczSys.AACF(bTx)
bTxPZ_ACF = bmoczSys.AACF(bTxPZ)
sbTx_ACF = sbmoczSys.AACF(sbTx)
jbTx_ACF = jbmoczSys.AACF(jbTx)

# Now we apply rotation to the transmitted zeros and observe how the
# peak in the PSD shifts

rotation = np.random.uniform(0, 2*np.pi)
bRx = ch.transmit(bTx, rotation)
bRxPZ = ch.transmit(bTxPZ, rotation)
sbRx = ch.transmit(sbTx, rotation)
jbRx = ch.transmit(jbTx, rotation)
# PSD for the received signal to observe the rotation
bRx_dtft, omega_dtft = bmoczSys.DTFT(bRx)
bRxPZ_dtft, omega_dtft = bmoczSys.DTFT(bRxPZ)
sbRx_dtft, omega_dtft = sbmoczSys.DTFT(sbRx)
jbRx_dtft, omega_dtft = jbmoczSys.DTFT(jbRx)

sbRx_corrected, rotationEst = sbmoczSys.rotationEst(sbRx)
jbRx_corrected, rotationEst_jb = jbmoczSys.rotationEst(jbRx)
# PSD for the rotation corrected signal
sbRxC_dtft, omega_dtft = sbmoczSys.DTFT(sbRx_corrected)
jbRxC_dtft, omega_dtft = jbmoczSys.DTFT(jbRx_corrected)
print(f"SBMOCZ - Rotation Appled: {rotation}, Rotation Est: {rotationEst}, Error: {abs(rotationEst-rotation)}")
print(f"JBMOCZ - Rotation Appled: {rotation}, Rotation Est: {rotationEst_jb}, Error: {abs(rotationEst_jb-rotation)}")

plt.figure(1, dpi=800)
plt.plot(omega_dtft, np.abs(bTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(bRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum / Power Spectral Density(PSD) of BMCOZ")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/bPSD.jpeg")

plt.figure(2, dpi=800)
plt.plot(omega_dtft, np.abs(sbTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(sbRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.plot(omega_dtft, np.abs(sbRxC_dtft)**2, '-.', linewidth=0.9, label='Rx Corrected')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum / Power Spectral Density(PSD) of SBMCOZ")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/sbPSD.jpeg")

plt.figure(3, dpi=800)
plt.plot(omega_dtft, np.abs(jbTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(jbRx_dtft)**2, '-', linewidth=0.9, label='Rx Signal')
plt.plot(omega_dtft, np.abs(jbRxC_dtft)**2, '-', linewidth=0.9, label='Rx Corrected')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum / Power Spectral Density(PSD)of JBMOCZ")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.grid(True, alpha=0.6, linestyle='--')
plt.savefig("results/BMOCZ/ACF/jbPSD.jpeg")

plt.figure(4, dpi=800)
plt.plot(omega_dtft, np.abs(bTx_dtft)**2, '-', linewidth=0.9, label='BMOCZ')
plt.plot(omega_dtft, np.abs(sbTx_dtft)**2, '--', linewidth=0.9, label='SBMOCZ')
plt.plot(omega_dtft, np.abs(bTxPZ_dtft)**2, '-.', linewidth=0.9, label="BMCOZ-PZ")
plt.plot(omega_dtft, np.abs(jbTx_dtft)**2, ':', linewidth=0.9, label='JBMOCZ')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title("PSD Analysis for MOCZ schemes")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/PSD.jpeg")

plt.figure(5, dpi=800)
plt.plot(omega_dtft, np.abs(bTxPZ_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(bRxPZ_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title("Power Spectral Density(PSD) of BMCOZ with Pilot-Zero")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/bPZ_PSD.jpeg")

plt.figure(6, dpi=800)
plt.plot(np.arange(-K, K+1), bTx_ACF, '-', linewidth=0.9, label='BMOCZ')
plt.plot(np.arange(-K, K+1), sbTx_ACF, '--', linewidth=0.9, label='SBMOCZ')
plt.plot(np.arange(-(K+1), K+2), bTxPZ_ACF, '-.', linewidth=0.9, label="BMCOZ-PZ")
plt.plot(np.arange(-K, K+1), jbTx_ACF, ':', linewidth=0.9, label='JBMOCZ')
plt.xlabel(f"Index - {msgTx}")
plt.ylabel("Amplitude")
plt.title("ACF Analysis for MOCZ schemes")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/AACF.jpeg")

template_dtft = np.fft.fftshift(jbmoczSys.template)
plt.figure(7, dpi=800)
plt.plot(omega_dtft, template_dtft, '-', linewidth=0.9, label='T(w)')
plt.plot(omega_dtft, np.roll(template_dtft, 128), '-', linewidth=0.9, label=f'T(w+{np.round(np.pi/4, 3)})')
plt.xlabel("Angular Frequency(w)")
plt.ylabel("Amplitude")
plt.title("JBMOCZ Template Vector (DTFT)")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/jTemplate.jpeg")