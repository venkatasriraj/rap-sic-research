"""
auto-correlation functions(ACF), PSD for MOCZ modulation schemes
ch: Block Fading Channel
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import BMOCZ, SBMOCZ, JBMOCZ, SlowFadingChannel, IMMOCZ, PMOCZ

zetaS = 0.0545
zetaJ = 1.07
K = 32
M = 4
PZrad = 1.25
bmoczSys = BMOCZ(K, PZrad=PZrad)
# singlePZ = [-2j*bmoczSys.R, 2j*bmoczSys.R]
singlePZ = [-PZrad*bmoczSys.R]
sbmoczSys = SBMOCZ(K, zetaS)
jbmoczSys = JBMOCZ(K, zetaJ)
immoczSys = IMMOCZ(K, M, PZradius=0)
pmoczSys = PMOCZ(K, M)
ch = SlowFadingChannel(noise_var=0.01) # operating in 20dB SNR region
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

msgLenIM = immoczSys.addBitsLen + K
msgTxIM = np.random.randint(0, 2, msgLenIM)
imTx = immoczSys.coeffCon(msgTxIM)
sigPower = np.mean(np.abs(imTx)**2)
imTx /= np.sqrt(sigPower)

msgLenP = (pmoczSys.bkLen + 1) * K
msgTxP = np.random.randint(0, 2, msgLenP)
pTx = pmoczSys.coeffCon(msgTxP)
sigPower = np.mean(np.abs(pTx)**2)
pTx /= np.sqrt(sigPower)

# PSD / Magnitude spectrum of Transmitted Signal
bTx_dtft, omega_dtft = bmoczSys.DTFT(bTx)
bTxPZ_dtft, omega_dtft = bmoczSys.DTFT(bTxPZ)
sbTx_dtft, omega_dtft = sbmoczSys.DTFT(sbTx)
jbTx_dtft, omega_dtft = jbmoczSys.DTFT(jbTx)
imTx_dtft, omega_dtft = immoczSys.DTFT(imTx)
pTx_dtft, omega_dtft = pmoczSys.DTFT(pTx)

# ACF of Transmitted Signal
bTx_ACF = bmoczSys.AACF(bTx)
bTxPZ_ACF = bmoczSys.AACF(bTxPZ)
sbTx_ACF = sbmoczSys.AACF(sbTx)
jbTx_ACF = jbmoczSys.AACF(jbTx)
imTx_ACF = immoczSys.AACF(imTx)
pTx_ACF = pmoczSys.AACF(pTx)

# Now we apply rotation to the transmitted zeros and observe how the
# peak in the PSD shifts

rotation = np.random.uniform(0, 2*np.pi)
# rotation = 0
bRx = ch.CFO(bTx, rotation)
bRxPZ = ch.CFO(bTxPZ, rotation)
sbRx = ch.CFO(sbTx, rotation)
jbRx = ch.CFO(jbTx, rotation)
imRx = ch.CFO(imTx, rotation)
pRx = ch.CFO(pTx, rotation)
# PSD for the received signal to observe the rotation
bRx_dtft, omega_dtft = bmoczSys.DTFT(bRx)
bRxPZ_dtft, omega_dtft = bmoczSys.DTFT(bRxPZ)
sbRx_dtft, omega_dtft = sbmoczSys.DTFT(sbRx)
jbRx_dtft, omega_dtft = jbmoczSys.DTFT(jbRx)
imRx_dtft, omega_dtft = immoczSys.DTFT(imRx)
pRx_dtft, omega_dtft = pmoczSys.DTFT(pRx)

sbRx_corrected, rotationEst = sbmoczSys.rotationEst(sbRx)
jbRx_corrected, rotationEst_jb = jbmoczSys.rotationEst(jbRx)
bPZRx_corrected, rotationEst_PZ = bmoczSys.rotationEstTemplate(bRxPZ)
# PSD for the rotation corrected signal
sbRxC_dtft, omega_dtft = sbmoczSys.DTFT(sbRx_corrected)
jbRxC_dtft, omega_dtft = jbmoczSys.DTFT(jbRx_corrected)
bPZRxC_dtft, omega_dtft = bmoczSys.DTFT(bPZRx_corrected)
print(f"SBMOCZ - Rotation Appled: {rotation}, Rotation Est: {rotationEst}, Error: {abs(rotationEst-rotation)}")
print(f"JBMOCZ - Rotation Appled: {rotation}, Rotation Est: {rotationEst_jb}, Error: {abs(rotationEst_jb-rotation)}")
print(f"BMOCZ-PZ - Rotation Appled: {rotation}, Rotation Est: {rotationEst_PZ}, Error: {abs(rotationEst_PZ-rotation)}")

plt.figure(1, dpi=800)
plt.plot(omega_dtft, np.abs(bTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(bRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum / Power Spectral Density(PSD) of BMCOZ")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/CFO/bPSD.jpeg")

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
plt.savefig("results/BMOCZ/ACF/CFO/sbPSD.jpeg")

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
plt.savefig("results/BMOCZ/ACF/CFO/jbPSD.jpeg")

plt.figure(4, dpi=800)
plt.plot(omega_dtft, np.abs(bTx_dtft)**2, '-', linewidth=0.9, label='BMOCZ')
plt.plot(omega_dtft, np.abs(sbTx_dtft)**2, '--', linewidth=0.9, label='SBMOCZ')
plt.plot(omega_dtft, np.abs(bTxPZ_dtft)**2, '-.', linewidth=0.9, label="BMCOZ-PZ")
plt.plot(omega_dtft, np.abs(jbTx_dtft)**2, ':', linewidth=0.9, label='JBMOCZ')
# plt.plot(omega_dtft, np.abs(pTx_dtft)**2, linestyle=(0, (5, 3, 1, 3)), linewidth=0.9, label="IM-MCOZ")
# plt.plot(omega_dtft, np.abs(imTx_dtft)**2, linestyle=(0, (5, 1, 1, 1)), linewidth=0.9, label="PMCOZ")
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title("PSD Analysis for MOCZ schemes")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/CFO/PSD.jpeg")

plt.figure(5, dpi=800)
plt.plot(omega_dtft, (np.abs(bTxPZ_dtft)**1), '--', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, (np.abs(bRxPZ_dtft)**1), '-', linewidth=0.9, label='Rx Signal')
plt.plot(omega_dtft, np.abs(bPZRxC_dtft)**1, '-.', linewidth=0.9, label='Rx Corrected')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"PSD of BMCOZ with Pilot-Zero CFO={np.round(rotation, 4)} Rad")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/CFO/bPZ_PSD.jpeg")

plt.figure(6, dpi=800)
plt.plot(np.arange(-K, K+1), bTx_ACF, '-', linewidth=0.9, label='BMOCZ')
plt.plot(np.arange(-K, K+1), sbTx_ACF, '--', linewidth=0.9, label='SBMOCZ')
plt.plot(np.arange(-(K+1), K+2), bTxPZ_ACF, '-.', linewidth=0.9, label="BMCOZ-PZ")
plt.plot(np.arange(-K, K+1), jbTx_ACF, ':', linewidth=0.9, label='JBMOCZ')
# plt.plot(np.arange(-(K+1), K+2), imTx_ACF, linestyle=(0, (5, 3, 1, 3)), linewidth=0.9, label="IM-MCOZ")
# plt.plot(np.arange(-(K+0), K+1), pTx_ACF, linestyle=(0, (5, 1, 1, 1)), linewidth=0.9, label="PMCOZ")
plt.xlabel(f"Index - {msgTx}")
plt.ylabel("Amplitude")
plt.title("ACF Analysis for MOCZ schemes")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/CFO/AACF.jpeg")

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
plt.savefig("results/BMOCZ/ACF/CFO/jTemplate.jpeg")

# IM-MOCZ
plt.figure(8, dpi=800)
plt.plot(omega_dtft, np.abs(imTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(imRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"Power Spectral Density(PSD) of IM-MCOZ - msg{msgTxIM}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/CFO/imPSD.jpeg")

# PMOCZ
plt.figure(9, dpi=800)
plt.plot(omega_dtft, np.abs(pTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(pRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"Power Spectral Density(PSD) of PMCOZ - msg{msgTxIM}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/CFO/pPSD.jpeg")

# styles = {
#     "dashed": (0, (5, 5)),
#     "long_dash": (0, (10, 5)),
#     "dash_dot": (0, (5, 3, 1, 3)),
#     "dotted": (0, (1, 3)),
#     "custom": (0, (8, 2, 2, 2, 2, 2)),
# }