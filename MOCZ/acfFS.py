"""
auto-correlation functions(ACF), PSD for MOCZ modulation schemes
ch: Frequency Selective Channel
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import BMOCZ, SBMOCZ, JBMOCZ, SlowFadingChannel, IMMOCZ, PMOCZ

zetaJ, zetaS, PZrad = 1.15, 0.0545, 1.578
Rj, Rs = 1.044, 1.0475
K, M, taps = 32, 4, 3
bmoczSys = BMOCZ(K, PZrad=PZrad)
singlePZ = [-PZrad*bmoczSys.R]
sbmoczSys = SBMOCZ(K, zetaS, Rs=Rs)
jbmoczSys = JBMOCZ(K, zetaJ, Rj=Rj)
ch = SlowFadingChannel(noise_var=0.01, taps=taps)

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

rotation = np.random.uniform(0, 2*np.pi)
# rotation = 0
bRx, h = ch.frequencySelective(bTx)
bRxPZ, h = ch.frequencySelective(bTxPZ)
sbRx, h = ch.frequencySelective(sbTx)
jbRx, h = ch.frequencySelective(jbTx)

# print(f"Tx: {bTx},\nh: {h}\nRx: {bRx}")

# PSD for the received signal to observe the rotation
bRx_dtft, omega_dtft = bmoczSys.DTFT(bRx)
bRxPZ_dtft, omega_dtft = bmoczSys.DTFT(bRxPZ)
sbRx_dtft, omega_dtft = sbmoczSys.DTFT(sbRx)
jbRx_dtft, omega_dtft = jbmoczSys.DTFT(jbRx)

sbRx_corrected, rotationEst = sbmoczSys.rotationEst(sbRx)
jbRx_corrected, rotationEst_jb = jbmoczSys.rotationEst(jbRx)
bPZRx_corrected, rotationEst_PZ = bmoczSys.rotationEstTemplate(bRxPZ)

# PSD for the rotation corrected signal
sbRxC_dtft, omega_dtft = sbmoczSys.DTFT(sbRx_corrected)
jbRxC_dtft, omega_dtft = jbmoczSys.DTFT(jbRx_corrected)
bPZRxC_dtft, omega_dtft = bmoczSys.DTFT(bPZRx_corrected)

# print(f"SBMOCZ - Rotation Appled: {rotation}, Rotation Est: {rotationEst}, Error: {abs(rotationEst-rotation)}")
# print(f"JBMOCZ - Rotation Appled: {rotation}, Rotation Est: {rotationEst_jb}, Error: {abs(rotationEst_jb-rotation)}")
# print(f"BMOCZ-PZ - Rotation Appled: {rotation}, Rotation Est: {rotationEst_PZ}, Error: {abs(rotationEst_PZ-rotation)}")

plt.figure(1, dpi=800)
plt.plot(omega_dtft, np.abs(bTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(bRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"Magnitude Spectrum / Power Spectral Density(PSD) of BMCOZ msg:{bmoczSys.bin2dec(msgTx)}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/FS/bPSD.jpeg")

plt.figure(2, dpi=800)
plt.plot(omega_dtft, np.abs(sbTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(sbRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
plt.plot(omega_dtft, np.abs(sbRxC_dtft)**2, '-.', linewidth=0.9, label='Rx Corrected')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"Power Spectral Density(PSD) of SBMCOZ msg:{bmoczSys.bin2dec(msgTx)}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/FS/sbPSD.jpeg")

plt.figure(3, dpi=800)
plt.plot(omega_dtft, np.abs(jbTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, np.abs(jbRx_dtft)**2, linestyle=(0, (5, 3, 1, 3)), linewidth=0.9, label='Rx Signal')
plt.plot(omega_dtft, np.abs(jbRxC_dtft)**2, linestyle=(0, (5, 1, 1, 1)), linewidth=0.9, label='Rx Corrected')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"Power Spectral Density(PSD) of JBMOCZ msg:{bmoczSys.bin2dec(msgTx)}")
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.grid(True, alpha=0.6, linestyle='--')
plt.savefig("results/BMOCZ/ACF/FS/jbPSD.jpeg")

plt.figure(5, dpi=800)
plt.plot(omega_dtft, 10*np.log10(np.abs(bTxPZ_dtft)**2), '--', linewidth=0.9, label='Tx Signal')
plt.plot(omega_dtft, 10*np.log10(np.abs(bRxPZ_dtft)**2), '-', linewidth=0.9, label='Rx Signal')
plt.plot(omega_dtft, 10*np.log10(np.abs(bPZRxC_dtft)**2), '-.', linewidth=0.9, label='Rx Corrected')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"PSD of BMCOZ with Pilot-Zero msg:{bmoczSys.bin2dec(msgTx)}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/BMOCZ/ACF/FS/bPZ_PSD.jpeg")

# # IM-MOCZ
# plt.figure(8, dpi=800)
# plt.plot(omega_dtft, np.abs(imTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
# plt.plot(omega_dtft, np.abs(imRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
# plt.xlabel("Angular Frequency (w)")
# plt.ylabel("Magnitude")
# plt.title(f"Power Spectral Density(PSD) of IM-MCOZ - msg{msgTxIM}")
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig("results/BMOCZ/ACF/FS/imPSD.jpeg")

# # PMOCZ
# plt.figure(9, dpi=800)
# plt.plot(omega_dtft, np.abs(pTx_dtft)**2, '-', linewidth=0.9, label='Tx Signal')
# plt.plot(omega_dtft, np.abs(pRx_dtft)**2, '--', linewidth=0.9, label='Rx Signal')
# plt.xlabel("Angular Frequency (w)")
# plt.ylabel("Magnitude")
# plt.title(f"Power Spectral Density(PSD) of PMCOZ - msg{msgTxIM}")
# plt.grid(True, alpha=0.6, linestyle='--')
# plt.legend(loc='upper right', framealpha=0.6, fontsize=7)
# plt.tight_layout()
# plt.savefig("results/BMOCZ/ACF/FS/pPSD.jpeg")