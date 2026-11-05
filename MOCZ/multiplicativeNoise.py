"""
Y(z) = X(z) * H(z); here we consider H(z) as the multiplicative noise
We conider log to transform multiplicative to additive noise
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import (
    BMOCZ, JBMOCZ, SBMOCZ, SlowFadingChannel
)

K, PZrad = 31, 1.35
zetaJ, Rj = 1.15, 1.044
zetaS, Rs = 0.0545, 1.045
Lh, chVar = 2, 0.86
ENERGY = Lh + K
mpeak = [0, 7, 17, 23]

bmoczSys = BMOCZ(K, PZrad=0)
bmoczSysPZ = BMOCZ(K, PZrad=PZrad)
sbmoczSys = SBMOCZ(K, zetaS)
jbmoczSys = JBMOCZ(K, zetaJ, Rj=1.044)
jbmoczSysM = JBMOCZ(K, zetaJ, mPeak=mpeak)
msgTx = np.random.randint(0, 2, K)

bTx = bmoczSys.transmitterRAP(msgTx, ENERGY)
bTxPZ = bmoczSysPZ.transmitterRAP(msgTx, ENERGY)
sTx = sbmoczSys.transmitterRAP(msgTx, ENERGY)
jTx = jbmoczSys.transmitterRAP(msgTx, ENERGY)
jTxM = jbmoczSysM.transmitterRAP(msgTx, ENERGY)

bTx_dtft, omega_dtft = bmoczSys.DTFT(bTx)
bTxPZ_dtft, _ = bmoczSysPZ.DTFT(bTxPZ)
sTx_dtft, _ = sbmoczSys.DTFT(sTx)
jTx_dtft, _ = jbmoczSys.DTFT(jTx)
jTxM_dtft, _ = jbmoczSysM.DTFT(jTxM)

ch = SlowFadingChannel(noise_var=10**(-20/10), chVar=chVar, taps=Lh)

bRx, h = ch.frequencySelective(bTx, noiseFlag=False)
bRxPZ, h = ch.frequencySelective(bTxPZ, noiseFlag=False)
sRx, h = ch.frequencySelective(sTx, noiseFlag=False)
jRx, h = ch.frequencySelective(jTx, noiseFlag=False)
jRxM, h = ch.frequencySelective(jTxM, noiseFlag=False)

bRx_dtft, omega_dtft = bmoczSys.DTFT(bRx)
bRxPZ_dtft, _ = bmoczSysPZ.DTFT(bRxPZ)
sRx_dtft, _ = sbmoczSys.DTFT(sRx)
jRx_dtft, _ = jbmoczSys.DTFT(jRx)
jRxM_dtft, _ = jbmoczSysM.DTFT(jRxM)

plt.figure(1, dpi=800)
plt.plot(omega_dtft, np.abs(bTx_dtft), '-', linewidth=1.2, label='BMOCZ')
plt.plot(omega_dtft, np.abs(bTxPZ_dtft), '-', linewidth=1.2, label='BMOCZ_PZ')
plt.plot(omega_dtft, np.abs(sTx_dtft), '-', linewidth=1.2, label='SBMOCZ')
plt.plot(omega_dtft, np.abs(jTx_dtft), '-', linewidth=1.2, label='JBMOCZ')
plt.plot(omega_dtft, np.abs(jTxM_dtft), '-', linewidth=1.2, label='JBMOCZ-M')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"Magnitude Spectrum / Power Spectral Density(PSD) of BMCOZ msg:{bmoczSys.bin2dec(msgTx)}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='best', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/ACF/absTxPSD.jpeg")

plt.figure(2, dpi=800)
plt.plot(omega_dtft, np.log(np.abs(bTx_dtft)), '-', linewidth=1.2, label='BMOCZ')
plt.plot(omega_dtft, np.log(np.abs(bTxPZ_dtft)), '-', linewidth=1.2, label='BMOCZ_PZ')
plt.plot(omega_dtft, np.log(np.abs(sTx_dtft)), '-', linewidth=1.2, label='SBMOCZ')
plt.plot(omega_dtft, np.log(np.abs(jTx_dtft)), '-', linewidth=1.2, label='JBMOCZ')
plt.plot(omega_dtft, np.log(np.abs(jTxM_dtft)), '-', linewidth=1.2, label='JBMOCZ-M')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"Magnitude Spectrum / Power Spectral Density(PSD) of BMCOZ msg:{bmoczSys.bin2dec(msgTx)}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='best', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/ACF/logTxPSD.jpeg")

plt.figure(3, dpi=800)
# plt.plot(omega_dtft, np.abs(bRx_dtft), '-', linewidth=1.2, label='BMOCZ')
# plt.plot(omega_dtft, np.abs(bRxPZ_dtft), '-', linewidth=1.2, label='BMOCZ_PZ')
# plt.plot(omega_dtft, np.abs(sRx_dtft), '-', linewidth=1.2, label='SBMOCZ')
plt.plot(omega_dtft, np.abs(jRx_dtft), '-', linewidth=1.2, label='JBMOCZ')
# plt.plot(omega_dtft, np.abs(jRxM_dtft), '-', linewidth=1.2, label='JBMOCZ-M')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"Magnitude Spectrum / Power Spectral Density(PSD) of BMCOZ msg:{bmoczSys.bin2dec(msgTx)}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='best', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/ACF/absRxPSD.jpeg")

plt.figure(4, dpi=800)
# plt.plot(omega_dtft, np.log(np.abs(bRx_dtft)), '-', linewidth=1.2, label='BMOCZ')
# plt.plot(omega_dtft, np.log(np.abs(bRxPZ_dtft)), '-', linewidth=1.2, label='BMOCZ_PZ')
# plt.plot(omega_dtft, np.log(np.abs(sRx_dtft)), '-', linewidth=1.2, label='SBMOCZ')
plt.plot(omega_dtft, np.log(np.abs(jRx_dtft)), '-', linewidth=1.2, label='JBMOCZ')
# plt.plot(omega_dtft, np.log(np.abs(jRxM_dtft)), '-', linewidth=1.2, label='JBMOCZ-M')
plt.xlabel("Angular Frequency (w)")
plt.ylabel("Magnitude")
plt.title(f"Magnitude Spectrum / Power Spectral Density(PSD) of BMCOZ msg:{bmoczSys.bin2dec(msgTx)}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.legend(loc='best', framealpha=0.6, fontsize=7)
plt.tight_layout()
plt.savefig("results/ACF/logRxPSD.jpeg")