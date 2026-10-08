"""
Analysis of zero positions at different stages of decoding
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import JUTTEDfs, SlowFadingChannel

K, Q_f = 127, 32
FLAG = "AWGN"
Lh, chVar = 2, 0.95
Q = int(np.ceil((Lh + K + 1)/K))
zeta, Rj = 1.32, None
jSystem = JUTTEDfs(K, zeta, Rj=Rj)
ch = SlowFadingChannel(noise_var=10**(-15/10), chVar=chVar, taps=Lh)
zerosCh = np.array([])
zerosRx = np.array([])
zerosFFO = np.array([])
zerosIRO = np.array([])

msgTx = np.random.randint(0, 2, K)
R = jSystem.R
zeroCollection = jSystem.zeroCollection
zerosTx = jSystem.zeroSelection(msgTx)

sigTx = jSystem.coeffCon(msgTx)        
sigPower = np.mean(np.abs(sigTx)**2)
sigTx /= np.sqrt(sigPower)

rotation = np.random.uniform(0, 2*np.pi)
sigRx, h = ch.frequencySelective(sigTx, noiseFlag=True, rotation=rotation)

sig_ffo, theta_hat = jSystem.ffoCorrection(sigRx, Q_f)
sig_corrected, phi_hat = jSystem.iroCorrection(sig_ffo, Q_f)
# sig_corrected, phi_hat = jSystem.iroCorrectionZeta(sig_ffo, Q_f)
msgRx = jSystem.fftDizet(sig_corrected, Q_f)
rotation_hat = theta_hat + phi_hat

zerosCh = np.roots(h[::-1])
zerosRx = np.roots(sigRx[::-1])
zerosFFO = np.roots(sig_ffo[::-1])
zerosIRO = np.roots(sig_corrected[::-1])

plt.figure(1, dpi=800)
ax = plt.gca()

ax.axhline(y=0, color='black', linewidth=0.6, linestyle=(0, (6, 6)))
ax.axvline(x=0, color='black', linewidth=0.6, linestyle=(0, (6, 6)))

ax.add_patch( plt.Circle((0, 0), radius=1, color='gray', fill=False, linestyle=(0, (2,2))) )
ax.add_patch( plt.Circle((0, 0), radius=R, color='gray', fill=False, linestyle=(0, (1,1))) )
ax.add_patch( plt.Circle((0, 0), radius=R**-1, color='gray', fill=False, linestyle=(0, (1,1))) )

plt.scatter(zeroCollection.real, zeroCollection.imag, edgecolors='black', facecolors='none', marker='o', linewidths=1.3, s=60, zorder=1, label='Zero collection')
plt.scatter(zerosCh.real, zerosCh.imag, marker='x', color='#D55E00', linewidths=1.5, s=40, zorder=2, label='Ch')
plt.scatter(zerosTx.real, zerosTx.imag, marker='D', facecolors='none', edgecolors='#0072B2', linewidths=1.4, s=45, zorder=3, label='Tx')
plt.scatter(zerosRx.real, zerosRx.imag, marker='^', facecolors='none', edgecolors='#009E73', linewidths=1.4, s=50, zorder=4, label='Rx')
plt.scatter(zerosFFO.real, zerosFFO.imag, marker='s', facecolors='none', edgecolors='#CC79A7', linewidths=1.4, s=48, zorder=5, label='FRO')
plt.scatter(zerosIRO.real, zerosIRO.imag, marker='*', color='#E69F00', linewidths=1.4, s=65, zorder=6, label='IRO')

plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.title(f"{FLAG}-{np.round(rotation, 4)}rad K-{K} for Lh-{Lh}, chTapVar-{chVar}")
plt.legend(loc='upper right', fontsize=7, framealpha=0.6)
plt.tight_layout()
plt.axis('equal')
plt.savefig(f"results/ACF/FS/IRO/{FLAG}Lim{Lh}_{chVar}.jpeg")