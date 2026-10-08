"""
Template Analysis for JBMOCZ over frequency selective channel
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import SlowFadingChannel, JBMOCZ, SBMOCZ, BMOCZ

K, PZrad = 32, 1.578
Rj, zetaJ = 1.044, 1.15
Rs, zetaS = 1.0475, 0.0545
taps, chVar = 2, 0.75
bmoczSys = BMOCZ(K, PZrad=PZrad)
sbmoczSys = SBMOCZ(K, zetaS, Rs=Rs)
jbmoczSys = JBMOCZ(K, zetaJ, Rj=Rj)
ch = SlowFadingChannel(noise_var=0.01, taps=taps, chVar=chVar)
MODES = ['PZ', 'S', 'J']
zerosCh = np.array([])

mode = MODES[2]
noIter = int(1e4)
plt.figure(1, dpi=800)

for _ in range(noIter):
    msgTx = np.random.randint(0, 2, K)

    if mode == 'J':
        sys = jbmoczSys
    elif mode == 'S':
        sys = sbmoczSys
    elif mode == 'PZ':
        sys = bmoczSys

    R = sys.R
    zeroCollection = sys.zeroCollection
    zerosTx = sys.zeroSelection(msgTx)
    sigTx = sys.coeffCon(msgTx)        
    sigPower = np.mean(np.abs(sigTx)**2)
    sigTx /= np.sqrt(sigPower)

    sigRx, h = ch.frequencySelective(sigTx)
    sigRx_dtft, omega_dtft = sys.DTFT(sigRx)
    zerosCh = np.concatenate((zerosCh, np.roots(h[::-1])), axis=None)

    plt.plot(omega_dtft, np.abs(sigRx_dtft)**1, linestyle=(0, (1, 3, 1, 2)), linewidth=0.9)
plt.xlabel("Angular Frequency(w)")
plt.ylabel("Magnitude")
plt.title(f"|DTFT of {mode}-Received Signal| K-{K} for Lh-{taps}, chTapVar-{chVar}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.savefig(f"results/ACF/FS/{mode}Template{taps}_{chVar}.jpeg")

zerosCh = np.asarray(zerosCh)
plt.figure(2, dpi=800)
ax = plt.gca()

ax.axhline(y=0, color='black', linewidth=0.6, linestyle=(0, (6, 6)))
ax.axvline(x=0, color='black', linewidth=0.6, linestyle=(0, (6, 6)))

ax.add_patch( plt.Circle((0, 0), radius=1, color='gray', fill=False, linestyle=(0, (2,2))) )
ax.add_patch( plt.Circle((0, 0), radius=R, color='gray', fill=False, linestyle=(0, (1,1))) )
ax.add_patch( plt.Circle((0, 0), radius=R**-1, color='gray', fill=False, linestyle=(0, (1,1))) )

plt.scatter(zeroCollection.real, zeroCollection.imag, edgecolors='black', facecolors='none', marker='o', alpha=0.7)
plt.scatter(zerosCh.real, zerosCh.imag, color='red', marker='1', alpha=0.6)
plt.scatter(zerosTx.real, zerosTx.imag, marker='+', alpha=0.6, s=20)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.title(f"{mode}-Zero Constellation K-{K} for Lh-{taps}, chTapVar-{chVar}")
plt.axis('equal')
plt.savefig(f"results/ACF/FS/constellation/{mode}{taps}_{chVar}.jpeg")


plt.figure(3, dpi=800)
ax = plt.gca()

ax.axhline(y=0, color='black', linewidth=0.6, linestyle=(0, (6, 6)))
ax.axvline(x=0, color='black', linewidth=0.6, linestyle=(0, (6, 6)))

ax.add_patch( plt.Circle((0, 0), radius=1, color='gray', fill=False, linestyle=(0, (2,2))) )
ax.add_patch( plt.Circle((0, 0), radius=R, color='gray', fill=False, linestyle=(0, (1,1))) )
ax.add_patch( plt.Circle((0, 0), radius=R**-1, color='gray', fill=False, linestyle=(0, (1,1))) )

plt.scatter(zeroCollection.real, zeroCollection.imag, edgecolors='black', facecolors='none', marker='o', alpha=0.7)
plt.scatter(zerosCh.real, zerosCh.imag, color='red', marker='.', alpha=0.4, s=10)
plt.scatter(zerosTx.real, zerosTx.imag, marker='+', alpha=0.6, s=20)

plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.title(f"{mode}-Zero Constellation K-{K} for Lh-{taps}, chTapVar-{chVar}")
plt.axis('equal')
if mode == 'PZ':
    plt.ylim([-PZrad, PZrad])
    plt.xlim([-PZrad, PZrad])
else:
    plt.ylim([-1.2, 1.2])
    plt.xlim([-1.2, 1.2])    
plt.savefig(f"results/ACF/FS/constellation/{mode}Lim{taps}_{chVar}.jpeg")