"""
We will analyse the edge cases for jutted and smooshed and template evolution
- Radius belongs to [Rs**-1, Rs]
- jutted channel zero phase [-pi/K, pi/K]
- smooshed channel zero phase [-zeta/2, zeta/2]
- pilot channel zero phase [pi-pi/K, pi+pi/K]
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import JBMOCZ, SBMOCZ, SlowFadingChannel, BMOCZ

K, PZrad = 32, 1.578
Rj, zetaJ = 1.044, 1.15
Rs, zetaS = 1.0475, 0.0545
taps, chVar = 2, 1
bmoczSys = BMOCZ(K, PZrad=PZrad)
sbmoczSys = SBMOCZ(K, zetaS, Rs=Rs)
jbmoczSys = JBMOCZ(K, zetaJ, Rj=Rj)
ch = SlowFadingChannel(noise_var=0.01, taps=taps, chVar=chVar)
MODES = ['PZ', 'S', 'J']
zeros = []

noIter = int(1e1)
mode = MODES[2]

plt.figure(1, dpi=800)
for _ in range(noIter):
    msgTx = np.random.randint(0, 2, K)
    if mode == 'PZ':
        sys = bmoczSys
        rad = sys.PZ[0]
        chZeroR = np.random.uniform(rad-0.2, rad+0.2)
        chZeroPhase = np.random.uniform(-np.pi/K, np.pi/K)
    elif mode == 'S':
        sys = sbmoczSys
        rad = sys.R
        chZeroR = np.random.uniform(rad**-1-0.2, rad+0.2)
        chZeroPhase = np.random.uniform(-sys.zeta/2, sys.zeta/2)
    elif mode == 'J':
        sys = jbmoczSys
        rad = sys.R
        chZeroR = np.random.uniform(rad**-1-0.2, rad+0.2)
        chZeroPhase = np.random.uniform(-np.pi/K, np.pi/K)
    R = sys.R
    zeroCollection = sys.zeroCollection
    zerosTx = sys.zeroSelection(msgTx)

    chZero = chZeroR * np.exp(1j*chZeroPhase)
    h = [1, -chZero**-1]
    zeros.append(chZero)

    sigTx = sys.coeffCon(msgTx)
    sigPower = np.mean(np.abs(sigTx)**2)
    sigTx /= np.sqrt(sigPower)

    sigRx = np.convolve(sigTx, h)
    sigRx_dtft, omega_dtft = sys.DTFT(sigRx)

    plt.plot(omega_dtft, np.abs(sigRx_dtft), linestyle=(0, (1, 3, 1, 2)), linewidth=0.9)
plt.xlabel("Angular Frequency(w)")
plt.ylabel("Magnitude")
plt.title(f"|DTFT of {mode}-Received Signal| K-{K} for Lh-{taps}, chTapVar-{chVar}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"results/ACF/FS/EdgeCases/{mode}Template{taps}_{chVar}.jpeg")

zeros = np.asarray(zeros, dtype=complex)
plt.figure(2, dpi=800)
ax = plt.gca()

ax.add_patch( plt.Circle((0, 0), radius=1, fill=False, color='gray', linestyle=(0, (2, 2))) )
ax.add_patch( plt.Circle((0, 0), radius=R, fill=False, color='gray', linestyle=(0, (1, 1))) )
ax.add_patch( plt.Circle((0, 0), radius=R**-1, fill=False, color='gray', linestyle=(0, (1, 1))) )

ax.axhline(y=0, color='black', linewidth=0.6, linestyle=(0, (6, 6)))
ax.axvline(x=0, color='black', linewidth=0.6, linestyle=(0, (6, 6)))

plt.scatter(zeroCollection.real, zeroCollection.imag, marker='o', edgecolor='black', facecolor='none', alpha=0.7)
plt.scatter(zeros.real, zeros.imag, marker='1', alpha=0.6, s=40, color='red')
plt.scatter(zerosTx.real, zerosTx.imag, marker='*', s=20, alpha=0.6)
plt.grid(True, alpha=0.4, linestyle='--')
plt.xlabel("real axis")
plt.ylabel("imaginary axis")
plt.title(f"channel zero distribution analysis K-{K}, taps-{taps}, chVar-{chVar}")
plt.axis('equal')
if mode != 'PZ':
    plt.ylim([-1.2, 1.2])
    plt.xlim([-1.2, 1.2])  
plt.savefig(f"results/ACF/FS/EdgeCases/{mode}{taps}_{chVar}.jpeg")