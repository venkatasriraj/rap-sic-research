"""
Evaluating the polynomial on the inner and outer circles 
for a given message
"""
import numpy as np
import matplotlib.pyplot as plt
from wirelessComm import JBMOCZ, SBMOCZ, SlowFadingChannel, BMOCZ, JUTTEDfs
from matplotlib.patches import Wedge, Arc

K, PZrad = 31, 1.578
Rj, zetaJ = 1.044, 1.15
Rs, zetaS = 1.0475, 0.0545
taps, chVar = 1, 0.95
Peak = [0, 7, 17, 23]
bmoczSys = BMOCZ(K, PZrad=0)
sbmoczSys = SBMOCZ(K, zetaS, Rs=Rs)
jbmoczSys = JBMOCZ(K, zetaJ, Rj=Rj, mPeak=[0])
ch = SlowFadingChannel(noise_var=0.01, taps=taps, chVar=chVar)
jSystem = JUTTEDfs(K, zetaJ, Rj=Rj, multiplePeak=True)
jbmcozSystem = JBMOCZ(K, zetaJ, Rj=Rj, mPeak=Peak)
MODES = ['PZ', 'S', 'J', 'MJ', 'DiZeT-CFO']
zeros = []
mode = MODES[3]

plt.rcParams.update({
    "font.family": "serif",
    # "font.size": 26,
    "axes.labelsize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    # "legend.fontsize": 25,
    # "axes.titlesize": 20,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
msgTx = np.random.randint(0, 2, K)
rad = 0
plt.figure(1, dpi=800)
if mode == 'PZ':
    sys = bmoczSys
    # rad = sys.PZ[0]
elif mode == 'S':
    sys = sbmoczSys
elif mode == 'J':
    sys = jbmoczSys
elif mode == 'MJ':
    sys = jbmcozSystem
elif mode == 'DiZeT-CFO':
    sys = jSystem
R = sys.R 
zeroCollection = sys.zeroCollection
zerosTx = sys.zeroSelection(msgTx)

sigTx = sys.coeffCon(msgTx)
sigPower = np.mean(np.abs(sigTx)**2)
sigTx /= np.sqrt(sigPower)

sigTx_dtft_oc, omega_dtft = sys.DTFT(sigTx, Mode='oc')

plt.plot(omega_dtft, np.abs(sigTx_dtft_oc), linestyle='-.', linewidth=1.2, color='black')
plt.xlim(-np.pi, np.pi)
plt.xticks(
    [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$|Y(Re^{jw})|$')
# plt.title(f"|DTFT of {mode}-Received Signal| K-{K} for Lh-{taps}, chTapVar-{chVar}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"results/ACF/{mode}Template{taps}_{chVar}_oc.pdf", bbox_inches="tight")


plt.figure(2, dpi=800)
sigTx_dtft_ic, omega_dtft = sys.DTFT(sigTx, Mode='ic')

plt.plot(omega_dtft, R**(len(sigTx)-1) * np.abs(sigTx_dtft_ic), linestyle='-.', linewidth=1.2, color='black')
plt.xlim(-np.pi, np.pi)
plt.xticks(
    [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$|Y(R^{-1}e^{jw})|$')
# plt.title(f"|DTFT of {mode}-Received Signal| K-{K} for Lh-{taps}, chTapVar-{chVar}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"results/ACF/{mode}Template{taps}_{chVar}_ic.pdf", bbox_inches="tight")


k_pair = 0                       
thK = 2 * np.pi / K     

plt.figure(3, dpi=800)
ax = plt.gca()

ax.add_patch( plt.Circle((0, 0), radius=1, fill=False, color='black', linestyle=(0, (2, 2))) )
ax.add_patch( plt.Circle((0, 0), radius=R, fill=False, color='gray', linestyle=(0, (1, 1))) )
ax.add_patch( plt.Circle((0, 0), radius=R**-1, fill=False, color='gray', linestyle=(0, (1, 1))) )

ax.axhline(y=0, color='black', linewidth=0.6, linestyle=(0, (6, 6)))
ax.axvline(x=0, color='black', linewidth=0.6, linestyle=(0, (6, 6)))

plt.scatter(zeroCollection.real, zeroCollection.imag, marker='o', edgecolor='black', facecolor='none', alpha=0.9, s=45)
plt.scatter(zerosTx.real, zerosTx.imag, marker='*', color='tab:blue', alpha=0.9, s=45)
# ---- highlighted zero pair: alpha_k^0 (inner, 1/R) and alpha_k^1 (outer, R)
a_k = k_pair * thK

z0 = (R*zetaJ)**-1 * np.exp(1j * a_k)
z1 = (R*zetaJ) * np.exp(1j * a_k)
plt.scatter([z0.real], [z0.imag], marker='o', s=40, color='tab:purple', zorder=6, facecolor='none')
plt.scatter([z1.real], [z1.imag], marker='o', s=40, color='tab:red', zorder=6, facecolor='none')
u = np.exp(1j * a_k)
ax.annotate(rf'$\alpha_{{{k_pair}}}^{{0}}$', xy=(z0.real, z0.imag), xytext=(-35, 6), textcoords='offset points',
            fontsize=28, color='tab:blue')
ax.annotate(rf'$\alpha_{{{k_pair}}}^{{1}}$', xy=(z1.real, z1.imag), xytext=(12, 8), textcoords='offset points',
            fontsize=28, color='tab:red')

# ---- sector between the rays of pair k and pair k+1 (angle theta_K)
a_k = (k_pair +4) * thK
a_n = (k_pair + 5) * thK
ax.plot([0, 1.12 * R * np.cos(a_k)], [0, 1.12 * R * np.sin(a_k)], color='k', lw=0.7, ls='--')
ax.plot([0, 1.12 * R * np.cos(a_n)], [0, 1.12 * R * np.sin(a_n)], color='k', lw=0.7, ls='--')
ax.add_patch(Wedge((0, 0), 1.12 * R, np.degrees(a_k), np.degrees(a_n), facecolor='orange', alpha=0.18, edgecolor='none'))
ax.add_patch(Arc((0, 0), 2 * 0.6, 2 * 0.6, theta1=np.degrees(a_k), theta2=np.degrees(a_n), color='k', lw=1.0))
am = (a_k + a_n) / 2
ax.text(0.72 * np.cos(am), 0.72 * np.sin(am), r'$\theta_K$', fontsize=32, ha='center', va='center')

plt.grid(True, alpha=0.4, linestyle='--')
plt.xlabel("real axis")
plt.ylabel("imaginary axis")
# plt.title(f"channel zero distribution analysis K-{K}, taps-{taps}, chVar-{chVar}")
plt.axis('equal')
# if mode != 'PZ':
#     plt.ylim([-1.2, 1.2])
#     plt.xlim([-1.2, 1.2])  
plt.savefig(f"results/ACF/{mode}{taps}_{chVar}.pdf", bbox_inches='tight')



plt.figure(4, dpi=800)
sigTx_dtft_uc, omega_dtft = sys.DTFT(sigTx, Mode='uc')
plt.plot(omega_dtft, np.abs(sigTx_dtft_uc), linestyle='-.', linewidth=1.2, color='black')
plt.xlim(-np.pi, np.pi)
plt.xticks(
    [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$|X(e^{jw})|$')
# plt.title(f"|DTFT of {mode}-Received Signal| K-{K} for Lh-{taps}, chTapVar-{chVar}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"results/ACF/{mode}Template{taps}_{chVar}_uc.pdf", bbox_inches="tight")


plt.figure(5, dpi=800)
plt.plot(omega_dtft, np.abs(sigTx_dtft_uc), linestyle='-.', linewidth=1.2, label=r'$|Y(e^{jw})|$')
plt.plot(omega_dtft, np.abs(sigTx_dtft_oc), linestyle='--', linewidth=1.2, label=r'$|Y(Re^{jw})|$')
plt.plot(omega_dtft, R**(len(sigTx)-1) * np.abs(sigTx_dtft_ic), linestyle=':', linewidth=1.2, label=r'$|Y(R^{-1}e^{jw})|$')
plt.xlim(-np.pi, np.pi)
plt.xticks(
    [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
    [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
)
plt.xlabel(r'$\omega$')
plt.ylabel(r'$|Y(re^{jw})|$')
# plt.title(f"|DTFT of {mode}-Received Signal| K-{K} for Lh-{taps}, chTapVar-{chVar}")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', framealpha=0.5, fontsize=18 )
plt.savefig(f"results/ACF/{mode}Template{taps}_{chVar}.pdf", bbox_inches="tight")