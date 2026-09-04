"""
Monte-carlo Simulation to verify reduction in PAPR by a insertion of pilot-zero
"""
import numpy as np
import itertools
import matplotlib.pyplot as plt
from wirelessComm import BMOCZ

K = np.arange(1, 10)
# Rzm = [-1, 1j, -1j]
Rzm = [-2, -1, 1j, -1j]
PAPR = {}; PAPR_PZ = {}; PAPR_Reduction = {}

for k in K:
    papr_nopz_max, papr_pz_max = 0, 0
    tx = BMOCZ(k)
    for bits in itertools.product([0,1], repeat=k):
        msg = np.array(bits)

        sig_nopz = tx.coeffCon(msg)
        sig_pz = tx.coeffCon(msg, Rzm)

        papr_nopz = tx.PAPR(sig_nopz)
        papr_pz = tx.PAPR(sig_pz)

        papr_nopz_max = papr_nopz if papr_nopz > papr_nopz_max else papr_nopz_max
        papr_pz_max = papr_pz if papr_pz > papr_pz_max else papr_pz_max
    PAPR[k] = papr_nopz_max
    PAPR_PZ[k] = papr_pz_max
    PAPR_Reduction[k] = ( papr_nopz_max - papr_pz_max ) * 100 / papr_nopz_max
    print(f"Block-length {k} done")

plt.figure(1, dpi=800)
plt.plot(PAPR_Reduction.keys(), PAPR_Reduction.values(), linestyle='-', linewidth=0.9)
plt.xlabel("Block-Length(K)")
plt.ylabel(f"% reduction of PAPR by pilot-zero")
plt.title(f"% reduction of PAPR by PZ vs K for PZ at {Rzm}")
plt.grid(True, alpha=0.6, linestyle='--')
plt.savefig(f"results/PilotZero/PAPR/{Rzm}.jpeg")
# plt.show()