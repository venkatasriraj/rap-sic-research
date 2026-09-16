"""
Simulation analysis of JBMOCZ point-to-point communication link.

"""
import numpy as np
from wirelessComm import JBMOCZ, MultiPathFading, PerformanceParameters

zeta = 1.2
K = 8
perParam = PerformanceParameters()
jbmcozSystem = JBMOCZ(K, zeta)

msgTx = np.random.randint(0, 2, K)
sigTx = jbmcozSystem.coeffCon(msgTx)

# sigRxCorrected, phi_hat = jbmcozSystem.rotationEst(sigTx)
# print(f'Est Rotation: {phi_hat}')
msgRx = jbmcozSystem.fftDiZeT(sigTx)
ber = perParam.ber(msgRx, msgTx)
print(f"BER: {ber}")
print(f"Msg Tx: {msgTx} \nMsg Rx: {msgRx}")