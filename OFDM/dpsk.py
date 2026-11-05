"""
Channel estimation analysis of DPSK - Differnetial BPSK 
"""
import numpy as np
from wirelessComm import SlowFadingChannel, ChannelEstimation, DPSK, PerformanceParameters

K = 4
energy = K + 1
SNR_dB = 15
ch = SlowFadingChannel(noise_var=10**(-SNR_dB/10))
chEst = ChannelEstimation()
dpsk = DPSK()
perParam = PerformanceParameters()

msgU1 = np.random.randint(0, 2, K)
msgU2 = np.random.randint(0, 2, K)

sigU1 = dpsk.transmitterRAP(msgU1, energy)
sigU2 = dpsk.transmitterRAP(msgU2, energy)

ch_coeffU1 = np.sqrt(1/2) * ( np.random.randn() + 1j*np.random.randn() )
ch_coeffU2 = np.sqrt(1/2) * ( np.random.randn() + 1j*np.random.randn() )

#  ----  Slot-1: U1 to slow-fading AWGN channel
sig_rxS1 = ch.sic_transmit([sigU1], [ch_coeffU1])
#  ------ Slot-2: U1 + U2 through slow-fading AWGN channel
sig_rxS2 = ch.sic_transmit([sigU1, sigU2], [ch_coeffU1, ch_coeffU2])

msgS1, _ = dpsk.receiverRAP(sig_rxS1)
ber = perParam.ber(msgS1, msgU1)
print(f'BER - U1: {ber}')

sigReconS1 = dpsk.transmitterRAP(msgS1, energy)

chEstS1 = chEst.leastSquares(sig_rxS1, sigReconS1)
print(f'Tx: {msgU1}\n')
print(f'Sig Tx: {sigU1}\nSig Rx: {sig_rxS1}')
coeff_error = np.abs(chEstS1 - ch_coeffU1)
con_error = perParam.mae(sig_rxS1, sigReconS1 * chEstS1) / np.mean( np.abs(sig_rxS1) )
print(f"h: {ch_coeffU1}, h_est: {chEstS1}, MAE: {np.abs(ch_coeffU1-chEstS1)}")
print(f'Con Error: {con_error}')
sig_recovS2 = sig_rxS2 - chEstS1 * sigReconS1

msgS2, _ = dpsk.receiverRAP(sig_recovS2)
ber = perParam.ber(msgS2, msgU2)
print(f'BER - U2: {ber}')