"""
Implementation of Channel Estimation using MOCZ.
We will not be estiming the phase since the channel is slow-fading and 
there will be no effect of channel on Zeros.
Zero rotation can only be observed in case of CFO.
"""

import numpy as np
import galois

from BMOCZ import (
    BMOCZReceiver,
    BMOCZTransmitter
) 
from CHANNEL import (
    SlowFadingChannel,
    ChannelEstimation
) 

SNR_db = 20
K = 16

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)

msg = [np.random.randint(2) for i in range(K)]

sig_tx = tx.coeffCon(msg) 
# signal is not normalized and the coefficients are of the order 
# [x0, x1, x2, x3, ....., xn]

sig_power = np.mean( np.abs(sig_tx)**2 )
sig_norm = sig_tx / sig_power

noise_var = sig_power * 10**(-SNR_db/10)

ch = SlowFadingChannel(noise_var)
sig_rx, ch_coeff = ch.transmit(sig_tx)

Q = int( 2**( np.log( np.ceil(len(sig_rx)/K) ) / np.log(2) ) )

sig_ffo = rx.ffoEstCor(sig_rx, Q)
msg_rx = rx.fftDizet(sig_ffo, Q)
ber = rx.ber(msg_rx, msg)
print(f'BER: {ber}')
## we need to identify the papr after applying the FFT
papr_rx = rx.PAPR(sig_rx)
print(f"The PAPR at Receiver is {np.abs(papr_rx)}")

# ----   Signal Reconstruction using the same BMOCZTransmitter Class  -----
sig_recon = tx.coeffCon(msg_rx)
chEst = ChannelEstimation()
ch_coeff_hat = chEst.leastSquares(sig_rx, sig_recon)
print(f"The channel coefficient is {ch_coeff}\n")
print(f"The estimated channel coefficient is {ch_coeff_hat}\n")
print(f"Absolute estimation error is {np.abs(ch_coeff - ch_coeff_hat)}\n")    