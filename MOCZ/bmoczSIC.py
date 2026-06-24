"""
Implementation of BMOCZ with SIC.
Channel is slow-fading with AWGN and no rotational effect on Zeros.
Channel paramters for singleton slots will be estimated 
and used during reconstruction of packet.
Distinct channels for users have been incorporated.
Here we will be considering the normalized coefficients in time-domain.
We will be skipping FFO estimatin as there is no effect through channel.
"""

import numpy as np

from BMOCZ import (
    BMOCZReceiver,
    BMOCZTransmitter
)
from CHANNEL import (
    SlowFadingChannel,
    ChannelEstimation
)

SND_db = 10
K = 31
sig_power = 1
noise_var = sig_power * 10**(-SND_db/10)
ch_var = 1
Q = 4  # n-fft used will be Q * K = 124

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
ch = SlowFadingChannel(noise_var)
chEst = ChannelEstimation()

msgU1 = [np.random.randint(2) for i in range(K)]
msgU2 = [np.random.randint(2) for i in range(K)]

sigU1 = tx.coeffCon(msgU1)
sigU1_power = np.mean( np.abs(sigU1)**2 )
sigU1_norm = sigU1 / sigU1_power

sigU2 = tx.coeffCon(msgU2)
sigU2_power = np.mean( np.abs(sigU2)**2 )
sigU2_norm = sigU2 / sigU2_power

ch_coeffU1 = np.sqrt(ch_var/2) * ( np.random.randn() + 1j*np.random.randn() )
ch_coeffU2 = np.sqrt(ch_var/2) * ( np.random.randn() + 1j*np.random.randn() )
#  ----  Slot-1: U1 to slow-fading AWGN channel
sig_rxS1 = ch.sic_transmit([sigU1_norm], [ch_coeffU1])
#  ------ Slot-2: U1 + U2 through slow-fading AWGN channel
sig_rxS2  = ch.sic_transmit([sigU1_norm, sigU2_norm], [ch_coeffU1, ch_coeffU2])

#  --- Receiver part will be implemeted from here 
#    which include channel estimation and packet reconstruction
#  ----  Slot-1
msgS1 = rx.fftDizet(sig_rxS1, Q)
print(f"Ber for S1: {rx.ber(msgS1, msgU1)}")
# --- Slot-2
sig_reconS1 = tx.coeffCon(msgS1)
sig_reconS1_power = np.mean(np.abs(sig_reconS1)**2)
sig_reconS1 /= sig_reconS1_power
chEstS1 = chEst.leastSquares(sig_rxS1, sig_reconS1)
print(f"Channel coefficient: {ch_coeffU1}, Estimated channel coefficient: {chEstS1},\n"
        f"Absolute Error: {np.abs(ch_coeffU1-chEstS1)}")
sig_recovS2 = sig_rxS2 - chEstS1 * sig_reconS1

msgS2 = rx.fftDizet(sig_recovS2, Q)
print(f"Ber for S2: {rx.ber(msgS2,msgU2)}")