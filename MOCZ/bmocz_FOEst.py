"""
Implementation of Interger and fractional phase offset estimation.
"""

import numpy as np
import galois

from BMOCZ import (
    BMOCZReceiver,
    BMOCZTransmitter,
    ACPC
) 
from CHANNEL import SlowFadingChannel

SNR_db = 20
sig_power = 1 # signal will be normalized
noise_var = sig_power * 10**(-SNR_db/10)
m = 5
K = 2**m-1
t = 2 # number of error correcting bits

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
ch = SlowFadingChannel(noise_var)
# we will be implementing ACPC(31, 21,,5)
acpc = ACPC(m)
gen_poly, bch_gen_poly = acpc.gen_poly(t)

gen_degree = gen_poly.degree
bch_gen_degree = bch_gen_poly.degree
# print(f"Generator polyniomial degree is {gen_degree}")
B = K - gen_degree
k_outer = K - bch_gen_degree

G, H = acpc.con_systematic(B, gen_poly)
G_bch, H_bch = acpc.con_systematic(k_outer, bch_gen_poly)

aff_trans = galois.GF2.Zeros(G_bch.shape[0])
aff_trans[0] = 1
g_affine = aff_trans @ G_bch

# will be looping from here to perform monte-carlo simulations
msgB = np.array([np.random.randint(2) for i in range(B)])
msg_encoded = acpc.msg_encoding(G, g_affine, msgB)

sig_tx = tx.coeffCon(msg_encoded) 
# signal is not normalized and the coefficients are of the order 
# [x0, x1, x2, x3, ....., xn]
sig_power = np.mean( np.abs(sig_tx)**2 )
sig_norm = sig_tx / sig_power

sig_rx = ch.transmit(sig_norm)

# Q = int( 2**( np.log( np.ceil(len(sig_rx)/K) ) / np.log(2) ) )
Q = 4
sig_ffo = rx.ffoEstCor(sig_rx,Q)
codeword_rx = rx.fftDizet(sig_ffo, Q)

msg_est, l_est = acpc.codeword_decoding(H, H_bch, codeword_rx, g_affine, t)

ber = rx.ber(msg_est, msgB)
print(f'BER: {ber}')