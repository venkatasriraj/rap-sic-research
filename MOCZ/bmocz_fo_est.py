"""
We will be integrating the BMOCZ with Integer frequency offset and recover the signal back.
The actual information to be transmitted is of size: 16 bits
The code-word transmitted is of size: 31 bits
"""
from utils import *

import numpy as np
import random
import galois
import itertools

m = 5
m = 5
n = 2**m -1

t = 2
d_min = 2 * t + 1

GF32 = galois.GF(2**m)
x = galois.Poly.Identity(galois.GF2)

one = galois.GF2(1)
x_n = x**n - one

factors_32, _ = galois.factors(x_n)

cyclotomic_coset = con_coset(n)

con_power = np.arange(1, 2*t+1)

g_bch = one

g_coset = np.array([])

for i in con_power:
    leader = [keys for keys, arr in cyclotomic_coset.items() if i in arr ]
    g_coset = np.append(g_coset, leader).astype(int)

g_bch_coset = np.unique(g_coset)

primitive_element = GF32.primitive_element

for i in g_bch_coset:
    g_bch *= GF32.minimal_poly(primitive_element**i)

g_bch_factors, _ = galois.factors(g_bch)

gin_choice = [i for i in factors_32[1:] if i not in g_bch_factors]
gin = gin_choice[2]

g = gin * g_bch

k_out = n - g_bch.degree
# print(f'Length of BCH message: {k_out}')
k = g.degree
# print(f"Degree of generator(G) polynomial: {k}")
B = n - k
# print(f"Length of ACPC code (or message bits): {B}")

G, H = con_systematic(x, n, B, g)

G_bch, H_bch = con_systematic(x, n, k_out, g_bch)

T_syn = con_Tsyndrome(H_bch, t)

e_affine = galois.GF2.Zeros(k_out)
e_affine[0] = 1

# affine codeword construction to add after mG
g_affine = e_affine @ G_bch

msg = galois.GF2.Random(B)
print(f'Message to be transmitted: {msg}')

codeword_gen = msg @ G + g_affine
print(f"Codeword to be transmitted: {codeword_gen}")

## after the code codeword generation we will be integrating the integer frequency 
## estimation with BMOCZ Base Code

## from the base code n = K

x = coeffCon(codeword_gen)

sigma = 1/np.sqrt(2)

h_com = sigma * ( np.random.randn() + 1j * np.random.randn() )

h_mag = np.abs(h_com)
h_phase = 180 * np.angle(h_com)/(np.pi)
print(f'Attenuation: {np.round(h_mag, 6)}, rotation: {np.round(h_phase, 6)}(in degrees) applied by channel')

y_com = x * h_com

Q = int( 2**( np.log( np.ceil(len(y_com/n)) ) / np.log(2) ) )
R = np.sqrt(1 + np.sin(np.pi/n))

zero_geometry = codebook_con(R, n)


theta_est = ffo_est(y_com, Q, n, R)
print(f'Estiamted fractional frequency rotation: {theta_est}')

# fractional offset correction
M_theta = np.diag( np.exp(-1j*theta_est) ** np.arange(len(y_com)) )
# print(M_theta)
y_ffo = y_com @ M_theta
# print(y_ffo)

#  Ro = np.sqrt(1 + np.sin(np.pi/K))
msg_RxS1 = fftDizet(y_ffo, n, Q, R)     
v = np.array(msg_RxS1, dtype=int)
v = galois.GF2(v)

s = v @ H_bch.T
# print(f'Syndrome identified as {s}')
s_tuple = tuple(s.tolist())
error_vec = T_syn[s_tuple]
print(f"Corresponding error vector: {error_vec}")

v_hat = v - error_vec
print(f"Corrected codeword: {v_hat}")

for i in range(len(v_hat)):
    c_hat = np.roll(v_hat, -i) - g_affine
    # print(f"Message Recovered: {i}, {c_hat}")
    if np.all(c_hat @ H.T == 0):
        l_est = i
        msg_hat = c_hat[-B:]
        break
    
print(f"Codeshifted by {l_est} sectors")
print(f"Message Recovered: {msg_hat}")

ber = np.mean(msg != msg_hat)
print(f'BER: {ber}')