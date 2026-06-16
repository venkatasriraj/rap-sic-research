"""
BMOCZ implementation without interger frequency offset recovery
"""
from utils import *

import numpy as np
# import matplotlib.pyplot as plt

K = 2 ** np.random.randint(1, 4)
print(f'Number of message bits to be transmitted: {K}')

message = [np.random.randint(2) for i in range(K)]
print(f'The message to be transmitted: {message}')

x = coeffCon(message)
print(f"The transmitted sequence will be: {x}")

# channel
p = np.random.uniform(0,1)
print(f'Complex channel variance: {p}')
h_com = np.sqrt(p/2)*np.random.randn(1) + 1j*np.sqrt(p/2)*np.random.randn(1) 
print(f'Flat fading channel coefficient: {h_com}')
h_mag = np.abs(h_com)
h_phase = np.angle(h_com)/(2*np.pi)
print(f'Attenuation: {np.round(h_mag, 6)}, rotation: {np.round(h_phase, 6)}(in degrees) applied by channel')

y_com = x * h_com
print(f'Received sequence through complex slow fading channel without noise: {y_com}')

Q = int( 2**( np.log( np.ceil(len(y_com)/K) ) / np.log(2) ) )
R = np.sqrt(1 + np.sin(np.pi/K))

# codebook construction at RX for REFERENCE
zero_geometry = codebook_con(R, K)

theta_est = ffo_est(y_com, Q, K, R)

# fractional offset correction
M_theta = np.diag( np.exp(-1j*theta_est) ** np.arange(len(y_com)) )
# print(M_theta)
y_ffo = y_com @ M_theta
# print(y_ffo)

#  Ro = np.sqrt(1 + np.sin(np.pi/K))
msg_RxS1 = fftDizet(y_ffo, K, Q, R)     
msg_RxS1 = np.array(msg_RxS1, dtype=int)
# print(msg_RxS1)

ber = np.mean(message != msg_RxS1)
print(f'BER: {ber}')
print(message)