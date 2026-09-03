"""
We will be implementing the ACPC code for integer rotation offset estimation 
for BMOCZ in case of CFO. 
Uniform random rotation will be applied to the transmitted signal with 
the help of MultiPathFading class.
"""
import numpy as np
import itertools
import galois
# import matplotlib.pyplot as plt
# from wirelessComm import (
#     BMOCZTransmitter, BMOCZReceiver, 
#     ACPC, MultiPathFading
# )
# signal_power = 1
# m = 5
# K = 2**m-1
t = 3
print(np.arange(1, 1))