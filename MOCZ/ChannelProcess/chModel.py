"""
Here we will be computing the channel statistics to characterize the channel
"""
import numpy as np
import pickle
import h5py

# with open("results.pkl", "rb") as f:
#     results = pickle.load(f)
# print(results)
f = h5py.File("simulation.h5", "r")

K = 32
SNR_dB = np.arange(-10, 21, 2)
noIter = int(1e5)  # no need for this
# error_vec = f["data"]["-10"]["error_vec"]
# x = np.array(error_vec[:])
# print(len(x.mean(axis=0)))
ber = {}; # it contains ber wrt each SNR
perBitBER = {}; # contains 32 length array of per bit ber wrt each SNR
errorWt = {} # contains errors per packet stats
covMatrix = {}
corrMatrix = {}
condProb = {}

for snr in SNR_dB:
    # m = f["data"][f"{snr}"]["msg"]
    # d = f["data"][f"{snr}"]["decoded_msg"]
    e = np.array(f["data"][f"{snr}"]["error_vec"][:])
    ber_ds = result.create_group(f"{snr}")
    
