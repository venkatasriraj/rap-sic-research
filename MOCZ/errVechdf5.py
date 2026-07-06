"""
Here we will be saving the data in HDF5 format to access only necessary data
"""
import numpy as np
import h5py
from BMOCZ import (
    BMOCZReceiver,
    BMOCZTransmitter
)
from CHANNEL import SlowFadingChannel

K = 32
Q = 2
SNR_dB = np.arange(-10, 21, 1)
signal_power = 1
noIter = int(1e5)

tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)

with h5py.File("simulation.h5", "w") as f:
    param = f.create_group("parameters")
    param.attrs["K"] = K
    param.attrs["Q"] = Q
    param.attrs["noIter"] = noIter
    param.attrs["SNR_dB"] = SNR_dB
    data = f.create_group("data")
    for snr in SNR_dB:
        snr_group = data.create_group(f"{snr}")
        msg_ds = snr_group.create_dataset(
            "msg",
            shape = (noIter, K),
            dtype = np.uint8
        )
        decoded_msg_ds = snr_group.create_dataset(
            "decoded_msg",
            shape = (noIter, K),
            dtype = np.uint8
        )
        error_vec_ds = snr_group.create_dataset(
            "error_vec",
            shape = (noIter, K),
            dtype = np.uint8
        )
        noise_var = signal_power * 10**(-snr/10)
        ch = SlowFadingChannel(noise_var)
        for i in range(noIter):
            msg = np.random.randint(0, 2, K, dtype=np.uint8)
            
            sig_tx = tx.coeffCon(msg)
            sig_power = np.mean(np.abs(sig_tx)**2)
            sig_tx /= np.sqrt(sig_power)

            sig_rx, _ = ch.transmit(sig_tx)

            sig_ffo = rx.ffoEstCor(sig_rx, Q)
            decoded_msg = rx.fftDizet(sig_ffo, Q)
            error_vec = msg ^ decoded_msg

            msg_ds[i] = msg
            error_vec_ds[i] = error_vec
            decoded_msg_ds[i] = decoded_msg
            # print(f"Message         : {msg},\n Message Decoded: {decoded_msg},\n Error Vector:    {error_vec}")
        print(f"SNR: {snr} done")