"""
Here before identifying the ECC for MOCZ, we need to identify the channel model for MOCZ
- binary symmetric channel
- binary erasure channel
- gilbert-elliott channel
- burst-errors channel

for that we need the error vectors and messages to compute the channel statistics
which will be computed here
"""
import numpy as np
import pickle
import json
from BMOCZ import (
    BMOCZReceiver,
    BMOCZTransmitter
)
from CHANNEL import (
    SlowFadingChannel,
)

K = 32
Q = 2
SNR_dB = np.arange(-10, 21, 5)
signal_power = 1
noIter = 10000
tx = BMOCZTransmitter(K)
rx = BMOCZReceiver(K)
data = {}; messages = {}; error_vec = {}; decoded_msg = {}
for snr in SNR_dB:
    messages_snr = {}; error_vec_snr = {}; decoded_msg_snr = {} 
    data_snr = {}
    noise_var = signal_power * 10**(-snr / 10)
    ch = SlowFadingChannel(noise_var)
    for i in range(noIter):
        pkt = {}
        msg = np.array([np.random.randint(2) for _ in range(K)])

        sig_tx = tx.coeffCon(msg)
        sig_power = np.mean( np.abs(sig_tx)**2 )
        sig_tx /= np.sqrt(sig_power)

        sig_rx, _ = ch.transmit(sig_tx)

        sig_ffo = rx.ffoEstCor(sig_rx, Q)
        msg_decoded = rx.fftDizet(sig_ffo, Q)

        err_vec = msg ^ msg_decoded
        # print(f"Message         : {msg},\n Message Decoded: {msg_decoded},\n Error Vector:    {err_vec}")
        pkt["msg"] = msg
        pkt["err_vec"] = err_vec
        pkt["decoded_msg"] = msg_decoded
        data_snr[i] = pkt
        # messages_snr[i] = msg
        # error_vec_snr[i] = err_vec
        # decoded_msg_snr[i] = msg_decoded
    data[snr] = data_snr
    # messages[snr] = messages_snr
    # error_vec[snr] = error_vec_snr
    # decoded_msg[snr] = decoded_msg_snr

# with open("results.pkl", "wb") as f:
#     pickle.dump(data, f)