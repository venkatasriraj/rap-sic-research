"""
In this code we are checking whether the samples after modulation will be same 
as modulated samples pass through rc filter and sampler to achieve the maximum SNR
"""

import numpy as np
import csv
import binascii
import sys
import glob
import re
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import random



ACCESS_CODE_BYTES = [0xE1, 0x5A, 0xE8, 0x93]
AC_THRESHOLD      = 30        # bit matches out of 32
MAX_DEGREE        = 16
CRC_BYTES         = 4
FILLER_LEN        = 8
PACKET_SIZE       = 100
# AC(4)+uid(2)+seq(2)+deg(1)+slot(1)+slot_list(16)+CRC(4)+filler(8) = 38
HEADER_FIXED      = 4 + 2 + 2 + 1 + 1 + MAX_DEGREE + CRC_BYTES + FILLER_LEN
PAYLOAD_LEN       = PACKET_SIZE - HEADER_FIXED        # = 62

SPS               = 1
SPS_RRC           = 5
RRC_ALPHA         = 0.350
RRC_NUM_TAPS      = 55   # 11 * sps
DIFF_INIT_STATE   = 0   # silence (0x00) precedes each packet from PDU_to_Timed_Byte_Stream


# ─────────────────────────────────────────────────────────────────────────────
#  DSP helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_rrc_taps(sps=SPS_RRC, alpha=RRC_ALPHA, num_taps=RRC_NUM_TAPS):
    taps   = np.zeros(num_taps)
    center = (num_taps - 1) / 2.0
    for i in range(num_taps):
        t = (i - center) / sps
        if t == 0.0:
            taps[i] = 1.0 + alpha * (4.0 / np.pi - 1.0)
        elif abs(t) == 1.0 / (4.0 * alpha):
            taps[i] = (alpha / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
            )
        else:
            num = (np.sin(np.pi * t * (1 - alpha)) +
                   4.0 * alpha * t * np.cos(np.pi * t * (1 + alpha)))
            den = np.pi * t * (1 - (4.0 * alpha * t) ** 2)
            taps[i] = num / den
    taps /= np.sqrt(np.sum(taps ** 2))
    return taps.astype(np.float32)


_RRC_TAPS = make_rrc_taps()


def bytes_to_bits(byte_list):
    bits = []
    for b in byte_list:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return np.array(bits, dtype=np.uint8)


def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    n    = (len(bits) // 8) * 8
    return [int(''.join(map(str, bits[i:i+8])), 2) for i in range(0, n, 8)]


def differential_encode(bits, init_state=DIFF_INIT_STATE):
    enc  = np.empty(len(bits), dtype=np.uint8)
    prev = init_state
    for i, b in enumerate(bits):
        enc[i] = int(b) ^ prev
        prev   = enc[i]
    return enc


def differential_decode(bits, init_state=DIFF_INIT_STATE):
    bits = np.asarray(bits, dtype=np.uint8)
    prev = np.empty(len(bits), dtype=np.uint8)
    prev[0]  = init_state
    prev[1:] = bits[:-1]
    return (bits ^ prev).astype(np.uint8)


def bpsk_modulate(bits):
    return (1.0 - 2.0 * np.asarray(bits, dtype=np.float32)).astype(np.complex64)


def bpsk_demodulate(symbols):
    return (symbols.real < 0).astype(np.uint8)

def build_template_RRC(init_state=DIFF_INIT_STATE):
    bits    = bytes_to_bits(ACCESS_CODE_BYTES)
    enc     = differential_encode(bits, init_state)
    symbols = bpsk_modulate(enc)   # 32 complex symbols, 1 per bit
    symbols /= np.max(np.abs(symbols))
    # return symbols
    up      = np.zeros(len(symbols) * SPS_RRC, dtype=np.complex64)
    up[::SPS_RRC] = symbols
    # Apply RRC twice (TX + RX matched filter)
    filtered = np.convolve(up, _RRC_TAPS, mode='full')
    filtered = np.convolve(filtered, _RRC_TAPS, mode='full').astype(np.complex64)
    # filtered /= np.max(np.abs(filtered))
    return filtered

def build_template(init_state=DIFF_INIT_STATE):
    bits    = bytes_to_bits(ACCESS_CODE_BYTES)
    enc     = differential_encode(bits, init_state)
    symbols = bpsk_modulate(enc)   # 32 complex symbols, 1 per bit
    symbols /= np.max(np.abs(symbols))
    return symbols


tem  = build_template()
tem_RRC = build_template_RRC()

# Calculate the group delay to find the peak SNR sampling index
delay = RRC_NUM_TAPS - 1  # 55 - 1 = 54
num_symbols = len(tem)    # 32 symbols

# Complete the sampling logic: [start : stop : step]
temp_RRC_sam = tem_RRC[delay : delay + (num_symbols * SPS_RRC) : SPS_RRC]

# Check if the sampled filtered symbols match the original symbols
# (Using .real because BPSK information is on the real axis)

# match_check = np.allclose(tem.real, temp_RRC_sam.real, atol=1e-3)
match_check = np.array_equal((tem.real > 0), (temp_RRC_sam.real > 0))
print(f"Do the sampled results match the original modulated symbols? {match_check}")

# ─────────────────────────────────────────────────────────────────────────────
#  Stem Plotting
# ─────────────────────────────────────────────────────────────────────────────

plt.figure(figsize=(10, 6))

# Plot 1: Original Modulated Symbols
plt.subplot(2, 1, 1)
# BPSK symbols are complex, but the data is on the real axis.
plt.stem(tem.real, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.title("Original Modulated BPSK Symbols")
plt.ylabel("Amplitude")
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Downsampled RX Symbols (After TX + RX Filters)
plt.subplot(2, 1, 2)
plt.stem(temp_RRC_sam.real, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title("Sampled Symbols (Matched Filter Output at Peak SNR)")
plt.xlabel("Symbol Index")
plt.ylabel("Amplitude")
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("my_plot.png")