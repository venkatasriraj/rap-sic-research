"""
In this code we will be processing the baseband I-Q samples from RC filter (Matched filter).
We will be modifying the functions from the sic_offset.py to achieve our objective to cancel the singleton from interfereced slot.

SPS = 5. We will be performing the correlation, reconstructing the packet using the mentioned sps value.

Here in slots there is a possibility of slot offset for that we are 
correlate with the access code and identifying the possible start of the packet in the given slot.

The offset complicates the interference cancellation since it doesn't know where the packet boundaries are in the interfered slot.


Combined offline IRSA processor:
  1. Frame Sync  — sliding window IQ correlation to find frame boundaries
  2. SIC         — successive interference cancellation per frame

Input  : rc_samples.bin  (complex64 raw IQ from GNU Radio File Sink)
Output : decoded_packets.csv
         sic_log.csv

Usage
─────
  python irsa_offline.py costas_out.bin [N_slots] [slot_samples]

  # defaults: N_slots=2, slot_samples=()
"""

import numpy as np
import csv
import binascii
import sys
import glob
import re
from scipy.signal import find_peaks
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
#  Constants — must match TX chain
# ─────────────────────────────────────────────────────────────────────────────

ACCESS_CODE_BYTES = [0xE1, 0x5A, 0xE8, 0x93]
FILLER_BYTES      = [0xDE, 0xAD, 0xBE, 0xEE, 0xDE, 0xAD, 0xBE, 0xEE]
AC_THRESHOLD      = 30        # bit matches out of 32
MAX_DEGREE        = 16
CRC_BYTES         = 4
FILLER_LEN        = 8
PACKET_SIZE       = 100
# AC(4)+uid(2)+seq(2)+deg(1)+slot(1)+slot_list(16)+CRC(4)+filler(8) = 38
HEADER_FIXED      = 4 + 2 + 2 + 1 + 1 + MAX_DEGREE + CRC_BYTES + FILLER_LEN
PAYLOAD_LEN       = PACKET_SIZE - HEADER_FIXED        # = 62

SPS               = 5
RRC_ALPHA         = 0.350
RRC_NUM_TAPS      = 55    # 11 * sps  ----- WH?
DIFF_INIT_STATE   = 0   # silence (0x00) precedes each packet from PDU_to_Timed_Byte_Stream


# ─────────────────────────────────────────────────────────────────────────────
#  DSP helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_rrc_taps(sps=SPS, alpha=RRC_ALPHA, num_taps=RRC_NUM_TAPS):
    taps = np.zeros(num_taps)
    center = (num_taps - 1) / 2.0
    for i in range(num_taps):
        t = (i - center) / sps
        if t == 0.0:
            taps[i] = 1.0 + alpha * (4.0 / np.pi - 1.0)
