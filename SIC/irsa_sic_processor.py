"""
irsa_sic_processor.py
─────────────────────
Offline IRSA Successive Interference Cancellation (SIC) processor.

Input  : Frame-synced IQ samples (complex64 numpy array or .npy file)
         — output of Costas loop, already aligned to frame start.
Output : decoded_packets.csv   — one row per successfully decoded packet
         sic_log.csv           — iteration-level SIC trace

Assumptions (match your GNU Radio chain)
─────────────────────────────────────────
  Modulation    : BPSK (1 bit/symbol)
  Encoding      : Differential (DBPSK), init state = 1
  SPS           : 10
  RRC alpha     : 0.350
  RRC taps      : 110
  Packet size   : 100 bytes  (matches IRSA_Packet_Generator)
  N_slots       : set at runtime
  slot_samples  : slot_duration * fs = 0.4 * 40000 = 16000

Packet byte layout (IRSA_Packet_Generator):
  [access_code 4B][user_id 2B][frame_seq 2B][degree 1B][this_slot 1B]
  [slot_list 16B][random_data NB][CRC32 4B][filler 8B]
"""

import numpy as np
import csv
import time
import binascii
import os
from copy import deepcopy


# ─────────────────────────────────────────────────────────────────────────────
#  Constants  (must match TX)
# ─────────────────────────────────────────────────────────────────────────────

ACCESS_CODE_BYTES  = [0xE1, 0x5A, 0xE8, 0x93]
FILLER_BYTES       = [0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF]  # x2
AC_THRESHOLD       = 28          # min bit matches out of 32
MAX_DEGREE         = 16
HEADER_FIXED_BYTES = 4+2+2+1+1+MAX_DEGREE   # AC+uid+seq+deg+slot+slot_list = 26
CRC_BYTES          = 4
FILLER_LEN         = 8
PACKET_SIZE        = 100
PAYLOAD_LEN        = PACKET_SIZE - HEADER_FIXED_BYTES - CRC_BYTES - FILLER_LEN  # = 62

SPS                = 10
RRC_ALPHA          = 0.350
RRC_NUM_TAPS       = 110
DIFF_INIT_STATE    = 0           # last bit of filler [..0xEF] = 1


# ─────────────────────────────────────────────────────────────────────────────
#  DSP Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_rrc_taps(sps=SPS, alpha=RRC_ALPHA, num_taps=RRC_NUM_TAPS):
    """Root Raised Cosine filter taps (normalised, matches GNU Radio firdes)."""
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
            num = (np.sin(np.pi * t * (1.0 - alpha)) +
                   4.0 * alpha * t * np.cos(np.pi * t * (1.0 + alpha)))
            den = np.pi * t * (1.0 - (4.0 * alpha * t) ** 2)
            taps[i] = num / den
    taps /= np.sqrt(np.sum(taps ** 2))
    return taps.astype(np.float32)


_RRC_TAPS = make_rrc_taps()   # computed once


def bytes_to_bits(byte_list):
    bits = []
    for b in byte_list:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return np.array(bits, dtype=np.uint8)


def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    n    = (len(bits) // 8) * 8
    bits = bits[:n]
    return [int(''.join(map(str, bits[i:i+8])), 2) for i in range(0, n, 8)]


def differential_decode(bits, init_state=DIFF_INIT_STATE):
    """DBPSK differential decode."""
    bits = np.asarray(bits, dtype=np.uint8)
    prev = np.empty(len(bits), dtype=np.uint8)
    prev[0]  = init_state
    prev[1:] = bits[:-1]
    return (bits ^ prev).astype(np.uint8)


def differential_encode(bits, init_state=DIFF_INIT_STATE):
    """DBPSK differential encode."""
    enc  = np.empty(len(bits), dtype=np.uint8)
    prev = init_state
    for i, b in enumerate(bits):
        enc[i] = b ^ prev
        prev   = enc[i]
    return enc


def bpsk_demodulate(iq_samples):
    """Hard-decision BPSK: real > 0 → 0, real < 0 → 1."""
    return (iq_samples.real < 0).astype(np.uint8)


def bpsk_modulate(bits):
    """Map bits to BPSK complex symbols: 0→+1, 1→-1."""
    return (1.0 - 2.0 * bits).astype(np.complex64)


def demodulate_slot(slot_iq):
    """
    IQ samples (post-Costas, post-RRC) → decoded bits.
    Steps: downsample (take every SPS-th sample) → hard decision → diff decode
    """
    # Downsample: symbol centres at sample 0, SPS, 2*SPS, ...
    symbols = slot_iq[::SPS]
    bits_dd = bpsk_demodulate(symbols)
    bits    = differential_decode(bits_dd)
    return bits


def remodulate_packet(payload_bits):
    """
    Reconstruct IQ waveform for a decoded packet (for SIC subtraction).
    Steps: diff encode → BPSK → upsample → RRC (TX side only).
    Note: RRC applied once here (TX); RX RRC already applied to received signal.
    """
    enc      = differential_encode(payload_bits)
    symbols  = bpsk_modulate(enc)
    # Upsample
    up       = np.zeros(len(symbols) * SPS, dtype=np.complex64)
    up[::SPS] = symbols
    # Apply RRC (TX side)
    filtered = np.convolve(up, _RRC_TAPS, mode='full').astype(np.complex64)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
#  Packet Parser
# ─────────────────────────────────────────────────────────────────────────────

def find_access_code(bits, threshold=AC_THRESHOLD):
    """
    Sliding window search for access code in bit stream.
    Returns index of first bit AFTER the access code, or -1.
    """
    ac_bits = bytes_to_bits(ACCESS_CODE_BYTES)
    ac_len  = len(ac_bits)   # 32
    for i in range(len(bits) - ac_len + 1):
        matches = int(np.sum(bits[i:i+ac_len] == ac_bits))
        if matches >= threshold:
            return i + ac_len
    return -1


def parse_irsa_packet(bits):
    """
    Parse IRSA packet bits (starting AFTER access code).
    Returns dict or None on failure.
    Layout after AC:
      user_id(2B) frame_seq(2B) degree(1B) this_slot(1B) slot_list(16B)
      random_data(PAYLOAD_LEN B) CRC32(4B) filler(8B)
    """
    raw = bits_to_bytes(bits)

    min_len = 2+2+1+1+MAX_DEGREE + PAYLOAD_LEN + CRC_BYTES + FILLER_LEN
    if len(raw) < min_len:
        return None

    idx        = 0
    user_id    = (raw[idx] << 8) | raw[idx+1];  idx += 2
    frame_seq  = (raw[idx] << 8) | raw[idx+1];  idx += 2
    degree     = raw[idx];                        idx += 1
    this_slot  = raw[idx];                        idx += 1
    slot_list  = [s for s in raw[idx:idx+MAX_DEGREE] if s != 0]; idx += MAX_DEGREE
    payload    = raw[idx:idx+PAYLOAD_LEN];        idx += PAYLOAD_LEN
    crc_rx     = raw[idx:idx+CRC_BYTES];          idx += CRC_BYTES
    # filler ignored

    # CRC32 check — covers user_id..payload (everything between AC and CRC)
    protected     = raw[:2+2+1+1+MAX_DEGREE+PAYLOAD_LEN]
    crc_expected  = binascii.crc32(bytes(protected)) & 0xFFFFFFFF
    crc_actual    = int.from_bytes(bytes(crc_rx), byteorder='little')
    crc_ok        = (crc_expected == crc_actual)

    return {
        'user_id'   : user_id,
        'frame_seq' : frame_seq,
        'degree'    : degree,
        'this_slot' : this_slot,
        'slot_list' : slot_list,
        'payload'   : payload,
        'crc_ok'    : crc_ok,
    }


def decode_slot(slot_iq):
    """
    Full decode pipeline for one slot's IQ samples.
    Returns parsed packet dict or None.
    """
    bits       = demodulate_slot(slot_iq)
    ac_end_idx = find_access_code(bits)
    if ac_end_idx < 0:
        return None
    return parse_irsa_packet(bits[ac_end_idx:])


# ─────────────────────────────────────────────────────────────────────────────
#  SIC Processor
# ─────────────────────────────────────────────────────────────────────────────

class IRSASICProcessor:
    """
    Offline IRSA SIC processor.

    Parameters
    ----------
    N_slots       : number of slots per frame
    slot_samples  : IQ samples per slot  (slot_duration * fs)
    max_iter      : maximum SIC iterations
    pkt_log       : output CSV for decoded packets
    sic_log       : output CSV for SIC iteration trace
    """

    def __init__(
        self,
        N_slots      = 2,
        slot_samples = 16000,
        max_iter     = 10,
        pkt_log      = "decoded_packets.csv",
        sic_log      = "sic_log.csv",
    ):
        self.N_slots      = N_slots
        self.slot_samples = slot_samples
        self.max_iter     = max_iter
        self.pkt_log      = pkt_log
        self.sic_log      = sic_log

        self._pkt_count   = 0
        self._frame_count = 0

        # Init CSV files
        with open(pkt_log, 'w', newline='') as f:
            csv.writer(f).writerow([
                'frame_id', 'sic_iter', 'slot_index', 'user_id',
                'frame_seq', 'degree', 'this_slot', 'slot_list',
                'crc_ok', 'payload_hex',
            ])
        with open(sic_log, 'w', newline='') as f:
            csv.writer(f).writerow([
                'frame_id', 'sic_iter', 'slot_index',
                'event',        # DECODED / CANCELLED / NO_AC / CRC_FAIL
                'user_id', 'degree', 'slot_list',
            ])

        print(
            f"[SIC] Init: N_slots={N_slots}, slot_samples={slot_samples}, "
            f"max_iter={max_iter}"
        )

    # ── Logging helpers ───────────────────────────────────────────────────

    def _log_packet(self, frame_id, sic_iter, slot_idx, pkt):
        self._pkt_count += 1
        pay_hex = ' '.join(f'{b:02X}' for b in pkt['payload'])
        with open(self.pkt_log, 'a', newline='') as f:
            csv.writer(f).writerow([
                frame_id, sic_iter, slot_idx,
                pkt['user_id'], pkt['frame_seq'], pkt['degree'],
                pkt['this_slot'], pkt['slot_list'],
                pkt['crc_ok'], pay_hex,
            ])

    def _log_sic(self, frame_id, sic_iter, slot_idx, event,
                 user_id='-', degree='-', slot_list='-'):
        with open(self.sic_log, 'a', newline='') as f:
            csv.writer(f).writerow([
                frame_id, sic_iter, slot_idx,
                event, user_id, degree, slot_list,
            ])

    # ── Replica reconstruction ────────────────────────────────────────────

    def _reconstruct_replica(self, pkt, slot_idx):
        """
        Reconstruct the IQ waveform for a decoded packet in a specific slot.
        Rebuilds the full packet bits and remodulates.
        """
        # Rebuild packet bits the TX would have sent for this slot
        ac_bits      = bytes_to_bits(ACCESS_CODE_BYTES)
        uid_bytes    = [(pkt['user_id']   >> 8) & 0xFF, pkt['user_id']   & 0xFF]
        seq_bytes    = [(pkt['frame_seq'] >> 8) & 0xFF, pkt['frame_seq'] & 0xFF]
        deg_byte     = [pkt['degree']  & 0xFF]
        slot_byte    = [slot_idx       & 0xFF]
        slot_field   = [s & 0xFF for s in pkt['slot_list']]
        slot_field  += [0x00] * (MAX_DEGREE - len(slot_field))
        payload      = pkt['payload']

        protected    = uid_bytes + seq_bytes + deg_byte + slot_byte + slot_field + list(payload)
        crc_val      = binascii.crc32(bytes(protected)) & 0xFFFFFFFF
        crc_bytes_   = list(crc_val.to_bytes(4, byteorder='little'))
        filler       = FILLER_BYTES

        packet_bytes = ACCESS_CODE_BYTES + protected + crc_bytes_ + filler
        packet_bits  = bytes_to_bits(packet_bytes)

        # Remodulate → IQ
        replica_iq   = remodulate_packet(packet_bits)

        # Trim/pad to slot_samples
        n = self.slot_samples
        if len(replica_iq) >= n:
            return replica_iq[:n]
        else:
            padded       = np.zeros(n, dtype=np.complex64)
            padded[:len(replica_iq)] = replica_iq
            return padded

    # ── Main SIC loop ─────────────────────────────────────────────────────

    def process_frame(self, frame_iq, frame_id=None):
        """
        Run IRSA SIC on one frame of IQ samples.

        Parameters
        ----------
        frame_iq : complex64 ndarray, length = N_slots * slot_samples
        frame_id : optional int label

        Returns
        -------
        decoded  : list of decoded packet dicts (crc_ok ones only)
        """
        if frame_id is None:
            frame_id = self._frame_count
        self._frame_count += 1

        expected_len = self.N_slots * self.slot_samples
        if len(frame_iq) < expected_len:
            print(f"[SIC] Frame {frame_id}: too short ({len(frame_iq)} < {expected_len}), skipping.")
            return []

        # Working copy of residual IQ per slot
        residual = [
            frame_iq[s * self.slot_samples : (s+1) * self.slot_samples].copy()
            for s in range(self.N_slots)
        ]

        decoded_packets = []           # all CRC-OK packets this frame
        cancelled_slots = set()        # slots fully resolved

        print(f"\n[SIC] ── Frame {frame_id:04d} ──────────────────────────")

        for sic_iter in range(self.max_iter):
            newly_decoded = []

            # ── Step 1: Try to decode every non-cancelled slot ────────────
            for slot_idx in range(self.N_slots):
                if slot_idx in cancelled_slots:
                    continue

                pkt = decode_slot(residual[slot_idx])

                if pkt is None:
                    print(f"[SIC]   iter={sic_iter} slot={slot_idx+1:02d}  NO_AC")
                    self._log_sic(frame_id, sic_iter, slot_idx+1, 'NO_AC')
                    continue

                if not pkt['crc_ok']:
                    print(
                        f"[SIC]   iter={sic_iter} slot={slot_idx+1:02d}  "
                        f"CRC_FAIL  user={pkt['user_id']}"
                    )
                    self._log_sic(frame_id, sic_iter, slot_idx+1, 'CRC_FAIL',
                                  pkt['user_id'], pkt['degree'], pkt['slot_list'])
                    continue

                print(
                    f"[SIC]   iter={sic_iter} slot={slot_idx+1:02d}  "
                    f"DECODED  user={pkt['user_id']}  "
                    f"d={pkt['degree']}  copies@{pkt['slot_list']}"
                )
                self._log_sic(frame_id, sic_iter, slot_idx+1, 'DECODED',
                              pkt['user_id'], pkt['degree'], pkt['slot_list'])
                self._log_packet(frame_id, sic_iter, slot_idx+1, pkt)

                newly_decoded.append((slot_idx, pkt))
                decoded_packets.append(pkt)

            if not newly_decoded:
                print(f"[SIC]   iter={sic_iter}  No new decodes — stopping SIC.")
                break

            # ── Step 2: Cancel all copies of each decoded packet ──────────
            for orig_slot_idx, pkt in newly_decoded:
                replica = self._reconstruct_replica(pkt, orig_slot_idx)

                for cancel_slot in pkt['slot_list']:
                    slot_idx_0 = cancel_slot - 1   # convert to 0-indexed
                    if slot_idx_0 < 0 or slot_idx_0 >= self.N_slots:
                        continue
                    residual[slot_idx_0] -= replica
                    cancelled_slots.add(slot_idx_0)
                    print(
                        f"[SIC]   iter={sic_iter}  CANCELLED slot={cancel_slot:02d}"
                        f"  (user={pkt['user_id']})"
                    )
                    self._log_sic(frame_id, sic_iter, cancel_slot, 'CANCELLED',
                                  pkt['user_id'], pkt['degree'], pkt['slot_list'])

            if len(cancelled_slots) == self.N_slots:
                print(f"[SIC]   All slots resolved after iter={sic_iter}.")
                break

        # ── Summary ───────────────────────────────────────────────────────
        crc_ok_count = sum(1 for p in decoded_packets if p['crc_ok'])
        print(
            f"[SIC] Frame {frame_id:04d} done — "
            f"decoded={crc_ok_count} packets, "
            f"resolved={len(cancelled_slots)}/{self.N_slots} slots"
        )
        return [p for p in decoded_packets if p['crc_ok']]


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point — process a .npy file of frame-synced IQ samples
# ─────────────────────────────────────────────────────────────────────────────

def process_file(
    iq_file,
    N_slots      = 2,
    slot_samples = 16000,
    max_iter     = 10,
    pkt_log      = "decoded_packets.csv",
    sic_log      = "sic_log.csv",
):
    """
    Load frame-synced IQ samples from a .npy file and run SIC.

    The file must contain a complex64 array where each consecutive
    N_slots * slot_samples samples is one frame.

    Parameters
    ----------
    iq_file      : path to .npy complex64 file (output of Costas loop / frame sync)
    N_slots      : slots per frame
    slot_samples : samples per slot
    max_iter     : max SIC iterations per frame
    pkt_log      : output CSV for decoded packets
    sic_log      : output CSV for SIC log
    """
    print(f"[SIC] Loading {iq_file} ...")
    iq = np.load(iq_file).astype(np.complex64)
    print(f"[SIC] Loaded {len(iq)} samples")

    processor    = IRSASICProcessor(N_slots, slot_samples, max_iter, pkt_log, sic_log)
    frame_len    = N_slots * slot_samples
    n_frames     = len(iq) // frame_len
    all_decoded  = []

    print(f"[SIC] Processing {n_frames} frame(s) ...")

    for f in range(n_frames):
        frame_iq = iq[f * frame_len : (f+1) * frame_len]
        pkts     = processor.process_frame(frame_iq, frame_id=f)
        all_decoded.extend(pkts)

    print(f"\n[SIC] Total decoded packets: {len(all_decoded)}")
    print(f"[SIC] Packet log  → {pkt_log}")
    print(f"[SIC] SIC log     → {sic_log}")
    return all_decoded


# ─────────────────────────────────────────────────────────────────────────────
#  Quick test with synthetic data (run standalone)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # Usage: python irsa_sic_processor.py <iq_file.npy> [N_slots] [slot_samples]
        iq_file      = sys.argv[1]
        N_slots      = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        slot_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
        process_file(iq_file, N_slots=N_slots, slot_samples=slot_samples)
    else:
        import glob, re

        # ── Load and process all saved frames ────────────────────────────
        N_slots      = 2       # change to match your setup
        slot_samples = 16000   # 0.4 * 40000

        files = sorted(
            glob.glob("frame_*.npy"),
            key=lambda f: int(re.search(r'\d+', f).group())
        )

        if not files:
            print("[SIC] No frame_*.npy files found in current directory.")
        else:
            print(f"[SIC] Found {len(files)} frame file(s).")
            proc = IRSASICProcessor(
                N_slots      = N_slots,
                slot_samples = slot_samples,
                max_iter     = 10,
                pkt_log      = "decoded_packets.csv",
                sic_log      = "sic_log.csv",
            )
            for i, f in enumerate(files):
                frame_iq = np.load(f).astype(np.complex64)
                proc.process_frame(frame_iq, frame_id=i)