"""
irsa_offline.py
───────────────
Combined offline IRSA processor:
  1. Frame Sync  — sliding window IQ correlation to find frame boundaries
  2. SIC         — successive interference cancellation per frame

Input  : sync_samples.bin  (complex64 raw IQ from GNU Radio File Sink)
Output : decoded_packets.csv
         sic_log.csv

Usage
─────
  python irsa_offline.py costas_out.bin [N_slots] [slot_samples]

  # defaults: N_slots=2, slot_samples=800

GNU Radio File Sink settings
─────────────────────────────
  Format       : Complex (float32 IQ interleaved)
  Place after  : Costas Loop output
"""

import numpy as np
import csv
import binascii
import sys
import glob
import re
from scipy.signal import find_peaks

import random


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

SPS               = 1
RRC_ALPHA         = 0.350
RRC_NUM_TAPS      = 110    # 11 * sps
DIFF_INIT_STATE   = 0   # silence (0x00) precedes each packet from PDU_to_Timed_Byte_Stream


# ─────────────────────────────────────────────────────────────────────────────
#  DSP helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_rrc_taps(sps=SPS, alpha=RRC_ALPHA, num_taps=RRC_NUM_TAPS):
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


# ─────────────────────────────────────────────────────────────────────────────
#  Frame Sync — build template and correlate
# ─────────────────────────────────────────────────────────────────────────────

def build_template(init_state=DIFF_INIT_STATE):
    bits    = bytes_to_bits(ACCESS_CODE_BYTES)
    enc     = differential_encode(bits, init_state)
    symbols = bpsk_modulate(enc)   # 32 complex symbols, 1 per bit
    symbols /= np.max(np.abs(symbols))
    return symbols
    # up      = np.zeros(len(symbols) * SPS, dtype=np.complex64)
    # up[::SPS] = symbols
    # # Apply RRC twice (TX + RX matched filter)
    # filtered = np.convolve(up, _RRC_TAPS, mode='full')
    # filtered = np.convolve(filtered, _RRC_TAPS, mode='full').astype(np.complex64)
    # filtered /= np.max(np.abs(filtered))
    # return filtered


def normalised_xcorr(signal, template):
    """Normalised cross-correlation magnitude, output in [0,1]."""
    corr        = np.correlate(signal, template, mode='valid')
    mag         = np.abs(corr)
    tmpl_len    = len(template)
    sig_energy  = np.convolve(np.abs(signal) ** 2,
                              np.ones(tmpl_len), mode='valid')
    sig_energy  = np.sqrt(np.maximum(sig_energy, 1e-12))
    tmpl_energy = np.sqrt(np.sum(np.abs(template) ** 2))
    return mag / (sig_energy * tmpl_energy + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
#  Packet decode helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_access_code(bits, threshold=AC_THRESHOLD):
    ac_bits = bytes_to_bits(ACCESS_CODE_BYTES)
    ac_len  = len(ac_bits)
    for i in range(len(bits) - ac_len + 1):
        if int(np.sum(bits[i:i+ac_len] == ac_bits)) >= threshold:
            return i + ac_len
    return -1


def parse_irsa_packet(bits):
    raw = bits_to_bytes(bits)
    # print(f"[DEBUG] total bits={len(bits)}, raw bytes={len(raw)}")

    min_len = 2+2+1+1+MAX_DEGREE + PAYLOAD_LEN + CRC_BYTES + FILLER_LEN
    # print(f"[DEBUG] expected bytes={min_len}, got={len(raw)}")

    if len(raw) < min_len:
        return None

    # Print payload region as hex
    idx1 = 2+2+1+1+MAX_DEGREE
    # print(f"[DEBUG] payload hex: {bytes(raw[idx1:idx1+8]).hex()}")
    # print(f"[DEBUG] crc region:  {bytes(raw[idx1+PAYLOAD_LEN:idx1+PAYLOAD_LEN+4]).hex()}")

    # In parse_irsa_packet, try parsing with 1-7 bit offsets
    # for offset in range(1, 8):
    #     shifted_raw = bits_to_bytes(bits[offset:])
    #     shifted_payload = shifted_raw[2+2+1+1+MAX_DEGREE : 2+2+1+1+MAX_DEGREE+8]
    #     print(f"offset={offset}: {bytes(shifted_payload).hex()}")


    idx       = 0
    user_id   = (raw[idx] << 8) | raw[idx+1];  idx += 2
    frame_seq = (raw[idx] << 8) | raw[idx+1];  idx += 2
    degree    = raw[idx];                        idx += 1
    this_slot = raw[idx];                        idx += 1
    slot_list = [s for s in raw[idx:idx+MAX_DEGREE] if s != 0]; idx += MAX_DEGREE
    payload   = raw[idx:idx+PAYLOAD_LEN];        idx += PAYLOAD_LEN
    crc_rx    = raw[idx:idx+CRC_BYTES]

    protected    = raw[:2+2+1+1+MAX_DEGREE+PAYLOAD_LEN]
        
    rng = random.Random(31)  # user_id=1, seed=1*31
    expected_payload = [rng.randint(0, 255) for _ in range(62)]
    # print(f"[DEBUG] expected payload: {bytes(expected_payload[:8]).hex()}")


    crc_expected = binascii.crc32(bytes(protected)) & 0xFFFFFFFF
    crc_actual   = int.from_bytes(bytes(crc_rx), byteorder='little')

    return {
        'user_id'   : user_id,
        'frame_seq' : frame_seq,
        'degree'    : degree,
        'this_slot' : this_slot,
        'slot_list' : slot_list,
        'payload'   : payload,
        'crc_ok'    : (crc_expected == crc_actual),
    }


def decode_slot(slot_iq):
    symbols  = slot_iq[::SPS]
    bits_raw = bpsk_demodulate(symbols)
    

    for init_state in [0, 1]:
        bits   = differential_decode(bits_raw, init_state)
        # After bpsk_demodulate, print first 40 bits
        # print("First 40 bits:", bits[:40])
        # print("Expected AC:  ", bytes_to_bits(ACCESS_CODE_BYTES)[:40])
        ac_end = find_access_code(bits)
        # print(f"  init_state={init_state} ac_end={ac_end}")    # ------------ debug  ---------------
        if ac_end < 0:
            continue
        # print(f'init_state for differential decoding: {init_state}')
        pkt = parse_irsa_packet(bits[ac_end:])

        #   ---------    for debug ------------
        # if pkt and not pkt['crc_ok']: 
        #     raw = bits_to_bytes(bits[ac_end:]) 
        #     print(f"[DEBUG] raw bytes after AC: {len(raw)}") 
        #     print(f"[DEBUG] user_id={pkt['user_id']} frame_seq={pkt['frame_seq']}") 
        #     print(f"[DEBUG] degree={pkt['degree']} this_slot={pkt['this_slot']}") 
        #     print(f"[DEBUG] slot_list={pkt['slot_list']}") 
        #     # recompute CRC manually 
        #     protected = raw[:2+2+1+1+MAX_DEGREE+PAYLOAD_LEN] 
        #     crc_exp = binascii.crc32(bytes(protected)) & 0xFFFFFFFF 
        #     crc_rx = int.from_bytes(bytes(raw[2+2+1+1+MAX_DEGREE+PAYLOAD_LEN: 
        #                                     2+2+1+1+MAX_DEGREE+PAYLOAD_LEN+4]), 
        #                             byteorder='little') 
        #     print(f"[DEBUG] CRC expected=0x{crc_exp:08X} actual=0x{crc_rx:08X}") 
        #     print(f"[DEBUG] total bits in slot: {len(bits)}") 

        if pkt and pkt['crc_ok']:
            return pkt

    # best-effort for debug
    bits   = differential_decode(bits_raw, 0)
    ac_end = find_access_code(bits)
    if ac_end < 0:
        return None
    return parse_irsa_packet(bits[ac_end:])


# ─────────────────────────────────────────────────────────────────────────────
#  Replica reconstruction for SIC
# ─────────────────────────────────────────────────────────────────────────────

# def reconstruct_replica(pkt, slot_idx, slot_samples, sps=SPS):
#     """Rebuild IQ waveform for a decoded packet (TX RRC only — RX RRC already applied)."""
#     uid_b    = [(pkt['user_id']   >> 8) & 0xFF, pkt['user_id']   & 0xFF]
#     seq_b    = [(pkt['frame_seq'] >> 8) & 0xFF, pkt['frame_seq'] & 0xFF]
#     deg_b    = [pkt['degree']  & 0xFF]
#     slot_b   = [slot_idx       & 0xFF]
#     sl_field = [s & 0xFF for s in pkt['slot_list']]
#     sl_field += [0x00] * (MAX_DEGREE - len(sl_field))
#     payload  = list(pkt['payload'])

#     protected  = uid_b + seq_b + deg_b + slot_b + sl_field + payload
#     crc_val    = binascii.crc32(bytes(protected)) & 0xFFFFFFFF
#     crc_b      = list(crc_val.to_bytes(4, byteorder='little'))
#     pkt_bytes  = ACCESS_CODE_BYTES + protected + crc_b + FILLER_BYTES
#     pkt_bits   = bytes_to_bits(pkt_bytes)


#     # ── TX chain ───────────────────────────────────────────────────────
#     enc     = differential_encode(pkt_bits)
#     symbols = bpsk_modulate(enc)          # complex64, 1 sample/symbol

#     if sps == 1:
#         # ── File sink is AFTER Symbol Sync ────────────────────────────
#         # Two RRC passes at sps=1 → net raised cosine → at sample points
#         # this equals a scalar (≈1.0 after normalisation). Just use symbols.
#         # But we must still apply the two-pass convolution to capture any
#         # inter-symbol "smearing" at non-integer offsets.
#         rrc = make_rrc_taps(sps=1, alpha=RRC_ALPHA, num_taps=RRC_NUM_TAPS)
#         replica = np.convolve(symbols, rrc, mode='full').astype(np.complex64)
#         replica = np.convolve(replica, rrc, mode='full').astype(np.complex64)

#         # The double convolution introduces a group delay of (num_taps-1) samples.
#         # We need to find the correct sampling phase — take every 1st sample
#         # starting after the filter delay:
#         delay = RRC_NUM_TAPS - 1          # total delay of two filters
#         replica = replica[delay:]         # align to symbol timing

#     else:
#         # ── File sink is BEFORE Symbol Sync (sps=10) ──────────────────
#         rrc = make_rrc_taps(sps=sps, alpha=RRC_ALPHA, num_taps=RRC_NUM_TAPS)

#         # TX: upsample then RRC
#         up = np.zeros(len(symbols) * sps, dtype=np.complex64)
#         up[::sps] = symbols
#         replica = np.convolve(up, rrc, mode='full').astype(np.complex64)

#         # RX matched filter
#         replica = np.convolve(replica, rrc, mode='full').astype(np.complex64)

#         # Remove group delay and downsample to sps=1 (Symbol Sync equivalent)
#         delay = RRC_NUM_TAPS - 1          # each filter contributes (N-1)/2 delay
#         replica = replica[delay::sps]     # sample at symbol instants

#     # ── Trim or zero-pad to slot_samples ──────────────────────────────
#     n_sym = len(replica)
#     if n_sym >= slot_samples:
#         return replica[:slot_samples]
#     out = np.zeros(slot_samples, dtype=np.complex64)
#     out[:n_sym] = replica

#     return out

def reconstruct_replica(pkt, slot_idx, slot_samples):
    """
    Rebuild IQ waveform for a decoded packet.
    
    File sink is after Symbol Sync → sps=1.
    Net effect of TX RRC + RX RRC sampled at symbol instants = scalar.
    Replica is simply the BPSK symbol stream, scaled to match received amplitude.
    
    Phase correction is applied by estimating the rotation between
    the replica and the received slot IQ before subtraction.
    """
    # ── Rebuild packet bytes ───────────────────────────────────────────
    uid_b    = [(pkt['user_id']   >> 8) & 0xFF, pkt['user_id']   & 0xFF]
    seq_b    = [(pkt['frame_seq'] >> 8) & 0xFF, pkt['frame_seq'] & 0xFF]
    deg_b    = [pkt['degree']  & 0xFF]
    slot_b   = [slot_idx       & 0xFF]
    sl_field = [s & 0xFF for s in pkt['slot_list']]
    sl_field += [0x00] * (MAX_DEGREE - len(sl_field))
    payload  = list(pkt['payload'])

    protected = uid_b + seq_b + deg_b + slot_b + sl_field + payload
    crc_val   = binascii.crc32(bytes(protected)) & 0xFFFFFFFF
    crc_b     = list(crc_val.to_bytes(4, byteorder='little'))
    pkt_bytes = ACCESS_CODE_BYTES + protected + crc_b + FILLER_BYTES
    pkt_bits  = bytes_to_bits(pkt_bytes)

    # ── TX chain: diff encode → BPSK ──────────────────────────────────
    enc     = differential_encode(pkt_bits)
    symbols = bpsk_modulate(enc)   # complex64, ±1+0j, one per bit

    # ── Trim or zero-pad to slot_samples ──────────────────────────────
    n_sym = len(symbols)
    if n_sym >= slot_samples:
        return symbols[:slot_samples]

    out = np.zeros(slot_samples, dtype=np.complex64)
    out[:n_sym] = symbols
    return out


def estimate_phase_scale(received_slot, replica):
    """
    Estimate complex scale factor (amplitude + phase) between
    received slot and ideal replica using least squares.
    
    scale = (replica^H · received) / (replica^H · replica)
    
    This is a single complex number:
      |scale| = amplitude ratio
      angle(scale) = phase rotation from Costas loop residual
    """
    # Only use non-zero replica samples (where signal actually exists)
    mask = np.abs(replica) > 0.1
    if np.sum(mask) < 10:
        return complex(1.0, 0.0)

    r = replica[mask]
    s = received_slot[mask]

    # Least squares: minimise ||s - scale·r||²
    scale = np.dot(np.conj(r), s) / np.dot(np.conj(r), r)
    return scale


# ─────────────────────────────────────────────────────────────────────────────
#  SIC processor
# ─────────────────────────────────────────────────────────────────────────────



# def find_frames(iq, N_slots, slot_samples, threshold=0.7):
#     # template  = build_template()
#     # tmpl_len  = len(template)
#     frame_len = N_slots * slot_samples
#     min_gap   = slot_samples

#     templates = [build_template(init_state=0), build_template(init_state=1)]
#     corr0 = normalised_xcorr(iq, templates[0])
#     corr1 = normalised_xcorr(iq, templates[1])
#     norm_corr = np.maximum(corr0, corr1)

#     peak_indices, _ = find_peaks(norm_corr, height=threshold, distance=min_gap)
#     peak_indices = peak_indices[np.argsort(peak_indices)]  # sort by position   


#     print(f"[SYNC] Template length : {tmpl_len} samples")
#     print(f"[SYNC] Frame length    : {frame_len} samples")
#     print(f"[SYNC] Total IQ samples: {len(iq)}")
#     print(f"[SYNC] Correlating ...")

#     # norm_corr   = normalised_xcorr(iq, template)
#     frames      = []
#     # last_detect = -min_gap
#     search_pos  = 0

#     for peak_idx in peak_indices:
#         peak_val = norm_corr[peak_idx]

#         # Skip peaks inside already-detected frame
#         if search_pos > 0 and peak_idx < search_pos:
#             print(f"[SYNC] Peak {peak_idx} inside already-detected frame, skipping.")
#             continue

#         if peak_val < threshold:
#             break

#         if peak_idx <= last_detect + min_gap:
#             search_pos = peak_idx + 1
#             continue

#         frame_start = None
#         frame_end   = None  # ← add this

#         # ── Primary: decode slot at peak, use this_slot for boundary ──────
#         slot_end = peak_idx + slot_samples
#         if slot_end <= len(iq):
#             pkt = decode_slot(iq[peak_idx:slot_end])
#             # print(f'CRC Check: {pkt['crc_ok']}')

#             if pkt and pkt['crc_ok']:
#                 fs = peak_idx - (pkt['this_slot'] - 1) * slot_samples
#                 fe = fs + frame_len
#                 if fs >= 0 and fe <= len(iq):
#                     frame_start = fs
#                     frame_end   = fe
#                     print(f"[SYNC] Peak {peak_idx}: this_slot={pkt['this_slot']} "
#                           f"→ frame [{fs}:{fe}]")

#         # ── Fallback: try all slot offsets ────────────────────────────────
#         if frame_start is None:
#             for slot_offset in range(N_slots):
#                 fs = peak_idx - slot_offset * slot_samples
#                 fe = fs + frame_len
#                 if fs < 0 or fe > len(iq):
#                     continue
#                 candidate = iq[fs:fe]
#                 for s in range(N_slots):
#                     pkt = decode_slot(candidate[s * slot_samples:(s+1) * slot_samples])
#                     if pkt and pkt['crc_ok']:
#                         frame_start = fs
#                         frame_end   = fe
#                         print(f"[SYNC] Peak {peak_idx}: fallback offset={slot_offset} "
#                               f"→ frame [{fs}:{fe}]")
#                         break
#                 if frame_start is not None:
#                     break

#         if frame_start is None:
#             print(f"[SYNC] Peak {peak_idx} (val={peak_val:.3f}): no valid alignment, skipping.")
#             search_pos = peak_idx + 1
#             continue

#         frames.append(iq[frame_start:frame_end].copy())
#         last_detect = frame_start
#         search_pos  = frame_end
#         print(f"[SYNC] Frame {len(frames):04d}: peak_val={peak_val:.3f} samples [{frame_start}:{frame_end}]")

#     print(f"[SYNC] Found {len(frames)} frame(s).")
#     return frames


def find_frames(iq, N_slots, slot_samples, threshold=0.7):
    frame_len = N_slots * slot_samples
    min_gap   = slot_samples  # reduced from frame_len//2 so beacon peak isn't suppressed

    templates = [build_template(init_state=0), build_template(init_state=1)]
    corr0 = normalised_xcorr(iq, templates[0])
    corr1 = normalised_xcorr(iq, templates[1])
    norm_corr = np.maximum(corr0, corr1)

    peak_indices, _ = find_peaks(norm_corr, height=threshold, distance=min_gap)
    peak_indices = peak_indices[np.argsort(peak_indices)]

    print(f"[SYNC] Template length : {len(templates[0])} samples")
    print(f"[SYNC] Frame length    : {frame_len} samples")
    print(f"[SYNC] Total IQ samples: {len(iq)}")
    print(f"[SYNC] Correlating ...")

    frames     = []
    search_pos = 0

    for peak_idx in peak_indices:
        if peak_idx < search_pos:
            continue

        peak_val = norm_corr[peak_idx]
        if peak_val < threshold:
            break

        slot_end = peak_idx + slot_samples
        if slot_end > len(iq):
            continue

        pkt = decode_slot(iq[peak_idx:slot_end])

        if pkt is None or not pkt['crc_ok']:
            print(f"[SYNC] Peak {peak_idx} (val={peak_val:.3f}): decode failed, skipping.")
            search_pos = peak_idx + 1
            continue

        # ── Beacon: frame_seq==0 -> data frame starts slot_samples later ──
        if pkt['frame_seq'] == 0:
            print(f"[SYNC] Peak {peak_idx} (val={peak_val:.3f}): BEACON "
                  f"(frame_seq=0) -> data frame starts at {peak_idx + slot_samples}")
            search_pos = peak_idx + slot_samples  # skip beacon, land on slot1 of data frame
            print(f"[SYNC] Beacon peak={peak_idx}, "
                    f"next data slot1 expected at {peak_idx + slot_samples}")
            continue

        # ── Data packet: back-calculate frame start from this_slot ────────
        fs = peak_idx - (pkt['this_slot'] - 1) * slot_samples
        fe = fs + frame_len

        print(f"[SYNC] Offset check: peak_idx={peak_idx} "
            f"fs={fs} diff={peak_idx - fs} expected={( pkt['this_slot']-1)*slot_samples}")
        if fs < 0 or fe > len(iq):
            print(f"[SYNC] Peak {peak_idx}: frame [{fs}:{fe}] out of bounds, skipping.")
            search_pos = peak_idx + 1
            continue

        frames.append(iq[fs:fe].copy())

        plot_slot_boundaries(iq[fs:fe], N_slots, slot_samples)

        search_pos = fe
        print(f"[SYNC] Frame {len(frames):04d}: peak_val={peak_val:.3f} "
              f"samples [{fs}:{fe}]  user={pkt['user_id']} "
              f"frame_seq={pkt['frame_seq']} this_slot={pkt['this_slot']}")

    print(f"[SYNC] Found {len(frames)} frame(s).")
    return frames

def debug_dump_slots(frame_iq, N_slots, slot_samples, label=""):
    """Dump per-slot diagnostics to help identify where AC search fails."""
    print(f"\n[DEBUG DUMP] {label}")
    template_bits = bytes_to_bits(ACCESS_CODE_BYTES)
    
    for s in range(N_slots):
        chunk = frame_iq[s * slot_samples:(s+1) * slot_samples]
        energy = np.mean(np.abs(chunk)**2)
        
        # Raw BPSK demod → differential decode → check AC
        symbols  = chunk[::SPS]
        bits_raw = bpsk_demodulate(symbols)
        
        best_ac_pos  = -1
        best_ac_hits = 0
        best_init    = -1
        
        for init_state in [0, 1]:
            bits    = differential_decode(bits_raw, init_state)
            ac_bits = bytes_to_bits(ACCESS_CODE_BYTES)
            ac_len  = len(ac_bits)
            
            # Scan for best AC match even below threshold
            for i in range(min(len(bits) - ac_len + 1, 200)):  # first 200 positions
                hits = int(np.sum(bits[i:i+ac_len] == ac_bits))
                if hits > best_ac_hits:
                    best_ac_hits = hits
                    best_ac_pos  = i
                    best_init    = init_state
        
        # Phase of first few symbols
        mean_phase = np.angle(np.mean(chunk[:32]))
        mean_amp   = np.mean(np.abs(chunk[:32]))
        
        print(f"  slot={s+1:02d}  energy={energy:.4f}  "
              f"amp={mean_amp:.4f}  phase={np.degrees(mean_phase):+.1f}°  "
              f"best_AC_hits={best_ac_hits}/32 @ pos={best_ac_pos} "
              f"(init={best_init})")
        
        # Print first 64 demodulated bits for the best init_state
        if best_init >= 0:
            bits_raw2 = bpsk_demodulate(chunk[::SPS])
            bits2     = differential_decode(bits_raw2, best_init)
            expected  = bytes_to_bits(ACCESS_CODE_BYTES)
            print(f"         first 64 bits : {bits2[:64]}")
            print(f"         expected AC   : {expected}")
            print(f"         match mask    : "
                  f"{''.join('1' if a==b else '0' for a,b in zip(bits2[:32], expected))}")


def process_frame_sic(frame_iq, N_slots, slot_samples, pkt_writer, sic_writer):

    debug_dump_slots(frame_iq, N_slots, slot_samples, label="PRE-SIC RAW FRAME")

    # ---------   debug to check the prresence of signal as there is no AC at slot1  --------------
    for s in range(N_slots):
        chunk = frame_iq[s*slot_samples:(s+1)*slot_samples]
        print(f"  slot{s+1} energy: {np.mean(np.abs(chunk)**2):.4f}")

    residual     = [frame_iq[s * slot_samples:(s+1) * slot_samples].copy()
                    for s in range(N_slots)]
    decoded_slots = set()   # slots cleanly decoded — don't re-attempt
    dirty_slots   = set()   # slots that had cancellation applied — re-attempt
    all_decoded   = []

    # Start with all slots in the queue
    queue = list(range(N_slots))
    frame_id = None

    print(f"\n[SIC] ── New Frame {'─'*40}")

    while queue:
        slot_idx = queue.pop(0)

        if slot_idx in decoded_slots:
            continue

        pkt = decode_slot(residual[slot_idx])

        if pkt and not pkt['crc_ok']: 
            print('Packets whose CRC check failed')
            print(f"[DEBUG] user_id={pkt['user_id']} frame_seq={pkt['frame_seq']}") 
            print(f"[DEBUG] degree={pkt['degree']} this_slot={pkt['this_slot']}") 
            print(f"[DEBUG] slot_list={pkt['slot_list']}") 

        if pkt is None:
            print(f"  slot={slot_idx+1:02d}  NO_AC")
            sic_writer.writerow([frame_id, slot_idx+1, 'NO_AC', '-', '-', '-'])
            continue

        # to discard the frame0 (beacon)
        if pkt['crc_ok'] and pkt['frame_seq'] == 0:
            print(f"  slot={slot_idx+1:02d}  BEACON (discarded)")
            continue
        

        if frame_id is None:
            frame_id = pkt['frame_seq']

        if not pkt['crc_ok']:
            print(f"  slot={slot_idx+1:02d}  CRC_FAIL  user={pkt['user_id']}")
            sic_writer.writerow([frame_id, slot_idx+1, 'CRC_FAIL',
                                 pkt['user_id'], pkt['degree'], pkt['slot_list']])
            # Re-queue only if dirty (cancellation was applied since last attempt)
            if slot_idx in dirty_slots:
                queue.append(slot_idx)
                dirty_slots.discard(slot_idx)
            continue

        print(f"  slot={slot_idx+1:02d}  DECODED  user={pkt['user_id']}  "
              f"d={pkt['degree']}  copies@{pkt['slot_list']}")
        sic_writer.writerow([frame_id, slot_idx+1, 'DECODED',
                             pkt['user_id'], pkt['degree'], pkt['slot_list']])

        pay_hex = ' '.join(f'{b:02X}' for b in pkt['payload'])
        pkt_writer.writerow([
            frame_id, slot_idx+1, pkt['user_id'], pkt['frame_seq'],
            pkt['degree'], pkt['this_slot'], pkt['slot_list'],
            pkt['crc_ok'], pay_hex,
        ])
        print('Packets which are successfully decoded')
        print(f"[DEBUG] user_id={pkt['user_id']} frame_seq={pkt['frame_seq']}") 
        print(f"[DEBUG] degree={pkt['degree']} this_slot={pkt['this_slot']}") 
        print(f"[DEBUG] slot_list={pkt['slot_list']}") 

        decoded_slots.add(slot_idx)
        all_decoded.append(pkt)

        # replica = reconstruct_replica(pkt, slot_idx + 1, slot_samples)
        # Cancel replica from each slot in slot_list using correct this_slot since each slot have different this_slot so crc varies
        for cancel_slot in pkt['slot_list']:
            s0 = cancel_slot - 1
            if s0 == slot_idx:
                continue                      # skip the slot we just decoded
            if not (0 <= s0 < N_slots):
                continue
            if s0 in decoded_slots:
                continue                      # already clean, no point

            # Build ideal replica for the target slot
            replica = reconstruct_replica(pkt, cancel_slot, slot_samples)

            # Estimate amplitude and phase from the actual received slot
            # so the replica matches the received signal before subtraction
            scale = estimate_phase_scale(residual[s0], replica)
            replica_scaled = scale * replica

            residual[s0] -= replica_scaled
            dirty_slots.add(s0)
            if s0 not in queue:
                queue.append(s0)
            print(f"  CANCELLED slot={cancel_slot:02d}  (user={pkt['user_id']})")
            sic_writer.writerow([frame_id, cancel_slot, 'CANCELLED',
                                 pkt['user_id'], pkt['degree'], pkt['slot_list']])

    print(f"[SIC] Frame {frame_id} — decoded={len(all_decoded)}, "
          f"resolved={len(decoded_slots)}/{N_slots} slots")
    return all_decoded

def plot_slot_boundaries(frame_iq, N_slots, slot_samples):
    """Print correlation profile across the frame to verify slot alignment."""
    template = build_template()
    corr = normalised_xcorr(frame_iq, template)
    
    print("\n[SLOT BOUNDARY CHECK]")
    for s in range(N_slots):
        expected_pos = s * slot_samples
        # Check correlation in a window around expected slot start
        window_start = max(0, expected_pos - 20)
        window_end   = min(len(corr), expected_pos + 20)
        window       = corr[window_start:window_end]
        local_peak   = np.argmax(window) + window_start
        print(f"  slot={s+1}  expected_start={expected_pos}  "
              f"corr_peak_in_window={local_peak}  "
              f"offset={local_peak - expected_pos:+d}  "
              f"peak_val={corr[local_peak]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def run(iq_file, N_slots=2, slot_samples=16000, max_iter=10,
        threshold=0.7,
        pkt_log="decoded_packets.csv",
        sic_log="sic_log.csv"):

    # ── Load IQ ───────────────────────────────────────────────────────────
    print(f"[MAIN] Loading {iq_file} ...")
    iq = np.fromfile(iq_file, dtype=np.complex64)
    print(f"[MAIN] Loaded {len(iq)} samples  ({len(iq)/40000:.2f} s @ 4 kHz)")  # should be changed to 4kHz

    # ── Frame sync ────────────────────────────────────────────────────────
    frames = find_frames(iq, N_slots, slot_samples, threshold)

    if not frames:
        print("[MAIN] No frames found. Check threshold or IQ file.")
        return

    # ── SIC ───────────────────────────────────────────────────────────────
    total_decoded = 0

    with open(pkt_log, 'w', newline='') as pf, \
         open(sic_log, 'w', newline='') as sf:

        pkt_writer = csv.writer(pf)
        sic_writer = csv.writer(sf)

        pkt_writer.writerow([
            'frame_id', 'sic_iter', 'slot_index', 'user_id',
            'frame_seq', 'degree', 'slot_list',
            'crc_ok', 'payload_hex',
        ])
        sic_writer.writerow([
            'frame_id', 'slot_index',
            'event', 'user_id', 'degree', 'slot_list',
        ])

        for i, frame_iq in enumerate(frames):
            pkts = process_frame_sic(
                frame_iq, N_slots, slot_samples,
                pkt_writer, sic_writer          # no frame_id, no max_iter
            )
            total_decoded += len(pkts)

    print(f"\n[MAIN] Done — total decoded packets : {total_decoded}")
    print(f"[MAIN] Packet log → {pkt_log}")
    print(f"[MAIN] SIC log    → {sic_log}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python irsa_offline.py <costas_out.bin> [N_slots] [slot_samples] [threshold]")
        print("       python irsa_offline.py costas_out.bin 2 16000 0.7")
        sys.exit(1)

    iq_file      = sys.argv[1]
    N_slots      = int(sys.argv[2])   if len(sys.argv) > 2 else 2
    slot_samples = int(sys.argv[3])   if len(sys.argv) > 3 else 16000
    threshold    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8

    run(iq_file, N_slots=N_slots, slot_samples=slot_samples, threshold=threshold)