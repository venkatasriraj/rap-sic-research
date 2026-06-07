"""
Here in slots there is a possibility of slot offset for that we are 
correlate with the access code and identifying the possible start of the packet in the given slot.

The offset complicates the interference cancellation since it doesn't know where the packet boundaries actually are present.    
"""

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

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
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
RRC_NUM_TAPS      = 55    # 11 * sps
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


def find_ac_peak_in_window(iq_chunk, search_radius=30):
    """
    Correlate the AC template against IQ samples within iq_chunk.
    Returns the sample index of the best AC match (not bit-offset — sample offset).
    Works even on collided slots since correlation picks the strongest signal.
    """
    templates = [build_template(init_state=0), build_template(init_state=1)]
    corr0 = normalised_xcorr(iq_chunk, templates[0])
    corr1 = normalised_xcorr(iq_chunk, templates[1])
    corr  = np.maximum(corr0, corr1)

    # Only search within search_radius samples of slot start
    search_end = min(search_radius, len(corr))
    best_idx   = int(np.argmax(corr[:search_end]))
    best_val   = corr[best_idx]

    return best_idx, best_val


def extract_slot_from_ac(frame_iq, slot_idx, slot_samples, search_radius=30):
    """
    For a given slot index, find where the AC actually starts (within
    search_radius samples of the nominal boundary) and return:
      - slot_iq   : slot_samples worth of IQ starting from AC peak
      - ac_offset : sample offset of AC from nominal slot boundary
    
    If no good peak found, falls back to nominal boundary (offset=0).
    """
    nominal_start = slot_idx * slot_samples

    # Search window: nominal_start ± search_radius, but grab extra
    # samples so we have slot_samples after the AC peak
    window_start = max(0, nominal_start - search_radius)
    window_end   = min(len(frame_iq), nominal_start + search_radius + slot_samples)
    window       = frame_iq[window_start:window_end]

    # Find AC peak within the search window
    templates = [build_template(init_state=0), build_template(init_state=1)]
    corr0 = normalised_xcorr(window, templates[0])
    corr1 = normalised_xcorr(window, templates[1])
    corr  = np.maximum(corr0, corr1)

    # Only look within ±search_radius of nominal start
    local_search_end = min(2 * search_radius + 1, len(corr))
    best_local       = int(np.argmax(corr[:local_search_end]))
    best_val         = corr[best_local]

    # Convert back to frame-relative offset
    ac_sample_in_frame = window_start + best_local
    ac_offset          = ac_sample_in_frame - nominal_start  # signed offset from boundary

    # Extract slot_samples starting from the AC peak
    extract_start = ac_sample_in_frame
    extract_end   = extract_start + slot_samples
    if extract_end > len(frame_iq):
        # zero-pad if near end of frame
        chunk = np.zeros(slot_samples, dtype=np.complex64)
        available = len(frame_iq) - extract_start
        chunk[:available] = frame_iq[extract_start:extract_start + available]
    else:
        chunk = frame_iq[extract_start:extract_end]

    print(f"  [EXTRACT] slot={slot_idx+1}  nominal={nominal_start}  "
          f"ac_peak={ac_sample_in_frame}  offset={ac_offset:+d}  "
          f"corr={best_val:.4f}")

    return chunk, ac_offset


# ─────────────────────────────────────────────────────────────────────────────
#  Frame Sync — build template and correlate
# ─────────────────────────────────────────────────────────────────────────────

def build_template(init_state=DIFF_INIT_STATE):
    bits    = bytes_to_bits(ACCESS_CODE_BYTES)
    enc     = differential_encode(bits, init_state)
    symbols = bpsk_modulate(enc)   # 32 complex symbols, 1 per bit
    symbols /= np.max(np.abs(symbols))
    return symbols


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

    min_len = 2+2+1+1+MAX_DEGREE + PAYLOAD_LEN + CRC_BYTES # + FILLER_LEN
    # print(f"[DEBUG] expected bytes={min_len}, got={len(raw)}")

    if len(raw) < min_len:
        return None

    # Print payload region as hex
    idx1 = 2+2+1+1+MAX_DEGREE

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


def find_ac_offset(slot_iq, sps=SPS):
    """
    Find the sample offset where the AC starts within a slot.
    Returns (offset_in_samples, init_state) or (-1, -1) if not found.
    """
    symbols  = slot_iq[::sps]
    bits_raw = bpsk_demodulate(symbols)
    
    ac_bits = bytes_to_bits(ACCESS_CODE_BYTES)
    ac_len  = len(ac_bits)
    
    for init_state in [0, 1]:
        bits = differential_decode(bits_raw, init_state)
        for i in range(len(bits) - ac_len + 1):
            hits = int(np.sum(bits[i:i+ac_len] == ac_bits))
            if hits >= AC_THRESHOLD:
                return i * sps, init_state   # convert bit offset → sample offset
    
    return -1, -1


def decode_slot(slot_iq):
    """Returns (pkt, ac_sample_offset) instead of just pkt."""
    symbols  = slot_iq[::SPS]
    bits_raw = bpsk_demodulate(symbols)

    for init_state in [0, 1]:
        bits   = differential_decode(bits_raw, init_state)
        ac_end = find_access_code(bits)
        if ac_end < 0:
            continue
        pkt = parse_irsa_packet(bits[ac_end:])
        if pkt and pkt['crc_ok']:
            ac_start_bits = ac_end - len(bytes_to_bits(ACCESS_CODE_BYTES))
            return pkt, ac_start_bits * SPS   # return sample offset too

    # best-effort
    bits   = differential_decode(bits_raw, 0)
    ac_end = find_access_code(bits)
    if ac_end < 0:
        return None, -1
    ac_start_bits = ac_end - len(bytes_to_bits(ACCESS_CODE_BYTES))
    return parse_irsa_packet(bits[ac_end:]), ac_start_bits * SPS


# ─────────────────────────────────────────────────────────────────────────────
#  Replica reconstruction for SIC
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_replica(pkt, slot_idx, slot_samples, ac_offset=0):
    """
    Rebuild IQ. ac_offset = sample index where AC starts in the received slot.
    The replica is zero-padded at the front to match.
    """
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

    enc     = differential_encode(pkt_bits)
    symbols = bpsk_modulate(enc)

    # Place symbols at ac_offset within the slot
    out = np.zeros(slot_samples, dtype=np.complex64)
    end = ac_offset + len(symbols)
    if end <= slot_samples:
        out[ac_offset:end] = symbols
    else:
        out[ac_offset:] = symbols[:slot_samples - ac_offset]
    return out


# def estimate_phase_scale(received_slot, replica):
#     """
#     Estimate complex scale factor (amplitude + phase) between
#     received slot and ideal replica using least squares.
    
#     scale = (replica^H · received) / (replica^H · replica)
    
#     This is a single complex number:
#       |scale| = amplitude ratio
#       angle(scale) = phase rotation from Costas loop residual
#     """
#     # Only use non-zero replica samples (where signal actually exists)
#     mask = np.abs(replica) > 0.1
#     if np.sum(mask) < 10:
#         return complex(1.0, 0.0)

#     r = replica[mask]
#     s = received_slot[mask]

#     # Least squares: minimise ||s - scale·r||²
#     scale = np.dot(np.conj(r), s) / np.dot(np.conj(r), r)
#     return scale

def estimate_phase_scale(received_slot, replica, estimation_len=32):
    """
    Estimate complex scale factor using only the preamble (Access Code)
    to avoid phase-wrapping cancellation over the whole packet.
    """
    mask = np.abs(replica) > 0.1
    
    # Get indices where signal exists
    active_indices = np.where(mask)[0]
    if len(active_indices) < 10:
        return complex(1.0, 0.0)
        
    # Only use the first N samples (e.g., the Access Code) for estimation
    start_idx = active_indices[0]
    end_idx = min(start_idx + estimation_len, len(replica))
    
    r = replica[start_idx:end_idx]
    s = received_slot[start_idx:end_idx]

    # Least squares on the short window
    scale = np.dot(np.conj(r), s) / np.dot(np.conj(r), r)
    return scale


# ─────────────────────────────────────────────────────────────────────────────
#  SIC processor
# ─────────────────────────────────────────────────────────────────────────────


def find_frames(iq, N_slots, slot_samples, threshold=0.7):
    frame_len = N_slots * slot_samples
    min_gap   = slot_samples

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

        pkt, ac_off = decode_slot(iq[peak_idx:slot_end])   # ← unpack tuple

        if pkt is None or not pkt['crc_ok']:
            print(f"[SYNC] Peak {peak_idx} (val={peak_val:.3f}): decode failed, skipping.")
            search_pos = peak_idx + 1
            continue

        # Beacon: frame_seq==0 → data frame starts slot_samples later
        if pkt['frame_seq'] == 0:
            print(f"[SYNC] Peak {peak_idx} (val={peak_val:.3f}): BEACON "
                  f"(frame_seq=0) → next data slot1 at {peak_idx + slot_samples}")
            search_pos = peak_idx + slot_samples
            continue

        # Data packet: back-calculate frame start from this_slot
        fs = peak_idx - (pkt['this_slot'] - 1) * slot_samples
        fe = fs + frame_len

        print(f"[SYNC] Offset check: peak_idx={peak_idx} "
              f"fs={fs} diff={peak_idx - fs} "
              f"expected={(pkt['this_slot']-1)*slot_samples}")

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


# def process_frame_sic(frame_iq, N_slots, slot_samples, pkt_writer, sic_writer):

#     # ── Build AC-anchored residuals ───────────────────────────────────────
#     # For each slot, find where the AC actually starts and extract from there
#     residual   = []
#     ac_offsets = {}   # slot_idx → signed offset from nominal boundary

#     print(f"\n[SIC] ── New Frame {'─'*40}")
#     print(f"  [EXTRACT] Finding AC anchors for all slots...")

#     for s in range(N_slots):
#         chunk, ac_off = extract_slot_from_ac(frame_iq, s, slot_samples)
#         residual.append(chunk.copy())
#         ac_offsets[s] = ac_off

#     decoded_slots = set()
#     dirty_slots   = set()
#     all_decoded   = []
#     queue         = list(range(N_slots))
#     frame_id      = None

#     while queue:
#         user_ac_offsets = {} 

#         slot_idx = queue.pop(0)
#         if slot_idx in decoded_slots:
#             continue

#         pkt, _ = decode_slot(residual[slot_idx])  # ac_off already known

#         if pkt is None:
#             print(f"  slot={slot_idx+1:02d}  NO_AC")
#             sic_writer.writerow([frame_id, slot_idx+1, 'NO_AC', '-', '-', '-'])
#             continue

#         if pkt['crc_ok'] and pkt['frame_seq'] == 0:
#             print(f"  slot={slot_idx+1:02d}  BEACON (discarded)")
#             continue

#         if frame_id is None:
#             frame_id = pkt['frame_seq']

#         if not pkt['crc_ok']:
#             print(f"  slot={slot_idx+1:02d}  CRC_FAIL  user={pkt['user_id']}")
#             sic_writer.writerow([frame_id, slot_idx+1, 'CRC_FAIL',
#                                  pkt['user_id'], pkt['degree'], pkt['slot_list']])
#             if slot_idx in dirty_slots:
#                 queue.append(slot_idx)
#                 dirty_slots.discard(slot_idx)
#             continue

#         print(f"  slot={slot_idx+1:02d}  DECODED  user={pkt['user_id']}  "
#               f"d={pkt['degree']}  copies@{pkt['slot_list']}  "
#               f"ac_off={ac_offsets[slot_idx]:+d}")
#         sic_writer.writerow([frame_id, slot_idx+1, 'DECODED',
#                              pkt['user_id'], pkt['degree'], pkt['slot_list']])

#         pay_hex = ' '.join(f'{b:02X}' for b in pkt['payload'])
#         pkt_writer.writerow([
#             frame_id, slot_idx+1, pkt['user_id'], pkt['frame_seq'],
#             pkt['degree'], pkt['this_slot'], pkt['slot_list'],
#             pkt['crc_ok'], pay_hex,
#         ])

#         user_ac_offsets[pkt['user_id']] = ac_off   # ac_off from decode_slot
#         print(f"  [OFFSET] Learned user={pkt['user_id']} ac_off={ac_off:+d}")   

#         decoded_slots.add(slot_idx)
#         all_decoded.append(pkt)

#         for cancel_slot in pkt['slot_list']:

#             s0 = cancel_slot - 1
#             if s0 == slot_idx or s0 in decoded_slots:
#                 continue
#             if not (0 <= s0 < N_slots):
#                 continue

#             # Use the offset we learned from this user's clean decode
#             # This is reliable regardless of what else is in the target slot
#             target_ac_off = user_ac_offsets.get(pkt['user_id'], 0)

#             replica        = reconstruct_replica(pkt, cancel_slot,
#                                                 slot_samples, target_ac_off)
#             scale          = estimate_phase_scale(residual[s0], replica)
#             replica_scaled = scale * replica
#             residual[s0]  -= replica_scaled

#             dirty_slots.add(s0)
#             if s0 not in queue:
#                 queue.append(s0)
#             print(f"  CANCELLED slot={cancel_slot:02d}  "
#                 f"(user={pkt['user_id']}  ac_off={target_ac_off:+d}  "
#                 f"learned={'yes' if pkt['user_id'] in user_ac_offsets else 'fallback'})")

#     print(f"[SIC] Frame {frame_id} — decoded={len(all_decoded)}, "
#           f"resolved={len(decoded_slots)}/{N_slots} slots")
#     return all_decoded


def process_frame_sic(frame_iq, N_slots, slot_samples, pkt_writer, sic_writer):

    # ── Cancellation accumulator — one per slot, in frame coordinates ────
    # cancelled[s] holds the sum of all replicas subtracted from slot s,
    # aligned to the frame (not to the slot boundary), so we can re-extract
    # at any offset cleanly.
    cancelled = [np.zeros(slot_samples + 64, dtype=np.complex64)
                 for _ in range(N_slots)]   # +64 headroom for offset shifts

    def get_residual(s, ac_off):
        """
        Extract slot s from frame, subtract accumulated cancellations,
        starting from nominal_start + ac_off.
        """
        nominal   = s * slot_samples
        start     = nominal + ac_off
        end       = start + slot_samples
        if end > len(frame_iq):
            chunk = np.zeros(slot_samples, dtype=np.complex64)
            avail = len(frame_iq) - start
            if avail > 0:
                chunk[:avail] = frame_iq[start:start + avail]
        else:
            chunk = frame_iq[start:end].copy()

        # Subtract the portion of cancellations that aligns with this window
        # cancelled[s] is stored relative to nominal start
        cancel_start = ac_off          # offset into cancelled[s]
        cancel_end   = ac_off + slot_samples
        if cancel_start >= 0 and cancel_end <= len(cancelled[s]):
            chunk -= cancelled[s][cancel_start:cancel_end]
        return chunk

    def get_slot(s, ac_off):
        nominal   = s * slot_samples
        start     = nominal + ac_off
        end       = start + slot_samples
        if end > len(frame_iq):
            chunk = np.zeros(slot_samples, dtype=np.complex64)
            avail = len(frame_iq) - start
            if avail > 0:
                chunk[:avail] = frame_iq[start:start + avail]
        else:
            chunk = frame_iq[start:end].copy()
        
        return chunk

    def apply_cancellation(s0, replica, ac_off):
        """
        Store replica into cancelled[s0] at position ac_off so it can be
        subtracted from any future re-extraction of slot s0.
        """
        end = ac_off + len(replica)
        if end <= len(cancelled[s0]):
            cancelled[s0][ac_off:end] += replica

    decoded_slots  = set()
    dirty_slots    = set()
    all_decoded    = []
    queue          = list(range(N_slots))
    frame_id       = None
    user_ac_offsets = {}   # user_id → ac_offset from clean singleton decode

    print(f"\n[SIC] ── New Frame {'─'*40}")


    # Grab the raw IQ for all 3 slots using nominal offset (0)
    slot1_iq = get_residual(0, 0) # Collided slot (User 1 + User 2)
    slot2_iq = get_residual(1, 0) # Singleton slot (User 1)
    # slot3_iq = get_residual(2, 0) # Singleton slot (User 2)

    fig, axs = plt.subplots(3, 2, figsize=(18, 10))
    fig.suptitle("IRSA Frame Analysis: Collision vs. User 1 vs. User 2", fontsize=16)

    # --- Row 1: Time Domain Amplitude (Envelope) ---
    axs[0, 0].plot(np.abs(slot1_iq), color='red', alpha=0.8)            # np.abs
    axs[0, 0].set_title("Slot 1: Amplitude (U1 + U2 Collision)")
    axs[0, 0].set_ylabel("Magnitude")

    axs[0, 1].plot(np.abs(slot2_iq), color='blue', alpha=0.8)
    axs[0, 1].set_title("Slot 2: Amplitude (User 1 Singleton)")

    # axs[0, 2].plot(np.abs(slot3_iq), color='green', alpha=0.8)
    # axs[0, 2].set_title("Slot 3: Amplitude (User 2 Singleton)")

    # --- Row 2: Phase Trajectory (Check for frequency offset) ---
    axs[1, 0].plot(np.unwrap(np.angle(slot1_iq)), color='red', alpha=0.8)
    axs[1, 0].set_title("Slot 1: Phase Trajectory")
    axs[1, 0].set_ylabel("Radians")

    axs[1, 1].plot(np.unwrap(np.angle(slot2_iq)), color='blue', alpha=0.8)
    axs[1, 1].set_title("Slot 2: Phase (User 1)")

    # axs[1, 2].plot(np.unwrap(np.angle(slot3_iq)), color='green', alpha=0.8)
    # axs[1, 2].set_title("Slot 3: Phase (User 2)")

    # --- Row 3: IQ Constellation Scatter ---
    symbols1 = slot1_iq[::SPS]
    axs[2, 0].scatter(np.real(symbols1), np.imag(symbols1), color='red', alpha=0.3, s=10)
    axs[2, 0].set_title("Slot 1: Constellation")
    axs[2, 0].axhline(0, color='black', lw=0.5); axs[2, 0].axvline(0, color='black', lw=0.5)

    symbols2 = slot2_iq[::SPS]
    axs[2, 1].scatter(np.real(symbols2), np.imag(symbols2), color='blue', alpha=0.3, s=10)
    axs[2, 1].set_title("Slot 2: Constellation (User 1)")
    axs[2, 1].axhline(0, color='black', lw=0.5); axs[2, 1].axvline(0, color='black', lw=0.5)

    # symbols3 = slot3_iq[::SPS]
    # axs[2, 2].scatter(np.real(symbols3), np.imag(symbols3), color='green', alpha=0.3, s=10)
    # axs[2, 2].set_title("Slot 3: Constellation (User 2)")
    # axs[2, 2].axhline(0, color='black', lw=0.5); axs[2, 2].axvline(0, color='black', lw=0.5)

    # Ensure y-axis limits for amplitude are the same across all 3 slots for fair power comparison
    max_amp = max(np.max(np.abs(slot1_iq)), np.max(np.abs(slot2_iq)))        # , np.max(np.abs(slot3_iq))
    axs[0, 0].set_ylim(0, max_amp * 1.1)
    axs[0, 1].set_ylim(0, max_amp * 1.1)
    # axs[0, 2].set_ylim(0, max_amp * 1.1)

    plt.tight_layout()
    plt.savefig('frame.png', dpi=600, bbox_inches='tight')
    plt.show()



    while queue:
        slot_idx = queue.pop(0)
        if slot_idx in decoded_slots:
            continue

        # Use learned offset for this slot's dominant user if known,
        # otherwise scan with offset=0 (nominal)
        # For collided slots on first pass, we may not know yet — that's OK,
        # SIC will come back after cancellation with the correct user offset.
        probe_off = 0   # start with nominal; will improve after first decode
        residual_chunk = get_residual(slot_idx, probe_off)

        pkt, ac_off = decode_slot(residual_chunk)

        if pkt is None:
            if slot_idx in dirty_slots:   # only if cancellation was applied
                debug_residual_slot(slot_idx, frame_iq, cancelled,
                                    slot_samples, user_ac_offsets,
                                    label="after cancellation")
                                    
            # If we know a user who has a copy here, try their offset explicitly
            helped = False
            for uid, uoff in user_ac_offsets.items():
                if uoff == probe_off:
                    continue
                residual_chunk2 = get_residual(slot_idx, uoff)
                pkt2, ac_off2   = decode_slot(residual_chunk2)
                if pkt2 and pkt2['crc_ok']:
                    pkt    = pkt2
                    ac_off = uoff
                    helped = True
                    print(f"  slot={slot_idx+1:02d}  recovered using "
                          f"user={uid} offset={uoff:+d}")
                    break
            if not helped:
                print(f"  slot={slot_idx+1:02d}  NO_AC")
                sic_writer.writerow([frame_id, slot_idx+1, 'NO_AC', '-', '-', '-'])
                continue
            

        if pkt['crc_ok'] and pkt['frame_seq'] == 0:
            print(f"  slot={slot_idx+1:02d}  BEACON (discarded)")
            continue

        if frame_id is None:
            frame_id = pkt['frame_seq']

        if not pkt['crc_ok']:
            print(f"[DEBUG] user_id={pkt['user_id']} frame_seq={pkt['frame_seq']}") 
            print(f"[DEBUG] degree={pkt['degree']} this_slot={pkt['this_slot']}") 
            print(f"[DEBUG] slot_list={pkt['slot_list']}") 

            print(f"  slot={slot_idx+1:02d}  CRC_FAIL  user={pkt['user_id']}")
            sic_writer.writerow([frame_id, slot_idx+1, 'CRC_FAIL',
                                 pkt['user_id'], pkt['degree'], pkt['slot_list']])
            if slot_idx in dirty_slots:
                queue.append(slot_idx)
                dirty_slots.discard(slot_idx)
            continue

        # Clean decode — record this user's offset
        if pkt['user_id'] not in user_ac_offsets:
            user_ac_offsets[pkt['user_id']] = ac_off
            print(f"  [OFFSET] Learned user={pkt['user_id']} ac_off={ac_off:+d}")

        print(f"  slot={slot_idx+1:02d}  DECODED  user={pkt['user_id']}  "
              f"d={pkt['degree']}  copies@{pkt['slot_list']}  ac_off={ac_off:+d}")
        sic_writer.writerow([frame_id, slot_idx+1, 'DECODED',
                             pkt['user_id'], pkt['degree'], pkt['slot_list']])

        pay_hex = ' '.join(f'{b:02X}' for b in pkt['payload'])
        pkt_writer.writerow([
            frame_id, slot_idx+1, pkt['user_id'], pkt['frame_seq'],
            pkt['degree'], pkt['this_slot'], pkt['slot_list'],
            pkt['crc_ok'], pay_hex,
        ])

        decoded_slots.add(slot_idx)
        all_decoded.append(pkt)

        clean_replica = reconstruct_replica(pkt, pkt['this_slot'], slot_samples, ac_offset=0)
        residual_check = get_slot(pkt['this_slot']-1, ac_off)
        scale = estimate_phase_scale(residual_check, clean_replica)
        print(f'The scale we got from correctly decoded slot: {scale}\n')

        for cancel_slot in pkt['slot_list']:
            s0 = cancel_slot - 1
            if s0 == slot_idx or s0 in decoded_slots:
                continue
            if not (0 <= s0 < N_slots):
                continue

            # Use the offset we KNOW is correct for this user
            target_ac_off = user_ac_offsets[pkt['user_id']]

            replica        = reconstruct_replica(pkt, cancel_slot, slot_samples, ac_offset=0)

            # get_residual already aligns to target_ac_off, so replica
            # starts at index 0 of whatever get_residual returns

            # residual_check = get_residual(s0, target_ac_off)
            # scale          = estimate_phase_scale(residual_check, replica)
            # replica_scaled = scale * replica


            # ------------------------------ INJECT DEBUG PLOTTING HERE ---------------------------
            if pkt['user_id'] == 2: # Only plot for User 1 to avoid spamming your screen
                fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                
                # Plot 1: Magnitude (Envelope)
                axs[0].plot(np.abs(residual_check), label='Received Collision (User 1 + User 2)', alpha=0.7)
                axs[0].plot(np.abs(replica), label='Replica (User 1)', linestyle='--')                      # replica_scaled
                axs[0].set_title("Amplitude / Envelope")
                axs[0].legend()

                # Plot 2: Real Component (Phase alignment)
                axs[1].plot(np.real(residual_check), label='Received Real', alpha=0.7)
                axs[1].plot(np.real(replica), label='Replica Real', linestyle='--')         # replica_scaled
                axs[1].set_title("Real Component (Phase Check)")
                axs[1].legend()

                # Plot 3: The Ghost (What is left behind for User 2)
                ghost_signal = residual_check - replica                 # replica_scaled
                axs[2].plot(np.abs(ghost_signal), color='red', label='Residual "Ghost" Signal')
                axs[2].set_title("After Cancellation (Should be User 2 only)")
                axs[2].legend()

                plt.tight_layout()
                plt.savefig('collision.png', dpi=600, bbox_inches='tight')
                plt.show()
            # ----------------------------------



            # apply_cancellation(s0, replica_scaled, target_ac_off)
            apply_cancellation(s0, replica, target_ac_off)

            dirty_slots.add(s0)
            if s0 not in queue:
                queue.append(s0)
            print(f"  CANCELLED slot={cancel_slot:02d}  "
                  f"(user={pkt['user_id']}  ac_off={target_ac_off:+d})")
            sic_writer.writerow([frame_id, cancel_slot, 'CANCELLED',
                                 pkt['user_id'], pkt['degree'], pkt['slot_list']])

    print(f"[SIC] Frame {frame_id} — decoded={len(all_decoded)}, "
          f"resolved={len(decoded_slots)}/{N_slots} slots")
    return all_decoded


def debug_residual_slot(slot_idx, frame_iq, cancelled, slot_samples,
                        user_ac_offsets, label=""):
    """
    Try every known user offset AND a blind scan to find any surviving AC.
    Prints what is left in the residual at each possible offset.
    """
    print(f"\n  [RESIDUAL DEBUG] slot={slot_idx+1}  {label}")
    nominal = slot_idx * slot_samples

    # Try offset=0, all known user offsets, and a blind scan 0..30
    offsets_to_try = sorted(set([0] + list(user_ac_offsets.values()) +
                                list(range(0, 32))))

    for off in offsets_to_try:
        chunk = _get_residual_chunk(frame_iq, cancelled, slot_idx,
                                    slot_samples, off)
        # Check best AC match
        symbols  = chunk[::SPS]
        bits_raw = bpsk_demodulate(symbols)
        best_hits, best_pos = 0, 0
        for init in [0, 1]:
            bits   = differential_decode(bits_raw, init)
            ac_b   = bytes_to_bits(ACCESS_CODE_BYTES)
            for i in range(min(len(bits) - len(ac_b) + 1, 50)):
                h = int(np.sum(bits[i:i+len(ac_b)] == ac_b))
                if h > best_hits:
                    best_hits, best_pos = h, i
        energy = np.mean(np.abs(chunk)**2)
        if best_hits >= 24:   # print anything promising
            print(f"    off={off:+3d}  energy={energy:.4f}  "
                  f"best_AC={best_hits}/32 @ bit_pos={best_pos}")


def _get_residual_chunk(frame_iq, cancelled, s, slot_samples, ac_off):
    """Helper: extract slot s from frame at ac_off and subtract cancellations."""
    nominal = s * slot_samples
    start   = nominal + ac_off
    end     = start + slot_samples
    if end > len(frame_iq):
        chunk = np.zeros(slot_samples, dtype=np.complex64)
        avail = max(0, len(frame_iq) - start)
        chunk[:avail] = frame_iq[start:start+avail]
    else:
        chunk = frame_iq[start:end].copy()
    # subtract accumulated cancellations
    c_start = ac_off
    c_end   = ac_off + slot_samples
    if 0 <= c_start and c_end <= len(cancelled[s]):
        chunk -= cancelled[s][c_start:c_end]
    return chunk


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