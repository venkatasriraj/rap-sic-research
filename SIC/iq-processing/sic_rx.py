"""
irsa_rx.py  —  Simplified offline IRSA receiver
=================================================
Handles exactly what the TX sends:
  • One beacon frame  (frame_seq = 0, slot 1 only)
  • One data frame    (N_slots slots, users transmit copies)

Slot offset problem
-------------------
The TX fires each slot timer at  t_frame + (slot-1)*slot_duration.
Timing jitter means the packet may start a few samples late inside
the nominal slot window.  We handle this by:
  1. Correlating the AC template inside a ±SEARCH_RADIUS sample window
     around each nominal slot boundary to find the true start.
  2. Once a packet is cleanly decoded (CRC ok), we record that user's
     exact AC offset.  We reuse that offset when cancelling the same
     user's replica in the other slot.

SIC (Successive Interference Cancellation) — one frame
-------------------------------------------------------
Pass 1: decode each slot independently.
        Singleton slots decode immediately.
        Collided slots fail CRC.

Pass 2 (repeat until stable):
        For each slot we just decoded, subtract a phase/amplitude-matched
        replica from every OTHER slot that packet claims to occupy.
        Then retry decoding the dirty slots.

Usage
-----
  python irsa_rx.py costas_out.bin [N_slots] [slot_samples] [threshold]

  defaults: N_slots=2, slot_samples=16000, threshold=0.7

Input  : complex64 IQ (GNU Radio File Sink after Costas Loop)
Outputs: decoded_packets.csv
         sic_log.csv
"""

import sys
import csv
import binascii
import random
import numpy as np
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────────────────────
#  Constants — must match TX exactly
# ─────────────────────────────────────────────────────────────────────────────

ACCESS_CODE   = [0xE1, 0x5A, 0xE8, 0x93]   # 4 bytes = 32 bits
FILLER        = [0xDE, 0xAD, 0xBE, 0xEE,
                 0xDE, 0xAD, 0xBE, 0xEE]    # 8 bytes, flushes pipeline
MAX_DEGREE    = 16                           # max copies per user per frame
CRC_BYTES     = 4
FILLER_LEN    = 8
PACKET_SIZE   = 100                          # bytes

# Derived: payload fills whatever is left after the fixed header
# Header: AC(4) + uid(2) + seq(2) + deg(1) + slot(1) + slot_list(16) + CRC(4) + filler(8)
HEADER_FIXED  = (len(ACCESS_CODE) + 2 + 2 + 1 + 1 +
                 MAX_DEGREE + CRC_BYTES + FILLER_LEN)   # = 38
PAYLOAD_LEN   = PACKET_SIZE - HEADER_FIXED               # = 62

SPS           = 1                 # samples per symbol (1 after Costas + matched filter)
AC_THRESHOLD  = 30                # minimum bit matches out of 32 to accept AC
SEARCH_RADIUS = 30                # ± samples around nominal slot boundary to hunt AC
DIFF_INIT     = 0                 # differential encoder initial state


# ─────────────────────────────────────────────────────────────────────────────
#  Bit / symbol helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def diff_encode(bits, init=DIFF_INIT):
    out  = np.empty(len(bits), dtype=np.uint8)
    prev = init
    for i, b in enumerate(bits):
        out[i] = int(b) ^ prev
        prev   = out[i]
    return out


def diff_decode(bits, init=DIFF_INIT):
    bits = np.asarray(bits, dtype=np.uint8)
    prev = np.empty(len(bits), dtype=np.uint8)
    prev[0]  = init
    prev[1:] = bits[:-1]
    return (bits ^ prev).astype(np.uint8)


def bpsk_mod(bits):
    # 0 → +1,  1 → -1
    return (1.0 - 2.0 * np.asarray(bits, dtype=np.float32)).astype(np.complex64)


def bpsk_demod(symbols):
    # real < 0 → bit 1
    return (np.asarray(symbols).real < 0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  AC template for correlation (same construction as TX)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ac_template(init=DIFF_INIT):
    """32-symbol complex BPSK template for the access code."""
    bits    = bytes_to_bits(ACCESS_CODE)
    enc     = diff_encode(bits, init)
    symbols = bpsk_mod(enc)
    symbols /= np.max(np.abs(symbols))
    return symbols


# Pre-build both possible templates (init=0 and init=1)
_TEMPLATES = [_build_ac_template(0), _build_ac_template(1)]


def _norm_xcorr(signal, template):
    """Normalised cross-correlation magnitude, values in [0, 1]."""
    corr       = np.correlate(signal, template, mode='valid')
    mag        = np.abs(corr)
    tlen       = len(template)
    sig_energy = np.sqrt(np.maximum(
        np.convolve(np.abs(signal)**2, np.ones(tlen), mode='valid'), 1e-12))
    tmpl_norm  = np.sqrt(np.sum(np.abs(template)**2))
    return mag / (sig_energy * tmpl_norm + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
#  Slot extraction — find the AC within a search window
# ─────────────────────────────────────────────────────────────────────────────

def extract_slot(frame_iq, slot_idx, slot_samples, hint_offset=None):
    """
    Find where the AC actually starts in slot `slot_idx` and return
    `slot_samples` of IQ beginning at that AC peak.

    hint_offset: if we already know this user's offset from a previous
                 clean decode, skip correlation and use it directly.

    Returns
    -------
    chunk      : complex64 array, length slot_samples
    ac_offset  : signed sample offset from the nominal slot boundary
    """
    nominal = slot_idx * slot_samples

    if hint_offset is not None:
        # Use the known offset — no need to correlate again
        start = nominal + hint_offset
        end   = start + slot_samples
        chunk = _safe_slice(frame_iq, start, end, slot_samples)
        return chunk, hint_offset

    # Grab a window ±SEARCH_RADIUS around the nominal boundary
    win_start = max(0, nominal - SEARCH_RADIUS)
    win_end   = min(len(frame_iq), nominal + SEARCH_RADIUS + slot_samples)
    window    = frame_iq[win_start:win_end]

    # Correlate with both templates and take the max
    corr = np.maximum(_norm_xcorr(window, _TEMPLATES[0]),
                      _norm_xcorr(window, _TEMPLATES[1]))

    # Only consider positions within ±SEARCH_RADIUS of nominal
    local_limit = min(2 * SEARCH_RADIUS + 1, len(corr))
    best_local  = int(np.argmax(corr[:local_limit]))

    ac_in_frame = win_start + best_local
    ac_offset   = ac_in_frame - nominal           # signed offset

    chunk = _safe_slice(frame_iq, ac_in_frame, ac_in_frame + slot_samples,
                        slot_samples)

    print(f"  [EXTRACT] slot={slot_idx+1}  nominal={nominal}  "
          f"ac_peak={ac_in_frame}  offset={ac_offset:+d}  "
          f"corr={corr[best_local]:.4f}")

    return chunk, ac_offset


def _safe_slice(iq, start, end, length):
    """Slice IQ with zero-padding if we run off the end."""
    if end <= len(iq):
        return iq[start:end].copy()
    out   = np.zeros(length, dtype=np.complex64)
    avail = max(0, len(iq) - start)
    out[:avail] = iq[start:start + avail]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Packet decode — one slot's worth of IQ → parsed packet dict
# ─────────────────────────────────────────────────────────────────────────────

def decode_slot(slot_iq):
    """
    Demodulate, differentially decode, find AC, parse header, check CRC.

    Returns (pkt_dict, ac_bit_offset) on success, (None, -1) on failure.
    ac_bit_offset is the bit position where the AC starts (used to
    compute the sample offset).
    """
    symbols  = slot_iq[::SPS]
    bits_raw = bpsk_demod(symbols)
    ac_bits  = bytes_to_bits(ACCESS_CODE)
    ac_len   = len(ac_bits)

    for init in [0, 1]:
        bits = diff_decode(bits_raw, init)

        # Scan for AC within the first ~50 bit positions
        # (offset is small — typically < SEARCH_RADIUS bits)
        for i in range(min(len(bits) - ac_len, SEARCH_RADIUS * SPS + 1)):
            if int(np.sum(bits[i:i+ac_len] == ac_bits)) >= AC_THRESHOLD:
                pkt = _parse_packet(bits[i + ac_len:])
                if pkt and pkt['crc_ok']:
                    return pkt, i      # i = AC start in bits = sample offset (SPS=1)

    return None, -1


def _parse_packet(bits_after_ac):
    """
    Parse the fixed IRSA header that follows the access code.
    Returns a dict or None if the byte count is too short.
    """
    raw = bits_to_bytes(bits_after_ac)

    # Minimum: uid(2)+seq(2)+deg(1)+slot(1)+slot_list(16)+payload(62)+CRC(4)+filler(8)
    min_bytes = 2 + 2 + 1 + 1 + MAX_DEGREE + PAYLOAD_LEN + CRC_BYTES + FILLER_LEN
    if len(raw) < min_bytes:
        return None

    idx       = 0
    user_id   = (raw[idx] << 8) | raw[idx+1];  idx += 2
    frame_seq = (raw[idx] << 8) | raw[idx+1];  idx += 2
    degree    = raw[idx];                        idx += 1
    this_slot = raw[idx];                        idx += 1
    slot_list = [s for s in raw[idx:idx+MAX_DEGREE] if s != 0]; idx += MAX_DEGREE
    payload   = raw[idx:idx+PAYLOAD_LEN];        idx += PAYLOAD_LEN
    crc_rx    = raw[idx:idx+CRC_BYTES]

    # CRC covers everything from uid through end of payload
    protected    = raw[:2+2+1+1+MAX_DEGREE+PAYLOAD_LEN]
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


# ─────────────────────────────────────────────────────────────────────────────
#  Replica reconstruction for SIC
# ─────────────────────────────────────────────────────────────────────────────

def build_replica(pkt, target_slot, slot_samples, ac_offset=0):
    """
    Reconstruct the ideal IQ for this packet as it would appear in
    `target_slot`, starting at sample `ac_offset` within the slot window.

    This re-encodes the exact same bytes the TX sent, so subtracting it
    (after phase/amplitude matching) removes the user's contribution.
    """
    uid_b    = [(pkt['user_id']   >> 8) & 0xFF, pkt['user_id']   & 0xFF]
    seq_b    = [(pkt['frame_seq'] >> 8) & 0xFF, pkt['frame_seq'] & 0xFF]
    sl_field = [s & 0xFF for s in pkt['slot_list']]
    sl_field += [0x00] * (MAX_DEGREE - len(sl_field))

    protected = (uid_b + seq_b + [pkt['degree'] & 0xFF] +
                 [target_slot & 0xFF] + sl_field + list(pkt['payload']))
    crc_val   = binascii.crc32(bytes(protected)) & 0xFFFFFFFF
    crc_b     = list(crc_val.to_bytes(4, byteorder='little'))

    pkt_bytes = ACCESS_CODE + protected + crc_b + FILLER
    symbols   = bpsk_mod(diff_encode(bytes_to_bits(pkt_bytes)))

    # Place the replica at the correct sample offset inside the slot window
    replica = np.zeros(slot_samples, dtype=np.complex64)
    end     = ac_offset + len(symbols)
    if end <= slot_samples:
        replica[ac_offset:end] = symbols
    else:
        replica[ac_offset:] = symbols[:slot_samples - ac_offset]
    return replica


def match_scale(received, replica):
    """
    Compute complex scale = (conj(replica) · received) / (conj(replica) · replica)
    so that  received ≈ scale * replica.
    Only uses positions where replica is non-zero (actual signal).
    """
    mask = np.abs(replica) > 0.1
    if mask.sum() < 10:
        return complex(1.0, 0.0)
    r = replica[mask]
    s = received[mask]
    return np.dot(r.conj(), s) / np.dot(r.conj(), r)


# ─────────────────────────────────────────────────────────────────────────────
#  Frame sync — find frame boundaries in the raw IQ stream
# ─────────────────────────────────────────────────────────────────────────────

def find_frames(iq, N_slots, slot_samples, threshold=0.7):
    """
    Slide the AC template across the full IQ stream.
    Each strong peak that decodes successfully tells us where a slot starts.
    From `this_slot` in the packet header we back-calculate the frame start.

    Returns a list of frame IQ arrays (each length N_slots * slot_samples).
    """
    frame_len = N_slots * slot_samples

    # Correlate once across the entire stream
    corr = np.maximum(_norm_xcorr(iq, _TEMPLATES[0]),
                      _norm_xcorr(iq, _TEMPLATES[1]))

    peaks, _ = find_peaks(corr, height=threshold, distance=slot_samples)
    peaks    = peaks[np.argsort(peaks)]

    print(f"[SYNC] {len(peaks)} peaks above threshold={threshold}")

    frames     = []
    search_pos = 0   # don't re-use samples before this point

    for peak in peaks:
        if peak < search_pos:
            continue

        pkt, _ = decode_slot(iq[peak:peak + slot_samples])
        if pkt is None or not pkt['crc_ok']:
            continue

        # Beacon (frame_seq == 0): skip — it's just the Costas loop warm-up
        if pkt['frame_seq'] == 0:
            print(f"[SYNC] Beacon at sample {peak}, skipping.")
            search_pos = peak + slot_samples
            continue

        # Back-calculate frame start from this_slot
        frame_start = peak - (pkt['this_slot'] - 1) * slot_samples
        frame_end   = frame_start + frame_len

        if frame_start < 0 or frame_end > len(iq):
            print(f"[SYNC] Frame [{frame_start}:{frame_end}] out of bounds, skipping.")
            continue

        frames.append(iq[frame_start:frame_end].copy())
        search_pos = frame_end

        print(f"[SYNC] Frame {len(frames):03d}: samples [{frame_start}:{frame_end}]  "
              f"user={pkt['user_id']}  seq={pkt['frame_seq']}  "
              f"this_slot={pkt['this_slot']}  peak_corr={corr[peak]:.3f}")

    print(f"[SYNC] Found {len(frames)} data frame(s).")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
#  SIC — process one frame
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(frame_iq, N_slots, slot_samples, pkt_writer, sic_writer):
    """
    Successive Interference Cancellation for one frame.

    State we track per slot:
      residual[s]   — accumulated replicas subtracted so far (in slot coordinates)
      ac_offsets[s] — AC sample offset used when last extracting slot s

    Each iteration:
      1. Try to decode every unresolved slot.
      2. Any clean decode → subtract replica from sibling slots → retry them.
      3. Stop when no new decodes happen.
    """

    # Per-slot cancellation accumulator, stored relative to nominal boundary
    cancelled   = [np.zeros(slot_samples + SEARCH_RADIUS + 64, dtype=np.complex64)
                   for _ in range(N_slots)]

    # Track per-user AC offset once cleanly decoded
    user_offsets = {}    # uid → sample offset

    decoded_slots = set()
    frame_id      = None

    # ── helper: get residual IQ for slot s at a given ac_offset ──────────
    def get_residual(s, ac_off):
        chunk = _safe_slice(frame_iq,
                            s * slot_samples + ac_off,
                            s * slot_samples + ac_off + slot_samples,
                            slot_samples)
        # Subtract whatever has been cancelled so far (aligned to ac_off)
        c_end = ac_off + slot_samples
        if 0 <= ac_off and c_end <= len(cancelled[s]):
            chunk = chunk - cancelled[s][ac_off:c_end]
        return chunk

    # ── helper: record a cancellation for slot s ──────────────────────────
    def add_cancellation(s, replica_scaled, ac_off):
        c_end = ac_off + len(replica_scaled)
        if c_end <= len(cancelled[s]):
            cancelled[s][ac_off:c_end] += replica_scaled

    print(f"\n[SIC] ── Frame ──────────────────────────────────────────────")

    # Iterate until nothing new is decoded
    max_passes = N_slots + 1
    for pass_num in range(max_passes):
        new_this_pass = 0

        for s in range(N_slots):
            if s in decoded_slots:
                continue

            # Use known user offset if we've seen this user before,
            # otherwise do a blind correlation search
            known_off = None
            for uid, off in user_offsets.items():
                # We don't know which user is here yet, so we can't pick —
                # blind search on first attempt, use hint on retry after cancel
                pass

            chunk, ac_off = extract_slot(frame_iq, s, slot_samples)

            # Subtract accumulated cancellations at this offset
            c_end = ac_off + slot_samples
            if 0 <= ac_off and c_end <= len(cancelled[s]):
                chunk = chunk - cancelled[s][ac_off:c_end]

            pkt, pkt_ac_off = decode_slot(chunk)

            if pkt is None:
                print(f"  pass={pass_num+1}  slot={s+1}  NO_DECODE")
                sic_writer.writerow([frame_id or '?', s+1, 'NO_DECODE', '-', '-', '-'])
                continue

            if pkt['frame_seq'] == 0:
                print(f"  pass={pass_num+1}  slot={s+1}  BEACON (skipped)")
                continue

            if frame_id is None:
                frame_id = pkt['frame_seq']

            if not pkt['crc_ok']:
                print(f"  pass={pass_num+1}  slot={s+1}  CRC_FAIL  user={pkt['user_id']}")
                sic_writer.writerow([frame_id, s+1, 'CRC_FAIL',
                                     pkt['user_id'], pkt['degree'], pkt['slot_list']])
                continue

            # ── Clean decode ───────────────────────────────────────────────
            # pkt_ac_off is the bit offset inside chunk where AC was found;
            # plus the offset chunk starts at (ac_off) gives the frame offset.
            true_ac_off = ac_off + pkt_ac_off

            if pkt['user_id'] not in user_offsets:
                user_offsets[pkt['user_id']] = true_ac_off
                print(f"  [OFFSET] user={pkt['user_id']}  ac_off={true_ac_off:+d}")

            print(f"  pass={pass_num+1}  slot={s+1}  OK  "
                  f"user={pkt['user_id']}  d={pkt['degree']}  "
                  f"copies@{pkt['slot_list']}  ac_off={true_ac_off:+d}")

            sic_writer.writerow([frame_id, s+1, 'DECODED',
                                 pkt['user_id'], pkt['degree'], pkt['slot_list']])
            pkt_writer.writerow([
                frame_id, s+1, pkt['user_id'], pkt['frame_seq'],
                pkt['degree'], pkt['this_slot'], pkt['slot_list'],
                pkt['crc_ok'],
                ' '.join(f'{b:02X}' for b in pkt['payload']),
            ])

            decoded_slots.add(s)
            new_this_pass += 1

            # ── Cancel this user's replica from all sibling slots ──────────
            for copy_slot in pkt['slot_list']:
                t = copy_slot - 1              # 0-indexed target slot
                if t == s or t in decoded_slots:
                    continue
                if not (0 <= t < N_slots):
                    continue

                # Replica uses the same ac_offset we found for this user
                target_off = user_offsets[pkt['user_id']]
                replica    = build_replica(pkt, copy_slot, slot_samples,
                                           ac_offset=target_off - t * slot_samples
                                           if False else 0)
                # Get the residual at the target offset to measure scale
                residual_t = get_residual(t, target_off)
                scale      = match_scale(residual_t, replica)
                add_cancellation(t, scale * replica, target_off)

                print(f"  CANCEL slot={copy_slot}  "
                      f"user={pkt['user_id']}  ac_off={target_off:+d}  "
                      f"|scale|={abs(scale):.3f}  ∠={np.degrees(np.angle(scale)):+.1f}°")
                sic_writer.writerow([frame_id, copy_slot, 'CANCELLED',
                                     pkt['user_id'], pkt['degree'], pkt['slot_list']])

        if new_this_pass == 0:
            print(f"  [SIC] No new decodes in pass {pass_num+1}, stopping.")
            break

    print(f"[SIC] Done — decoded {len(decoded_slots)}/{N_slots} slots")
    return decoded_slots


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def run(iq_file, N_slots=2, slot_samples=16000, threshold=0.7,
        pkt_log='decoded_packets.csv', sic_log='sic_log.csv'):

    print(f"[MAIN] Loading {iq_file} ...")
    iq = np.fromfile(iq_file, dtype=np.complex64)
    print(f"[MAIN] {len(iq)} samples loaded")

    frames = find_frames(iq, N_slots, slot_samples, threshold)
    if not frames:
        print("[MAIN] No frames found — check threshold or input file.")
        return

    total_decoded = 0

    with open(pkt_log, 'w', newline='') as pf, \
         open(sic_log, 'w', newline='') as sf:

        pw = csv.writer(pf)
        sw = csv.writer(sf)

        pw.writerow(['frame_id', 'slot', 'user_id', 'frame_seq',
                     'degree', 'this_slot', 'slot_list', 'crc_ok', 'payload_hex'])
        sw.writerow(['frame_id', 'slot', 'event', 'user_id', 'degree', 'slot_list'])

        for frame_iq in frames:
            resolved = process_frame(frame_iq, N_slots, slot_samples, pw, sw)
            total_decoded += len(resolved)

    print(f"\n[MAIN] Total decoded: {total_decoded} packets")
    print(f"[MAIN] Packet log  → {pkt_log}")
    print(f"[MAIN] SIC log     → {sic_log}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python irsa_rx.py <costas_out.bin> "
              "[N_slots] [slot_samples] [threshold]")
        sys.exit(1)

    run(
        iq_file      = sys.argv[1],
        N_slots      = int(sys.argv[2])   if len(sys.argv) > 2 else 2,
        slot_samples = int(sys.argv[3])   if len(sys.argv) > 3 else 16000,
        threshold    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7,
    )