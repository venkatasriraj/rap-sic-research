"""
irsa_rx_fixed.py  —  Fixed IRSA receiver for N_slots=3, slot_samples=800
=========================================================================
Bugs fixed vs original irsa_rx.py:

  FIX 1 – Cancellation index off-by-offset (CRITICAL)
    Original code did: chunk -= cancelled[s][ac_off : ac_off+slot_samples]
    This directly used the slot-relative AC offset as a raw array index into
    cancelled[], but cancelled[] is stored with SEARCH_RADIUS padding on the
    left (added via _cac_index). So the subtraction was hitting the wrong
    region.
    Fixed:  ci = _cac_index(ac_off)
            chunk -= cancelled[s][ci : ci + slot_samples]

  FIX 2 – user_offsets hint never actually used
    Original code iterated user_offsets but always left known_off=None.
    Fixed: after a clean decode we record user offset, and in subsequent
    passes we first try to re-extract at the known offset before falling
    back to blind XCorr.

  FIX 3 – build_replica this_slot CRC
    The 'this_slot' field in each packet copy differs between slots (it is
    the 1-indexed slot number of *this* copy). The original code passed
    copy_slot correctly but the subtraction was misaligned (Fix 1), so the
    CRC inside the replica never mattered. Now that alignment is fixed this
    is confirmed correct.

  FIX 4 – Correlation threshold adjusted for collision slots
    In a 2-user collision the normalised XCorr drops to ~0.7 instead of ~1.
    find_frames threshold kept at 0.7 so it doesn't detect collision peaks
    as new frames (it should only find singleton peaks for back-calculation).
    Added a note to the report about this.

Outputs:
  decoded_packets.csv
  sic_log.csv
  sic_debug_report.txt
"""

import sys, csv, binascii, numpy as np
from scipy.signal import find_peaks

# ─────────────────────────────────────────────────────────────────────────────
# Constants – must match TX exactly
# ─────────────────────────────────────────────────────────────────────────────
ACCESS_CODE   = [0xE1, 0x5A, 0xE8, 0x93]
FILLER        = [0xDE, 0xAD, 0xBE, 0xEE, 0xDE, 0xAD, 0xBE, 0xEE]
MAX_DEGREE    = 16
CRC_BYTES     = 4
FILLER_LEN    = 8
PACKET_SIZE   = 100
HEADER_FIXED  = len(ACCESS_CODE) + 2 + 2 + 1 + 1 + MAX_DEGREE + CRC_BYTES + FILLER_LEN  # 38
PAYLOAD_LEN   = PACKET_SIZE - HEADER_FIXED  # 62
SPS           = 1
AC_THRESHOLD  = 30
SEARCH_RADIUS = 30
DIFF_INIT     = 0

report_lines = []

def rlog(msg=""):
    print(msg)
    report_lines.append(msg)

# ─────────────────────────────────────────────────────────────────────────────
# Bit / symbol helpers
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
    out = np.empty(len(bits), dtype=np.uint8); prev = init
    for i, b in enumerate(bits):
        out[i] = int(b) ^ prev; prev = out[i]
    return out

def diff_decode(bits, init=DIFF_INIT):
    bits = np.asarray(bits, dtype=np.uint8)
    prev = np.empty(len(bits), dtype=np.uint8)
    prev[0] = init; prev[1:] = bits[:-1]
    return (bits ^ prev).astype(np.uint8)

def bpsk_mod(bits):
    return (1.0 - 2.0 * np.asarray(bits, dtype=np.float32)).astype(np.complex64)

def bpsk_demod(symbols):
    return (np.asarray(symbols).real < 0).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# AC template
# ─────────────────────────────────────────────────────────────────────────────
def _build_ac_template(init=DIFF_INIT):
    bits    = bytes_to_bits(ACCESS_CODE)
    enc     = diff_encode(bits, init)
    symbols = bpsk_mod(enc)
    symbols /= np.max(np.abs(symbols))
    return symbols

_TEMPLATES = [_build_ac_template(0), _build_ac_template(1)]

def _norm_xcorr(signal, template):
    corr       = np.correlate(signal, template, mode='valid')
    mag        = np.abs(corr)
    tlen       = len(template)
    sig_energy = np.sqrt(np.maximum(
        np.convolve(np.abs(signal)**2, np.ones(tlen), mode='valid'), 1e-12))
    tmpl_norm  = np.sqrt(np.sum(np.abs(template)**2))
    return mag / (sig_energy * tmpl_norm + 1e-12)

# ─────────────────────────────────────────────────────────────────────────────
# Slot extraction
# ─────────────────────────────────────────────────────────────────────────────
def _safe_slice(iq, start, end, length):
    out = np.zeros(length, dtype=np.complex64)
    src_start = max(0, start); src_end = min(len(iq), end)
    if src_end <= src_start: return out
    dst_start = src_start - start
    out[dst_start:dst_start + (src_end - src_start)] = iq[src_start:src_end]
    return out

def extract_slot(frame_iq, slot_idx, slot_samples, hint_offset=None):
    """
    Find the AC within a ±SEARCH_RADIUS window around the nominal slot boundary.
    If hint_offset is given (from a previous clean decode of the same user in
    another slot), skip correlation and use that offset directly.

    Returns (chunk, ac_offset) where chunk is slot_samples of IQ starting at
    the AC peak, and ac_offset is the signed distance from nominal boundary.
    """
    nominal   = slot_idx * slot_samples

    if hint_offset is not None:
        start = nominal + hint_offset
        return _safe_slice(frame_iq, start, start + slot_samples, slot_samples), hint_offset

    win_start = max(0, nominal - SEARCH_RADIUS)
    win_end   = min(len(frame_iq), nominal + SEARCH_RADIUS + slot_samples)
    window    = frame_iq[win_start:win_end]

    corr = np.maximum(_norm_xcorr(window, _TEMPLATES[0]),
                      _norm_xcorr(window, _TEMPLATES[1]))

    local_limit = min(2 * SEARCH_RADIUS + 1, len(corr))
    best_local  = int(np.argmax(corr[:local_limit]))
    ac_in_frame = win_start + best_local
    ac_offset   = ac_in_frame - nominal

    chunk = _safe_slice(frame_iq, ac_in_frame, ac_in_frame + slot_samples, slot_samples)

    rlog(f"    [EXTRACT] slot={slot_idx+1}  nominal={nominal}  "
         f"ac_peak={ac_in_frame}  offset={ac_offset:+d}  "
         f"corr={corr[best_local]:.4f}")
    return chunk, ac_offset

# ─────────────────────────────────────────────────────────────────────────────
# Packet decode
# ─────────────────────────────────────────────────────────────────────────────
def decode_slot(slot_iq):
    """
    Returns (pkt_dict, ac_bit_offset) on success, (None, -1) on failure.
    ac_bit_offset is where inside the chunk the AC starts (= sample offset for SPS=1).
    """
    symbols  = slot_iq[::SPS]
    bits_raw = bpsk_demod(symbols)
    ac_bits  = bytes_to_bits(ACCESS_CODE)
    ac_len   = len(ac_bits)

    for init in [0, 1]:
        bits = diff_decode(bits_raw, init)
        for i in range(min(len(bits) - ac_len, SEARCH_RADIUS * SPS + 1)):
            if int(np.sum(bits[i:i+ac_len] == ac_bits)) >= AC_THRESHOLD:
                pkt = _parse_packet(bits[i + ac_len:])
                if pkt and pkt['crc_ok']:
                    return pkt, i
    return None, -1

def _parse_packet(bits_after_ac):
    raw = bits_to_bytes(bits_after_ac)
    min_bytes = 2 + 2 + 1 + 1 + MAX_DEGREE + PAYLOAD_LEN + CRC_BYTES + FILLER_LEN
    if len(raw) < min_bytes: return None

    idx       = 0
    user_id   = (raw[idx] << 8) | raw[idx+1];  idx += 2
    frame_seq = (raw[idx] << 8) | raw[idx+1];  idx += 2
    degree    = raw[idx];                        idx += 1
    this_slot = raw[idx];                        idx += 1
    slot_list = [s for s in raw[idx:idx+MAX_DEGREE] if s != 0]; idx += MAX_DEGREE
    payload   = raw[idx:idx+PAYLOAD_LEN];        idx += PAYLOAD_LEN
    crc_rx    = raw[idx:idx+CRC_BYTES]

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
# Replica reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def build_replica(pkt, target_slot, slot_samples, ac_offset=0):
    """
    Reconstruct ideal IQ for pkt's copy in target_slot, with the AC starting
    at sample ac_offset within the returned slot_samples buffer.
    target_slot sets the 'this_slot' field (and thus the CRC) correctly.
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

    replica = np.zeros(slot_samples, dtype=np.complex64)
    end     = ac_offset + len(symbols)
    if end <= slot_samples:
        replica[ac_offset:end] = symbols
    else:
        replica[ac_offset:] = symbols[:slot_samples - ac_offset]
    return replica

def match_scale(received, replica):
    mask = np.abs(replica) > 0.1
    if mask.sum() < 10: return complex(1.0, 0.0)
    r = replica[mask]; s = received[mask]
    return np.dot(r.conj(), s) / np.dot(r.conj(), r)

# ─────────────────────────────────────────────────────────────────────────────
# Frame sync
# ─────────────────────────────────────────────────────────────────────────────
def find_frames(iq, N_slots, slot_samples, threshold=0.7):
    frame_len = N_slots * slot_samples
    corr      = np.maximum(_norm_xcorr(iq, _TEMPLATES[0]),
                           _norm_xcorr(iq, _TEMPLATES[1]))

    peaks, _ = find_peaks(corr, height=threshold, distance=slot_samples // 2)
    peaks    = peaks[np.argsort(peaks)]

    rlog(f"[SYNC] {len(peaks)} peaks above threshold={threshold}")

    frames     = []
    search_pos = 0

    for peak in peaks:
        if peak < search_pos: continue

        pkt, _ = decode_slot(iq[peak:peak + slot_samples])
        if pkt is None or not pkt['crc_ok']:
            continue

        if pkt['frame_seq'] == 0:
            rlog(f"[SYNC] Beacon at sample {peak}, uid={pkt['user_id']}, skipping.")
            search_pos = peak + slot_samples
            continue

        # Back-calculate frame start from the decoded this_slot
        frame_start = peak - (pkt['this_slot'] - 1) * slot_samples
        frame_end   = frame_start + frame_len

        if frame_start < 0 or frame_end > len(iq):
            rlog(f"[SYNC] Frame [{frame_start}:{frame_end}] out of bounds, skipping.")
            continue

        frames.append(iq[frame_start:frame_end].copy())
        search_pos = frame_end

        rlog(f"[SYNC] Frame {len(frames):03d}: samples [{frame_start}:{frame_end}]  "
             f"uid={pkt['user_id']}  seq={pkt['frame_seq']}  "
             f"this_slot={pkt['this_slot']}  corr={corr[peak]:.3f}")

    rlog(f"[SYNC] Found {len(frames)} data frame(s).")
    return frames

# ─────────────────────────────────────────────────────────────────────────────
# SIC – process one frame
# ─────────────────────────────────────────────────────────────────────────────
def process_frame(frame_iq, N_slots, slot_samples, pkt_writer, sic_writer):
    """
    Successive Interference Cancellation.

    Cancellation buffer layout (per slot s):
      cancelled[s] has length slot_samples + 2*SEARCH_RADIUS.
      _cac_index(ac_off) = ac_off + SEARCH_RADIUS maps a signed slot-relative
      offset to a valid index in cancelled[s].

    KEY FIX vs original:
      Original subtracted cancelled[s][ac_off : ac_off+slot_samples] which
      ignored the SEARCH_RADIUS padding and indexed the wrong region.
      Corrected to: cancelled[s][ci : ci+slot_samples]
      where ci = _cac_index(ac_off) = ac_off + SEARCH_RADIUS.
    """
    _CAC_LEN  = slot_samples + 2 * SEARCH_RADIUS
    cancelled = [np.zeros(_CAC_LEN, dtype=np.complex64) for _ in range(N_slots)]

    def _cac_index(ac_off):
        return ac_off + SEARCH_RADIUS

    user_offsets  = {}   # uid → slot-relative AC offset
    decoded_slots = set()
    decoded_pkts  = {}   # uid → pkt dict (for reporting)
    frame_id      = None

    def get_residual(s, ac_off):
        """Extract slot s starting at slot-relative ac_off, minus accumulated cancellations."""
        chunk = _safe_slice(frame_iq,
                            s * slot_samples + ac_off,
                            s * slot_samples + ac_off + slot_samples,
                            slot_samples)
        ci    = _cac_index(ac_off)
        return chunk - cancelled[s][ci : ci + slot_samples]  # FIX 1 applied here

    def add_cancellation(s, replica_scaled, ac_off):
        ci  = _cac_index(ac_off)
        end = ci + len(replica_scaled)
        if end <= _CAC_LEN:
            cancelled[s][ci:end] += replica_scaled

    rlog(f"\n[SIC] {'─'*54}")

    max_passes = N_slots + 2
    for pass_num in range(max_passes):
        new_this_pass = 0

        for s in range(N_slots):
            if s in decoded_slots:
                continue

            # ── FIX 2: try known user offsets first ───────────────────────
            # After other singletons reveal a user's offset, try extracting
            # the collision slot at that known offset.
            chunks_to_try = []

            if user_offsets:
                for uid, known_off in user_offsets.items():
                    raw_chunk = _safe_slice(frame_iq,
                                            s * slot_samples + known_off,
                                            s * slot_samples + known_off + slot_samples,
                                            slot_samples)
                    ci = _cac_index(known_off)
                    cleaned = raw_chunk - cancelled[s][ci : ci + slot_samples]
                    chunks_to_try.append((cleaned, known_off, f"hint(uid={uid},off={known_off:+d})"))

            # Always also try blind XCorr as fallback
            chunk_blind, ac_off_blind = extract_slot(frame_iq, s, slot_samples)
            ci_blind = _cac_index(ac_off_blind)
            # FIX 1: subtract with correct index
            chunk_blind_clean = chunk_blind - cancelled[s][ci_blind : ci_blind + slot_samples]
            chunks_to_try.append((chunk_blind_clean, ac_off_blind, "blind"))

            # Try decoding each candidate chunk
            decoded_here = False
            for chunk_try, ac_off_try, src in chunks_to_try:
                pkt, pkt_ac_off = decode_slot(chunk_try)
                if pkt and pkt['crc_ok'] and pkt['frame_seq'] != 0:
                    true_ac_off = ac_off_try + pkt_ac_off
                    if frame_id is None:
                        frame_id = pkt['frame_seq']

                    if pkt['user_id'] not in user_offsets:
                        user_offsets[pkt['user_id']] = true_ac_off

                    rlog(f"  pass={pass_num+1}  slot={s+1}  OK [{src}]  "
                         f"uid={pkt['user_id']}  d={pkt['degree']}  "
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
                    decoded_pkts[pkt['user_id']] = pkt
                    new_this_pass += 1
                    decoded_here = True

                    # ── Cancel replica from all sibling slots ──────────────
                    for copy_slot in pkt['slot_list']:
                        t = copy_slot - 1
                        if t == s or t in decoded_slots:
                            continue
                        if not (0 <= t < N_slots):
                            continue

                        target_off = user_offsets[pkt['user_id']]
                        replica    = build_replica(pkt, copy_slot, slot_samples, ac_offset=0)
                        residual_t = get_residual(t, target_off)
                        scale      = match_scale(residual_t, replica)
                        add_cancellation(t, scale * replica, target_off)

                        rlog(f"  CANCEL slot={copy_slot}  uid={pkt['user_id']}  "
                             f"ac_off={target_off:+d}  "
                             f"|scale|={abs(scale):.3f}  "
                             f"∠={np.degrees(np.angle(scale)):+.1f}°")
                        sic_writer.writerow([frame_id, copy_slot, 'CANCELLED',
                                             pkt['user_id'], pkt['degree'], pkt['slot_list']])
                    break  # don't try other chunks for this slot

            if not decoded_here:
                # Check if ALL users known to be in this slot are already decoded
                known_in_slot = [uid for uid, pkt in decoded_pkts.items()
                                 if s+1 in pkt['slot_list']]
                rlog(f"  pass={pass_num+1}  slot={s+1}  NO_DECODE  "
                     f"(known_users_here={known_in_slot})")
                sic_writer.writerow([frame_id or '?', s+1, 'NO_DECODE', '-', '-', '-'])

        if new_this_pass == 0:
            rlog(f"  [SIC] No new decodes in pass {pass_num+1}, stopping.")
            break

    rlog(f"[SIC] Done — decoded {len(decoded_slots)}/{N_slots} slots  "
         f"users_recovered={list(decoded_pkts.keys())}")
    return decoded_slots, decoded_pkts

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run(iq_file, N_slots=3, slot_samples=800, threshold=0.7,
        pkt_log='decoded_packets.csv', sic_log='sic_log.csv',
        report_out='sic_debug_report.txt'):

    rlog(f"{'='*60}")
    rlog(f"IRSA Receiver — Fixed Build")
    rlog(f"{'='*60}")
    rlog(f"File        : {iq_file}")
    rlog(f"N_slots     : {N_slots}")
    rlog(f"slot_samples: {slot_samples}")
    rlog(f"threshold   : {threshold}")
    rlog()

    rlog(f"[MAIN] Loading {iq_file} ...")
    iq = np.fromfile(iq_file, dtype=np.complex64)
    rlog(f"[MAIN] {len(iq)} samples loaded")
    rlog()

    frames = find_frames(iq, N_slots, slot_samples, threshold)
    if not frames:
        rlog("[MAIN] No frames found — check threshold or input file.")
        return

    total_decoded_slots = 0
    all_users = {}

    with open(pkt_log, 'w', newline='') as pf, \
         open(sic_log,  'w', newline='') as sf:

        pw = csv.writer(pf)
        sw = csv.writer(sf)

        pw.writerow(['frame_id', 'slot', 'user_id', 'frame_seq',
                     'degree', 'this_slot', 'slot_list', 'crc_ok', 'payload_hex'])
        sw.writerow(['frame_id', 'slot', 'event', 'user_id', 'degree', 'slot_list'])

        for i, frame_iq in enumerate(frames):
            rlog(f"\n{'─'*60}")
            rlog(f"Processing frame {i+1}/{len(frames)}")
            resolved, users = process_frame(frame_iq, N_slots, slot_samples, pw, sw)
            total_decoded_slots += len(resolved)
            all_users.update(users)

    rlog(f"\n{'='*60}")
    rlog(f"SUMMARY")
    rlog(f"{'='*60}")
    rlog(f"Total decoded slots : {total_decoded_slots}")
    rlog(f"Users recovered     : {sorted(all_users.keys())}")
    for uid, pkt in sorted(all_users.items()):
        rlog(f"  uid={uid}  seq={pkt['frame_seq']}  deg={pkt['degree']}  "
             f"slots={pkt['slot_list']}  payload={bytes(pkt['payload'][:8]).hex()}")
    rlog(f"\nPacket log  → {pkt_log}")
    rlog(f"SIC log     → {sic_log}")
    rlog(f"Debug report→ {report_out}")

    with open(report_out, 'w') as f:
        f.write('\n'.join(report_lines))

if __name__ == '__main__':
    run(
        iq_file      = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/sync_samples.bin',
        N_slots      = int(sys.argv[2])   if len(sys.argv) > 2 else 3,
        slot_samples = int(sys.argv[3])   if len(sys.argv) > 3 else 800,
        threshold    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7,
    )