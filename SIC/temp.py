# initial version – 30-05-2026  
 
""" 
irsa_offline.py 
─────────────── 
Combined offline IRSA processor: 
1. Frame Sync — sliding window IQ correlation to find frame boundaries 
2. SIC — successive interference cancellation per frame 
 
 
Input : costas_out.bin (complex64 raw IQ from GNU Radio File Sink) 
Output : decoded_packets.csv 
sic_log.csv 
 
 
Usage 
───── 
python irsa_offline.py costas_out.bin [N_slots] [slot_samples] 
 
 
# defaults: N_slots=2, slot_samples=8000 
 
 
GNU Radio File Sink settings 
───────────────────────────── 
Format : Complex (float32 IQ interleaved) 
Place after : Costas Loop output 
""" 
 
 
import numpy as np 
import csv 
import binascii 
import sys 
import glob 
import re 
 
 
 
# ───────────────────────────────────────────────────────────────────────────── 
# Constants — must match TX chain 
# ───────────────────────────────────────────────────────────────────────────── 
 
 
ACCESS_CODE_BYTES = [0xE1, 0x5A, 0xE8, 0x93] 
FILLER_BYTES = [0xDE, 0xAD, 0xBE, 0xEE, 0xDE, 0xAD, 0xBE, 0xEE] 
AC_THRESHOLD = 30 # bit matches out of 32 
MAX_DEGREE = 16 
CRC_BYTES = 4 
FILLER_LEN = 8 
PACKET_SIZE = 100 
# AC(4)+uid(2)+seq(2)+deg(1)+slot(1)+slot_list(16)+CRC(4)+filler(8) = 38 
HEADER_FIXED = 4 + 2 + 2 + 1 + 1 + MAX_DEGREE + CRC_BYTES + FILLER_LEN 
PAYLOAD_LEN = PACKET_SIZE - HEADER_FIXED # = 62 
 
 
SPS = 1 
RRC_ALPHA = 0.350 
RRC_NUM_TAPS = 110 
DIFF_INIT_STATE = 0 # silence (0x00) precedes each packet from PDU_to_Timed_Byte_Stream 
 
 
 
# ───────────────────────────────────────────────────────────────────────────── 
# DSP helpers 
# ───────────────────────────────────────────────────────────────────────────── 
 
 
def make_rrc_taps(sps=SPS, alpha=RRC_ALPHA, num_taps=RRC_NUM_TAPS): 
    taps = np.zeros(num_taps) 
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
    n = (len(bits) // 8) * 8 
    return [int(''.join(map(str, bits[i:i+8])), 2) for i in range(0, n, 8)] 
 
 
 
def differential_encode(bits, init_state=DIFF_INIT_STATE): 
    enc = np.empty(len(bits), dtype=np.uint8) 
    prev = init_state 
    for i, b in enumerate(bits): 
        enc[i] = int(b) ^ prev 
        prev = enc[i] 
    return enc 
 
 
 
def differential_decode(bits, init_state=DIFF_INIT_STATE): 
    bits = np.asarray(bits, dtype=np.uint8) 
    prev = np.empty(len(bits), dtype=np.uint8) 
    prev[0] = init_state 
    prev[1:] = bits[:-1] 
    return (bits ^ prev).astype(np.uint8) 
 
 
 
def bpsk_modulate(bits): 
    return (1.0 - 2.0 * np.asarray(bits, dtype=np.float32)).astype(np.complex64) 
 
 
 
def bpsk_demodulate(symbols): 
    return (symbols.real < 0).astype(np.uint8) 
 
 
 
# ───────────────────────────────────────────────────────────────────────────── 
# Frame Sync — build template and correlate 
# ───────────────────────────────────────────────────────────────────────────── 
 
 
def build_template(init_state=DIFF_INIT_STATE): 
    bits = bytes_to_bits(ACCESS_CODE_BYTES) 
    enc = differential_encode(bits, init_state) 
    symbols = bpsk_modulate(enc) # 32 complex symbols, 1 per bit 
    symbols /= np.max(np.abs(symbols)) 
    return symbols 
 
 
 
def normalised_xcorr(signal, template): 
    """Normalised cross-correlation magnitude, output in [0,1].""" 
    corr = np.correlate(signal, template, mode='valid') 
    mag = np.abs(corr) 
    tmpl_len = len(template) 
    sig_energy = np.convolve(np.abs(signal) ** 2, 
                             np.ones(tmpl_len), mode='valid') 
    sig_energy = np.sqrt(np.maximum(sig_energy, 1e-12)) 
    tmpl_energy = np.sqrt(np.sum(np.abs(template) ** 2)) 
    return mag / (sig_energy * tmpl_energy + 1e-12) 
 
 
 
# def find_frames(iq, N_slots, slot_samples, threshold=0.7, 
#                 sps=SPS, alpha=RRC_ALPHA, num_taps=RRC_NUM_TAPS): 
#     """ 
#     Scan raw Costas-output IQ and extract all frames. 
#
#     Returns list of complex64 arrays, each of length N_slots * slot_samples. 
#     """ 
#     # template = build_template(sps, alpha, num_taps) 
#     template = build_template() 
#     tmpl_len = len(template) 
#     frame_len = N_slots * slot_samples 
#     min_gap = frame_len // 2 
#
#     print(f"[SYNC] Template length : {tmpl_len} samples") 
#     print(f"[SYNC] Frame length : {frame_len} samples") 
#     print(f"[SYNC] Total IQ samples: {len(iq)}") 
#     print(f"[SYNC] Correlating ...") 
#
#     norm_corr = normalised_xcorr(iq, template) 
#     frames = [] 
#     last_detect = -min_gap 
#     search_pos = 0 
#
#     while search_pos < len(norm_corr): 
#         window = norm_corr[search_pos:] 
#         local_peak = int(np.argmax(window)) 
#         peak_idx = search_pos + local_peak 
#         peak_val = norm_corr[peak_idx] 
#
#         if peak_val < threshold: 
#             break 
#
#         if peak_idx <= last_detect + min_gap: 
#             search_pos = peak_idx + 1 
#             continue 
#
#         frame_start = peak_idx # + tmpl_len 
#         frame_end = frame_start + frame_len 
#
#         if frame_end > len(iq): 
#             print(f"[SYNC] Frame {len(frames):04d}: detected at {peak_idx} " 
#                   f"but not enough samples — stopping.") 
#             break 
#
#         frame_iq = iq[frame_start:frame_end].copy() 
#         frames.append(frame_iq) 
#         last_detect = peak_idx 
#         search_pos = frame_end # advance past this frame 
#
#         print(f"[SYNC] Frame {len(frames)-1:04d}: " 
#               f"peak_idx={peak_idx} peak_val={peak_val:.3f} " 
#               f"samples [{frame_start}:{frame_end}]") 
#
#     print(f"[SYNC] Found {len(frames)} frame(s).") 
#     return frames 
 
 
def find_frames(iq, N_slots, slot_samples, threshold=0.7): 
    template = build_template() 
    tmpl_len = len(template) 
    frame_len = N_slots * slot_samples 
    min_gap = frame_len // 2 
 
 
    print(f"[SYNC] Template length : {tmpl_len} samples") 
    print(f"[SYNC] Frame length : {frame_len} samples") 
    print(f"[SYNC] Total IQ samples: {len(iq)}") 
    print(f"[SYNC] Correlating ...") 
 
 
    norm_corr = normalised_xcorr(iq, template) 
    frames = [] 
    last_detect = -min_gap 
    search_pos = 0 
 
 
    while search_pos < len(norm_corr): 
        window = norm_corr[search_pos:] 
        local_peak = int(np.argmax(window)) 
        peak_idx = search_pos + local_peak 
        peak_val = norm_corr[peak_idx] 
 
 
        if peak_val < threshold: 
            break 
 
 
        if peak_idx <= last_detect + min_gap: 
            search_pos = peak_idx + 1 
            continue 
 
 
        # ── Generalized slot-offset search ──────────────────────────────── 
        # The AC can belong to any slot. Try each possible frame start 
        # (peak_idx - slot_offset * slot_samples) and pick the first one 
        # where at least one slot decodes with a valid CRC. 
        best_frame = None 
        for slot_offset in range(N_slots): 
            frame_start = peak_idx - slot_offset * slot_samples 
            if frame_start < 0: 
                continue 
            frame_end = frame_start + frame_len 
            if frame_end > len(iq): 
                continue 
 
 
            candidate = iq[frame_start:frame_end].copy() 
 
 
            # Check if any slot in this candidate frame decodes cleanly 
            for s in range(N_slots): 
                slot_iq = candidate[s * slot_samples:(s+1) * slot_samples] 
                pkt = decode_slot(slot_iq) 
                if pkt is not None and pkt['crc_ok']: 
                    best_frame = (frame_start, frame_end, candidate) 
                    break 
 
 
            if best_frame is not None: 
                break 
 
 
        if best_frame is None: 
            # No offset gave a valid decode — skip this peak 
            print(f"[SYNC] Peak at {peak_idx} (val={peak_val:.3f}): " 
                  f"no valid frame alignment found, skipping.") 
            search_pos = peak_idx + 1 
            continue 
 
 
        frame_start, frame_end, frame_iq = best_frame 
        frames.append(frame_iq) 
        last_detect = peak_idx 
        search_pos = frame_end 
 
 
        print(f"[SYNC] Frame {len(frames)-1:04d}: " 
              f"peak_idx={peak_idx} peak_val={peak_val:.3f} " 
              f"samples [{frame_start}:{frame_end}]") 
 
 
    print(f"[SYNC] Found {len(frames)} frame(s).") 
    return frames 
 
 
 
# ───────────────────────────────────────────────────────────────────────────── 
# Packet decode helpers 
# ───────────────────────────────────────────────────────────────────────────── 
 
 
def find_access_code(bits, threshold=AC_THRESHOLD): 
    ac_bits = bytes_to_bits(ACCESS_CODE_BYTES) 
    ac_len = len(ac_bits) 
    for i in range(len(bits) - ac_len + 1): 
        if int(np.sum(bits[i:i+ac_len] == ac_bits)) >= threshold: 
            return i + ac_len 
    return -1 
 
 
 
def parse_irsa_packet(bits): 
    raw = bits_to_bytes(bits) 
    min_len = 2+2+1+1+MAX_DEGREE + PAYLOAD_LEN + CRC_BYTES + FILLER_LEN 
    if len(raw) < min_len: 
        return None 
 
 
    idx = 0 
    user_id = (raw[idx] << 8) | raw[idx+1]; idx += 2 
    frame_seq = (raw[idx] << 8) | raw[idx+1]; idx += 2 
    degree = raw[idx]; idx += 1 
    this_slot = raw[idx]; idx += 1 
    slot_list = [s for s in raw[idx:idx+MAX_DEGREE] if s != 0]; idx += MAX_DEGREE 
    payload = raw[idx:idx+PAYLOAD_LEN]; idx += PAYLOAD_LEN 
    crc_rx = raw[idx:idx+CRC_BYTES] 
 
 
    protected = raw[:2+2+1+1+MAX_DEGREE+PAYLOAD_LEN] 
    crc_expected = binascii.crc32(bytes(protected)) & 0xFFFFFFFF 
    crc_actual = int.from_bytes(bytes(crc_rx), byteorder='little') 
 
 
    return { 
        'user_id' : user_id, 
        'frame_seq' : frame_seq, 
        'degree' : degree, 
        'this_slot' : this_slot, 
        'slot_list' : slot_list, 
        'payload' : payload, 
        'crc_ok' : (crc_expected == crc_actual), 
    } 
 
 
 
def decode_slot(slot_iq): 
    symbols = slot_iq[::SPS] 
    bits_raw = bpsk_demodulate(symbols) 
    bits = differential_decode(bits_raw) 
    ac_end = find_access_code(bits) 
    # print(f"[DEBUG] AC found at bit position: {ac_end}") # add this 
    if ac_end < 0: 
        return None 
    pkt = parse_irsa_packet(bits[ac_end:]) 
    if pkt and not pkt['crc_ok']: 
        raw = bits_to_bytes(bits[ac_end:]) 
        print(f"[DEBUG] raw bytes after AC: {len(raw)}") 
        print(f"[DEBUG] user_id={pkt['user_id']} frame_seq={pkt['frame_seq']}") 
        print(f"[DEBUG] degree={pkt['degree']} this_slot={pkt['this_slot']}") 
        print(f"[DEBUG] slot_list={pkt['slot_list']}") 
        # recompute CRC manually 
        protected = raw[:2+2+1+1+MAX_DEGREE+PAYLOAD_LEN] 
        crc_exp = binascii.crc32(bytes(protected)) & 0xFFFFFFFF 
        crc_rx = int.from_bytes(bytes(raw[2+2+1+1+MAX_DEGREE+PAYLOAD_LEN: 
                                          2+2+1+1+MAX_DEGREE+PAYLOAD_LEN+4]), 
                                byteorder='little') 
        print(f"[DEBUG] CRC expected=0x{crc_exp:08X} actual=0x{crc_rx:08X}") 
        print(f"[DEBUG] total bits in slot: {len(bits)}") 
    return pkt 
 
 
# def decode_slot(slot_iq): 
#     symbols = slot_iq[::SPS] 
#     bits_raw = bpsk_demodulate(symbols) 
#     for init_state in [0, 1]: 
#         bits = differential_decode(bits_raw, init_state) 
#         ac_end = find_access_code(bits) 
#         if ac_end < 0: 
#             continue 
#         pkt = parse_irsa_packet(bits[ac_end:]) 
#         if pkt and pkt['crc_ok']: 
#             return pkt 
#     return None 
 
 
 
# ───────────────────────────────────────────────────────────────────────────── 
# Replica reconstruction for SIC 
# ───────────────────────────────────────────────────────────────────────────── 
 
 
def reconstruct_replica(pkt, slot_idx, slot_samples): 
    """Rebuild IQ waveform for a decoded packet (TX RRC only — RX RRC already applied).""" 
    uid_b = [(pkt['user_id'] >> 8) & 0xFF, pkt['user_id'] & 0xFF] 
    seq_b = [(pkt['frame_seq'] >> 8) & 0xFF, pkt['frame_seq'] & 0xFF] 
    deg_b = [pkt['degree'] & 0xFF] 
    slot_b = [slot_idx & 0xFF] 
    sl_field = [s & 0xFF for s in pkt['slot_list']] 
    sl_field += [0x00] * (MAX_DEGREE - len(sl_field)) 
    payload = list(pkt['payload']) 
 
 
    protected = uid_b + seq_b + deg_b + slot_b + sl_field + payload 
    crc_val = binascii.crc32(bytes(protected)) & 0xFFFFFFFF 
    crc_b = list(crc_val.to_bytes(4, byteorder='little')) 
    pkt_bytes = ACCESS_CODE_BYTES + protected + crc_b + FILLER_BYTES 
    pkt_bits = bytes_to_bits(pkt_bytes) 
 
 
    # TX chain: diff encode → BPSK → upsample → RRC (once) 
    enc = differential_encode(pkt_bits) 
    symbols = bpsk_modulate(enc) 
    up = np.zeros(len(symbols) * SPS, dtype=np.complex64) 
    up[::SPS] = symbols 
    replica = np.convolve(up, _RRC_TAPS, mode='full').astype(np.complex64) 
 
 
    # Trim / pad to slot_samples 
    if len(replica) >= slot_samples: 
        return replica[:slot_samples] 
    out = np.zeros(slot_samples, dtype=np.complex64) 
    out[:len(replica)] = replica 
    return out 
 
 
 
# ───────────────────────────────────────────────────────────────────────────── 
# SIC processor 
# ───────────────────────────────────────────────────────────────────────────── 
 
 
def process_frame_sic(frame_iq, frame_id, N_slots, slot_samples, 
                      max_iter, pkt_writer, sic_writer): 
    """Run SIC on one frame. Writes results to open CSV writers.""" 
 
 
    residual = [ 
        frame_iq[s * slot_samples:(s+1) * slot_samples].copy() 
        for s in range(N_slots) 
    ] 
 
 
    decoded_packets = [] 
    cancelled_slots = set() 
 
 
    print(f"\n[SIC] ── Frame {frame_id:04d} {'─'*40}") 
 
 
    for sic_iter in range(max_iter): 
        newly_decoded = [] 
 
 
        for slot_idx in range(N_slots): 
            if slot_idx in cancelled_slots: 
                continue 
 
 
            pkt = decode_slot(residual[slot_idx]) 
 
 
            if pkt is None: 
                print(f" iter={sic_iter} slot={slot_idx+1:02d} NO_AC") 
                sic_writer.writerow([frame_id, sic_iter, slot_idx+1, 
                                     'NO_AC', '-', '-', '-']) 
                continue 
 
 
            if not pkt['crc_ok']: 
                print(f" iter={sic_iter} slot={slot_idx+1:02d} " 
                      f"CRC_FAIL user={pkt['user_id']}") 
                sic_writer.writerow([frame_id, sic_iter, slot_idx+1, 
                                     'CRC_FAIL', pkt['user_id'], 
                                     pkt['degree'], pkt['slot_list']]) 
                continue 
 
 
            print(f" iter={sic_iter} slot={slot_idx+1:02d} " 
                  f"DECODED user={pkt['user_id']} " 
                  f"d={pkt['degree']} copies@{pkt['slot_list']}") 
            sic_writer.writerow([frame_id, sic_iter, slot_idx+1, 
                                 'DECODED', pkt['user_id'], 
                                 pkt['degree'], pkt['slot_list']]) 
 
 
            pay_hex = ' '.join(f'{b:02X}' for b in pkt['payload']) 
            pkt_writer.writerow([ 
                frame_id, sic_iter, slot_idx+1, 
                pkt['user_id'], pkt['frame_seq'], pkt['degree'], 
                pkt['this_slot'], pkt['slot_list'], pkt['crc_ok'], pay_hex, 
            ]) 
 
 
            newly_decoded.append((slot_idx, pkt)) 
            decoded_packets.append(pkt) 
 
 
        if not newly_decoded: 
            print(f" iter={sic_iter} No new decodes — stopping.") 
            break 
 
 
        # Cancel replicas 
        for orig_idx, pkt in newly_decoded: 
            replica = reconstruct_replica(pkt, orig_idx + 1, slot_samples) 
            for cancel_slot in pkt['slot_list']: 
                s0 = cancel_slot - 1 
                if 0 <= s0 < N_slots: 
                    residual[s0] -= replica 
                    cancelled_slots.add(s0) 
                    print(f" iter={sic_iter} CANCELLED slot={cancel_slot:02d}" 
                          f" (user={pkt['user_id']})") 
                    sic_writer.writerow([frame_id, sic_iter, cancel_slot, 
                                         'CANCELLED', pkt['user_id'], 
                                         pkt['degree'], pkt['slot_list']]) 
 
 
        if len(cancelled_slots) == N_slots: 
            print(f" All slots resolved after iter={sic_iter}.") 
            break 
 
 
    crc_ok = [p for p in decoded_packets if p['crc_ok']] 
    print(f"[SIC] Frame {frame_id:04d} — decoded={len(crc_ok)}, " 
          f"resolved={len(cancelled_slots)}/{N_slots} slots") 
    return crc_ok 
 
 
 
# ───────────────────────────────────────────────────────────────────────────── 
# Main 
# ───────────────────────────────────────────────────────────────────────────── 
 
 
def run(iq_file, N_slots=2, slot_samples=16000, max_iter=10, 
        threshold=0.7, 
        pkt_log="decoded_packets.csv", 
        sic_log="sic_log.csv"): 
 
 
    # ── Load IQ ─────────────────────────────────────────────────────────── 
    print(f"[MAIN] Loading {iq_file} ...") 
    iq = np.fromfile(iq_file, dtype=np.complex64) 
    print(f"[MAIN] Loaded {len(iq)} samples ({len(iq)/40000:.2f} s @ 4 kHz)") # should be changed to 4kHz 
 
 
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
            'frame_seq', 'degree', 'this_slot', 'slot_list', 
            'crc_ok', 'payload_hex', 
        ]) 
        sic_writer.writerow([ 
            'frame_id', 'sic_iter', 'slot_index', 
            'event', 'user_id', 'degree', 'slot_list', 
        ]) 
 
 
        for i, frame_iq in enumerate(frames): 
            pkts = process_frame_sic( 
                frame_iq, i, N_slots, slot_samples, 
                max_iter, pkt_writer, sic_writer 
            ) 
            total_decoded += len(pkts) 
 
 
    print(f"\n[MAIN] Done — total decoded packets : {total_decoded}") 
    print(f"[MAIN] Packet log → {pkt_log}") 
    print(f"[MAIN] SIC log → {sic_log}") 
 
 
 
if __name__ == '__main__': 
    if len(sys.argv) < 2: 
        print("Usage: python irsa_offline.py <costas_out.bin> [N_slots] [slot_samples] [threshold]") 
        print("       python irsa_offline.py costas_out.bin 2 16000 0.7") 
        sys.exit(1) 
 
 
    iq_file = sys.argv[1] 
    N_slots = int(sys.argv[2]) if len(sys.argv) > 2 else 2 
    slot_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 16000 
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7 
 
 
    run(iq_file, N_slots=N_slots, slot_samples=slot_samples, threshold=threshold)