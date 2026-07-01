"""
irsa_sic_demo.py
─────────────────
Pure Python IRSA SIC demonstration — no GNU Radio required.

Frame layout:
  Slot 0 : User 1 only  (singleton — used for channel estimation)
  Slot 1 : User 1 + User 2  (collision — SIC target)

Steps:
  1. Build packets for both users using the same byte layout as IRSA_Packet_Generator
  2. BPSK modulate (differential, no RRC — symbol-domain)
  3. Apply per-user complex channel (amplitude + phase)
  4. Add AWGN
  5. Decode slot 0 → CRC check → estimate channel h1 via LS
  6. Reconstruct User 1 replica → scale by h1 → subtract from slot 1
  7. Decode residual → User 2 packet

Usage:
  python irsa_sic_demo.py [snr_db]   # default SNR=20 dB
"""

import numpy as np
import binascii
import random
import sys

# ──────────────────────────────────────────────────────────────────────────────
#  Constants — match TX chain
# ──────────────────────────────────────────────────────────────────────────────
ACCESS_CODE_BYTES = [0xE1, 0x5A, 0xE8, 0x93]
FILLER_BYTES      = [0xDE, 0xAD, 0xBE, 0xEE, 0xDE, 0xAD, 0xBE, 0xEE]
MAX_DEGREE        = 16
CRC_BYTES         = 4
FILLER_LEN        = 8
PACKET_SIZE       = 100
HEADER_FIXED      = 4 + 2 + 2 + 1 + 1 + MAX_DEGREE + CRC_BYTES + FILLER_LEN   # = 38
PAYLOAD_LEN       = PACKET_SIZE - HEADER_FIXED                                  # = 62
AC_THRESHOLD      = 30     # bit matches out of 32


# ──────────────────────────────────────────────────────────────────────────────
#  Bit / byte helpers
# ──────────────────────────────────────────────────────────────────────────────
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

def differential_encode(bits, init_state=0):
    enc  = np.empty(len(bits), dtype=np.uint8)
    prev = init_state
    for i, b in enumerate(bits):
        enc[i] = int(b) ^ prev
        prev   = enc[i]
    return enc

def differential_decode(bits, init_state=0):
    bits = np.asarray(bits, dtype=np.uint8)
    prev = np.empty(len(bits), dtype=np.uint8)
    prev[0]  = init_state
    prev[1:] = bits[:-1]
    return (bits ^ prev).astype(np.uint8)

def bpsk_modulate(bits):
    return (1.0 - 2.0 * np.asarray(bits, dtype=np.float64)).astype(np.complex128)

def bpsk_demodulate(symbols):
    return (symbols.real < 0).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
#  Packet build  (mirrors IRSA_Packet_Generator._build_copy)
# ──────────────────────────────────────────────────────────────────────────────
def build_packet(user_id, frame_seq, degree, this_slot, slot_list):
    """Return packet bytes (list[int], length == PACKET_SIZE)."""
    uid_b    = [(user_id   >> 8) & 0xFF, user_id   & 0xFF]
    seq_b    = [(frame_seq >> 8) & 0xFF, frame_seq & 0xFF]
    deg_b    = [degree & 0xFF]
    slot_b   = [this_slot & 0xFF]
    sl_field = [s & 0xFF for s in slot_list]
    sl_field += [0x00] * (MAX_DEGREE - len(sl_field))

    # Deterministic payload — seeded by user_id (same as TX)
    rng_seed = (user_id * 31) & 0xFFFF
    rng      = random.Random(rng_seed)
    payload  = [rng.randint(0, 255) for _ in range(PAYLOAD_LEN)]

    protected = uid_b + seq_b + deg_b + slot_b + sl_field + payload
    crc_val   = binascii.crc32(bytes(protected)) & 0xFFFFFFFF
    crc_b     = list(crc_val.to_bytes(4, byteorder='little'))

    pkt = ACCESS_CODE_BYTES + protected + crc_b + FILLER_BYTES
    assert len(pkt) == PACKET_SIZE, f"Packet size mismatch: {len(pkt)}"
    return pkt


def modulate_packet(pkt_bytes):
    """bytes → differential BPSK complex symbols."""
    bits = bytes_to_bits(pkt_bytes)
    enc  = differential_encode(bits, init_state=0)
    return bpsk_modulate(enc)


# ──────────────────────────────────────────────────────────────────────────────
#  Packet decode helpers
# ──────────────────────────────────────────────────────────────────────────────
def find_access_code(bits):
    ac_bits = bytes_to_bits(ACCESS_CODE_BYTES)
    ac_len  = len(ac_bits)
    for i in range(len(bits) - ac_len + 1):
        if int(np.sum(bits[i:i+ac_len] == ac_bits)) >= AC_THRESHOLD:
            return i + ac_len   # index just after AC
    return -1


def parse_packet(bits):
    raw = bits_to_bytes(bits)
    min_len = 2+2+1+1+MAX_DEGREE + PAYLOAD_LEN + CRC_BYTES
    if len(raw) < min_len:
        return None

    idx       = 0
    user_id   = (raw[idx] << 8) | raw[idx+1]; idx += 2
    frame_seq = (raw[idx] << 8) | raw[idx+1]; idx += 2
    degree    = raw[idx];                       idx += 1
    this_slot = raw[idx];                       idx += 1
    slot_list = [s for s in raw[idx:idx+MAX_DEGREE] if s != 0]; idx += MAX_DEGREE
    payload   = raw[idx:idx+PAYLOAD_LEN];       idx += PAYLOAD_LEN
    crc_rx    = raw[idx:idx+CRC_BYTES]

    protected    = raw[:2+2+1+1+MAX_DEGREE+PAYLOAD_LEN]
    crc_expected = binascii.crc32(bytes(protected)) & 0xFFFFFFFF
    crc_actual   = int.from_bytes(bytes(crc_rx), byteorder='little')

    return {
        'user_id'  : user_id,
        'frame_seq': frame_seq,
        'degree'   : degree,
        'this_slot': this_slot,
        'slot_list': slot_list,
        'payload'  : payload,
        'crc_ok'   : (crc_expected == crc_actual),
    }


def decode_slot(slot_iq):
    """Attempt differential BPSK decode of a slot. Returns (pkt | None, ac_offset)."""
    symbols  = slot_iq.copy()                # already 1 SPS
    bits_raw = bpsk_demodulate(symbols)

    for init_state in [0, 1]:
        bits   = differential_decode(bits_raw, init_state)
        ac_end = find_access_code(bits)
        if ac_end < 0:
            continue
        pkt = parse_packet(bits[ac_end:])
        if pkt and pkt['crc_ok']:
            ac_start = ac_end - len(bytes_to_bits(ACCESS_CODE_BYTES))
            return pkt, ac_start
    return None, -1


# ──────────────────────────────────────────────────────────────────────────────
#  Channel estimation
# ──────────────────────────────────────────────────────────────────────────────
def estimate_channel(received, replica):
    """
    Least-squares channel estimate from a clean singleton slot.
    h = (replica^H · received) / (replica^H · replica)
    """
    mask = np.abs(replica) > 0.1
    if np.sum(mask) < 10:
        return complex(1.0, 0.0)
    r = replica[mask]
    s = received[mask]
    return np.dot(np.conj(r), s) / np.dot(np.conj(r), r)


# ──────────────────────────────────────────────────────────────────────────────
#  Replica reconstruction
# ──────────────────────────────────────────────────────────────────────────────
def reconstruct_replica(pkt, for_slot, slot_len, ac_offset=0):
    """
    Build ideal IQ replica for `pkt` transmitted in `for_slot`,
    zero-padded to slot_len at ac_offset.
    """
    pkt_bytes = build_packet(
        pkt['user_id'], pkt['frame_seq'],
        pkt['degree'],  for_slot, pkt['slot_list'],
    )
    symbols = modulate_packet(pkt_bytes)

    out  = np.zeros(slot_len, dtype=np.complex128)
    end  = ac_offset + len(symbols)
    clip = min(end, slot_len) - ac_offset
    out[ac_offset:ac_offset + clip] = symbols[:clip]
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  AWGN
# ──────────────────────────────────────────────────────────────────────────────
def add_awgn(signal, snr_db):
    snr_lin    = 10 ** (snr_db / 10.0)
    sig_power  = np.mean(np.abs(signal) ** 2)
    noise_std  = np.sqrt(sig_power / (2.0 * snr_lin))
    noise      = noise_std * (np.random.randn(*signal.shape) +
                              1j * np.random.randn(*signal.shape))
    return signal + noise


# ──────────────────────────────────────────────────────────────────────────────
#  Main SIC demo
# ──────────────────────────────────────────────────────────────────────────────
def run_sic_demo(snr_db=20.0, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    FRAME_SEQ  = 1
    SLOT_LEN   = PACKET_SIZE * 8    # 800 symbols — one packet fits exactly

    print("=" * 60)
    print(f"  IRSA SIC Demo  |  SNR = {snr_db:.1f} dB")
    print("=" * 60)
    print(f"  Packet size    : {PACKET_SIZE} bytes  ({PACKET_SIZE*8} symbols)")
    print(f"  Slot length    : {SLOT_LEN} symbols")
    print(f"  Frame layout   : Slot 0 = U1 only | Slot 1 = U1+U2")
    print()

    # ── Step 1: Build packets ─────────────────────────────────────────────
    # User 1: degree=2, appears in slots 1 and 2 (1-indexed → 0 and 1 in array)
    # User 2: degree=1, appears in slot 2 only
    pkt_u1_s0 = build_packet(user_id=1, frame_seq=FRAME_SEQ,
                              degree=2, this_slot=1, slot_list=[1, 2])
    pkt_u1_s1 = build_packet(user_id=1, frame_seq=FRAME_SEQ,
                              degree=2, this_slot=2, slot_list=[1, 2])
    pkt_u2_s1 = build_packet(user_id=2, frame_seq=FRAME_SEQ,
                              degree=1, this_slot=2, slot_list=[2])

    sym_u1_s0 = modulate_packet(pkt_u1_s0)
    sym_u1_s1 = modulate_packet(pkt_u1_s1)
    sym_u2_s1 = modulate_packet(pkt_u2_s1)

    # ── Step 2: Apply per-user complex channels ───────────────────────────
    # Ideal channel — unit amplitude, zero phase
    # Channel estimation error is zero; any SIC failure is a logic bug
    h1 = complex(1.0, 0.0)
    h2 = complex(1.0, 0.0)

    # Pad symbols to slot_len
    def pad(sym, length):
        out = np.zeros(length, dtype=np.complex128)
        n   = min(len(sym), length)
        out[:n] = sym[:n]
        return out

    tx_s0 = h1 * pad(sym_u1_s0, SLOT_LEN)
    tx_s1 = h1 * pad(sym_u1_s1, SLOT_LEN) + h2 * pad(sym_u2_s1, SLOT_LEN)

    # ── Step 3: Add AWGN ──────────────────────────────────────────────────
    rx_s0 = add_awgn(tx_s0, snr_db)
    rx_s1 = add_awgn(tx_s1, snr_db)

    print("─" * 60)
    print("  STEP A: Decode singleton slot 0 (User 1 only)")
    print("─" * 60)

    pkt_decoded, ac_off = decode_slot(rx_s0)

    if pkt_decoded is None or not pkt_decoded['crc_ok']:
        print("  ✗  Slot 0 decode FAILED — cannot proceed with SIC.")
        return

    print(f"user_id={pkt_decoded['user_id']}  "
          f"frame_seq={pkt_decoded['frame_seq']}  "
          f"degree={pkt_decoded['degree']}  "
          f"slot_list={pkt_decoded['slot_list']}  "
          f"CRC OK  ac_offset={ac_off}")

    # ── Step 4: Channel estimate from singleton ───────────────────────────
    print()
    print("─" * 60)
    print("  STEP B: Channel estimation from slot 0")
    print("─" * 60)

    replica_s0   = reconstruct_replica(pkt_decoded, for_slot=1,
                                       slot_len=SLOT_LEN, ac_offset=ac_off)
    h1_estimated = estimate_channel(rx_s0, replica_s0)

    print(f"  True  h1 : {h1:.4f}  "
          f"(|h|={abs(h1):.4f}, phase={np.angle(h1, deg=True):.2f}°)")
    print(f"  Estim h1 : {h1_estimated:.4f}  "
          f"(|h|={abs(h1_estimated):.4f}, phase={np.angle(h1_estimated, deg=True):.2f}°)")
    print(f"  Error    : {abs(h1_estimated - h1):.6f}")

    # ── Step 5: Reconstruct and subtract User 1 from slot 1 ──────────────
    print()
    print("─" * 60)
    print("  STEP C: Reconstruct User 1 replica for slot 1 and subtract")
    print("─" * 60)

    replica_s1       = reconstruct_replica(pkt_decoded, for_slot=2,
                                           slot_len=SLOT_LEN, ac_offset=ac_off)
    replica_s1_scaled = h1_estimated * replica_s1
    residual          = rx_s1 - replica_s1_scaled

    power_before = np.mean(np.abs(rx_s1) ** 2)
    power_after  = np.mean(np.abs(residual) ** 2)
    print(f"  Slot 1 power before SIC : {power_before:.4f}")
    print(f"  Slot 1 power after  SIC : {power_after:.4f}")
    print(f"  Cancellation            : {10*np.log10(power_before/power_after):.2f} dB")

    # ── Step 6: Decode User 2 from residual ──────────────────────────────
    print()
    print("─" * 60)
    print("  STEP D: Decode User 2 from residual")
    print("─" * 60)

    pkt_u2_decoded, ac_off2 = decode_slot(residual)

    if pkt_u2_decoded is None or not pkt_u2_decoded['crc_ok']:
        print("    User 2 decode FAILED after SIC.")
        print("     (Try increasing SNR or check replica alignment)")
    else:
        print(f"user_id={pkt_u2_decoded['user_id']}  "
              f"frame_seq={pkt_u2_decoded['frame_seq']}  "
              f"degree={pkt_u2_decoded['degree']}  "
              f"slot_list={pkt_u2_decoded['slot_list']}  "
              f"CRC OK  ac_offset={ac_off2}")

        # Verify payload matches expected
        rng_expected = random.Random((2 * 31) & 0xFFFF)
        expected_payload = [rng_expected.randint(0, 255) for _ in range(PAYLOAD_LEN)]
        payload_ok = (list(pkt_u2_decoded['payload']) == expected_payload)
        print(f"  Payload match : {'PASS' if payload_ok else 'FAIL'}")

    print()
    print("─" * 60)
    print("  SUMMARY")
    print("─" * 60)
    u1_ok = pkt_decoded is not None and pkt_decoded['crc_ok']
    u2_ok = pkt_u2_decoded is not None and pkt_u2_decoded['crc_ok'] if 'pkt_u2_decoded' in dir() else False
    print(f"  User 1 (singleton slot 0) : {' decoded' if u1_ok else ' failed'}")
    print(f"  User 2 (after SIC)        : {' decoded' if u2_ok else ' failed'}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
#  Monte Carlo sweep
# ──────────────────────────────────────────────────────────────────────────────
def monte_carlo(snr_range=None, trials=200):
    if snr_range is None:
        snr_range = np.arange(0, 21, 2)

    print()
    print("=" * 60)
    print("  Monte Carlo BER sweep")
    print("=" * 60)
    print(f"  {'SNR (dB)':<12} {'U1 success':<16} {'U2 success':<16} {'Both OK'}")
    print("  " + "-" * 55)

    for snr in snr_range:
        u1_ok = u2_ok = both_ok = 0

        for trial in range(trials):
            np.random.seed(trial)
            random.seed(trial)

            FRAME_SEQ = 1
            SLOT_LEN  = PACKET_SIZE * 8

            pkt_u1_s0 = build_packet(1, FRAME_SEQ, 2, 1, [1, 2])
            pkt_u1_s1 = build_packet(1, FRAME_SEQ, 2, 2, [1, 2])
            pkt_u2_s1 = build_packet(2, FRAME_SEQ, 1, 2, [2])

            sym_u1_s0 = modulate_packet(pkt_u1_s0)
            sym_u1_s1 = modulate_packet(pkt_u1_s1)
            sym_u2_s1 = modulate_packet(pkt_u2_s1)

            h1 = complex(1.0, 0.0)
            h2 = complex(1.0, 0.0)

            def pad(sym, n):
                out = np.zeros(n, dtype=np.complex128)
                out[:min(len(sym), n)] = sym[:min(len(sym), n)]
                return out

            rx_s0 = add_awgn(h1 * pad(sym_u1_s0, SLOT_LEN), snr)
            rx_s1 = add_awgn(h1 * pad(sym_u1_s1, SLOT_LEN) +
                              h2 * pad(sym_u2_s1, SLOT_LEN), snr)

            # Decode singleton
            pkt1, ac1 = decode_slot(rx_s0)
            if pkt1 is None or not pkt1['crc_ok']:
                continue
            u1_ok += 1

            # Channel estimate + SIC
            rep_s0    = reconstruct_replica(pkt1, for_slot=1, slot_len=SLOT_LEN, ac_offset=ac1)
            h1_est    = estimate_channel(rx_s0, rep_s0)
            rep_s1    = reconstruct_replica(pkt1, for_slot=2, slot_len=SLOT_LEN, ac_offset=ac1)
            residual  = rx_s1 - h1_est * rep_s1

            pkt2, _   = decode_slot(residual)
            if pkt2 is not None and pkt2['crc_ok']:
                u2_ok  += 1
                both_ok += 1

        print(f"  {snr:<12.1f} "
              f"{u1_ok/trials*100:<16.1f} "
              f"{u2_ok/trials*100:<16.1f} "
              f"{both_ok/trials*100:.1f}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    snr = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    run_sic_demo(snr_db=snr)
    monte_carlo()