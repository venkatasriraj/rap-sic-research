"""
ALOHA Performance Analysis
===========================
Formulas used
-------------
1.  Mean interval (measured)   = mean of wait_time_s column from TX log
2.  Packet rate (lambda)       = 1 / mean_interval                        [pkts/s]
3.  Packet duration (T)        = TX_PACKET_BYTES * 8 / channel_bit_rate   [s]
4.  User bit rate              = TX_PACKET_BYTES * 8 * packet_rate        [bps]
                                 (full 84-byte on-air packet: AC+seq+uid+payload+CRC+filler)
5.  Channel bit rate           = sample_rate / samples_per_symbol          [bps]
                                 (40000 sps, DBPSK -> 1 bit/symbol -> 4000 bps)
6.  Offered load per user (G)  = lambda * T
7.  Total offered load (G_tot) = sum of G over all users
8.  Throughput (S)             = (correct_packets / total_tx_packets) * G_tot
                                 correct = matched AND crc_ok AND zero bit errors
9.  BER                        = total_bit_errors / total_bits_compared   (payload bytes only)
10. PER                        = (lost + errored + crc_failed) / total_tx_packets
                                 "matched" means seq found in both TX and RX for same user_id
"""

import csv, glob, re

# ── System parameters ────────────────────────────────────────────────
SAMPLE_RATE        = 40_000    # sps
SAMPLES_PER_SYMBOL = 10        # DBPSK post-Symbol-Sync output
CHANNEL_BIT_RATE   = SAMPLE_RATE / SAMPLES_PER_SYMBOL   # 4000 bps

TX_PACKET_BYTES = 84           # full on-air packet size
# TX layout: [AC(4)][seq(2)][uid(2)][payload(64)][CRC(4)][filler(8)]
PAYLOAD_START   = 8            # skip AC + seq + uid
PAYLOAD_END     = -8           # skip CRC + filler

RX_FILE      = "rx_log.csv"
TX_FILE_GLOB = "data_tx*.csv"
NUM_USERS    = None            # None = auto-detect all

# ── Loaders ───────────────────────────────────────────────────────────
def load_tx(path, user_id):
    tx, intervals = {}, []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if int(row['user_id']) != user_id:
                continue
            raw = [int(x, 16) for x in row['data_hex'].split()]
            seq = (raw[4] << 8) | raw[5]
            tx[seq] = raw[PAYLOAD_START:PAYLOAD_END]   # 64-byte payload
            intervals.append(float(row['wait_time_s']))
    return tx, intervals

def load_rx(path, user_id):
    """Returns dict: seq -> (payload_bytes, crc_ok)"""
    rx = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if int(row['user_id']) != user_id:
                continue
            seq     = int(row['seq_num'])
            payload = [int(x, 16) for x in row['payload_hex'].split()]
            crc_ok  = row['crc_ok'].strip().lower() == 'true'
            rx[seq] = (payload, crc_ok)
    return rx

# ── BER ───────────────────────────────────────────────────────────────
def bit_errors(a, b):
    return sum(bin(x ^ y).count('1') for x, y in zip(a, b))

# ── Per-user analysis ─────────────────────────────────────────────────
def analyse_user(user_id, tx, rx, intervals):
    tx_seqs = set(tx.keys())
    rx_seqs = set(rx.keys())
    matched = sorted(tx_seqs & rx_seqs)
    lost    = sorted(tx_seqs - rx_seqs)   # TX'd but never received
    extra   = sorted(rx_seqs - tx_seqs)   # received with unknown seq

    total_errors, total_bits = 0, 0
    errored_pkts  = 0   # matched but has bit errors in payload
    crc_fail_pkts = 0   # matched but CRC failed
    correct_pkts  = 0   # matched AND crc_ok AND zero bit errors

    for seq in matched:
        t_payload        = tx[seq]
        r_payload, crc_ok = rx[seq]
        n = min(len(t_payload), len(r_payload))
        e = bit_errors(t_payload[:n], r_payload[:n])
        total_errors += e
        total_bits   += n * 8
        if not crc_ok:
            crc_fail_pkts += 1
        if e > 0:
            errored_pkts += 1
        if crc_ok and e == 0:
            correct_pkts += 1

    # PER: lost + any matched packet that is not fully correct
    bad_pkts = len(lost) + (len(matched) - correct_pkts)
    per      = bad_pkts / len(tx_seqs) if tx_seqs else 0
    ber      = total_errors / total_bits if total_bits else 0

    mean_interval  = sum(intervals) / len(intervals) if intervals else 1
    packet_rate    = 1.0 / mean_interval
    packet_dur_s   = (TX_PACKET_BYTES * 8) / CHANNEL_BIT_RATE
    user_bit_rate  = (TX_PACKET_BYTES * 8) * packet_rate   # full on-air packet
    offered_load_G = packet_rate * packet_dur_s

    return dict(
        user_id=user_id,
        tx_pkts=len(tx_seqs), rx_pkts=len(rx_seqs),
        matched=len(matched), lost=len(lost), extra=len(extra),
        correct_pkts=correct_pkts,
        errored_pkts=errored_pkts, crc_fail_pkts=crc_fail_pkts,
        total_errors=total_errors, total_bits=total_bits,
        ber=ber, per=per,
        mean_interval=mean_interval, packet_rate=packet_rate,
        packet_dur_ms=packet_dur_s * 1e3,
        user_bit_rate=user_bit_rate,
        offered_load_G=offered_load_G,
        matched_seqs=matched, tx_dict=tx, rx_dict=rx,
    )

# ── Print ─────────────────────────────────────────────────────────────
def print_user_report(s):
    W = 57
    print(f"\n{'='*W}")
    print(f"  User {s['user_id']} Report")
    print(f"{'='*W}")
    print(f"\n  --- Packet counts ---")
    print(f"  TX packets              : {s['tx_pkts']}")
    print(f"  RX packets              : {s['rx_pkts']}")
    print(f"  Matched (seq found)     : {s['matched']}")
    print(f"    Correct (crc_ok+no err) : {s['correct_pkts']}")
    print(f"    Bit errors              : {s['errored_pkts']}")
    print(f"    CRC failed              : {s['crc_fail_pkts']}")
    print(f"  Lost (not received)     : {s['lost']}")
    print(f"  Extra (unknown seq)     : {s['extra']}")
    print(f"\n  --- Bit / Packet error ---")
    print(f"  Total bits compared     : {s['total_bits']}")
    print(f"  Total bit errors        : {s['total_errors']}")
    print(f"  BER                     : {s['ber']:.6f}  ({s['ber']*100:.4f}%)")
    print(f"  PER                     : {s['per']:.6f}  ({s['per']*100:.2f}%)")
    print(f"\n  --- Traffic metrics ---")
    print(f"  TX packet size          : {TX_PACKET_BYTES} bytes ({TX_PACKET_BYTES*8} bits)")
    print(f"  Mean interval           : {s['mean_interval']*1e3:.2f} ms")
    print(f"  Packet rate (λ)         : {s['packet_rate']:.3f} pkts/s")
    print(f"  Packet duration (T)     : {s['packet_dur_ms']:.3f} ms")
    print(f"  User bit rate           : {s['user_bit_rate']:.1f} bps  ({s['user_bit_rate']/1e3:.3f} kbps)")
    print(f"  Offered load G          : {s['offered_load_G']:.4f}")
    print(f"\n  --- Per-packet BER ---")
    print(f"  {'seq':>5}  {'errors':>8}  {'bits':>6}  {'BER':>10}  {'crc_ok':>7}")
    print(f"  {'-'*44}")
    # for seq in s['matched_seqs']:
    #     t            = s['tx_dict'][seq]
    #     r_payload, crc_ok = s['rx_dict'][seq]
    #     n = min(len(t), len(r_payload))
    #     e = bit_errors(t[:n], r_payload[:n])
    #     print(f"  {seq:>5}  {e:>8}  {n*8:>6}  {e/(n*8):>10.6f}  {str(crc_ok):>7}")
    # print(f"{'='*W}")

def print_summary(results):
    W = 57
    G_total       = sum(r['offered_load_G'] for r in results)
    total_tx      = sum(r['tx_pkts']        for r in results)
    total_correct = sum(r['correct_pkts']   for r in results)
    S = (total_correct / total_tx) * G_total if total_tx else 0

    print(f"\n{'='*W}")
    print(f"  Summary — All Users")
    print(f"{'='*W}")
    print(f"  Channel bit rate : {CHANNEL_BIT_RATE/1e3:.1f} kbps")
    print(f"  TX packet size   : {TX_PACKET_BYTES} bytes\n")
    print(f"  {'User':>5}  {'TX':>5}  {'Correct':>8}  {'Lost':>5}  "
          f"{'λ(p/s)':>8}  {'G':>7}  {'BER':>10}  {'PER':>7}")
    print(f"  {'-'*65}")
    for r in results:
        print(f"  {r['user_id']:>5}  {r['tx_pkts']:>5}  {r['correct_pkts']:>8}  "
              f"{r['lost']:>5}  {r['packet_rate']:>8.3f}  {r['offered_load_G']:>7.4f}  "
              f"{r['ber']:>10.6f}  {r['per']:>7.4f}")
    print(f"  {'-'*65}")
    print(f"  {'Total':>5}  {total_tx:>5}  {total_correct:>8}  {'':>5}  "
          f"{'':>8}  {G_total:>7.4f}")
    print(f"\n  Total offered load G : {G_total:.4f}")
    print(f"  Throughput S         : {S:.4f}")
    print(f"  Pure ALOHA theory    : G·e^(-2G) = {G_total * 2.718**(-2*G_total):.4f}")
    print(f"{'='*W}\n")

# ── Main ──────────────────────────────────────────────────────────────
def main():
    tx_files = sorted(glob.glob(TX_FILE_GLOB))
    if not tx_files:
        print(f"No TX files found matching '{TX_FILE_GLOB}'")
        return

    user_ids = []
    for f in tx_files:
        m = re.search(r'data_tx(\d+)\.csv', f)
        if m:
            uid = int(m.group(1))
            if NUM_USERS is None or uid <= NUM_USERS:
                user_ids.append((uid, f))

    results = []
    for user_id, tx_file in user_ids:
        tx, intervals = load_tx(tx_file, user_id)
        rx            = load_rx(RX_FILE, user_id)
        s             = analyse_user(user_id, tx, rx, intervals)
        print_user_report(s)
        results.append(s)

    print_summary(results)

if __name__ == "__main__":
    main()