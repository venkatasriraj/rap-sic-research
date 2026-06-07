import numpy as np 
import matplotlib.pyplot as plt 

_ACCESS_CODE_BYTES = [0xE1, 0x5A, 0xE8, 0x93]

_DIFF_INIT_STATE   = 1
sps = 10
fs = 40 * 1e3
alpha = 0.35
num_taps = 110

def _bytes_to_bits(byte_list):
    """Convert list of bytes to list of bits, MSB first."""
    bits = []
    for b in byte_list:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


def _differential_encode(bits, init_state=1):
    """
    DBPSK differential encoding (matches GNU Radio Differential Encoder).
    out[0] = in[0] XOR init_state
    out[k] = in[k] XOR out[k-1]
    """
    enc  = []
    prev = init_state
    for b in bits:
        s    = b ^ prev
        enc.append(s)
        prev = s
    return enc


def _bpsk_map(bits):
    """Map bits to BPSK symbols: 0→+1, 1→-1."""
    return np.array([1.0 - 2.0 * b for b in bits], dtype=np.complex64)


def _make_rrc_taps(sps, alpha, num_taps):
    """
    Generate Root Raised Cosine filter taps.
    Matches GNU Radio's firdes.root_raised_cosine with gain=1.
    """
    taps   = np.zeros(num_taps)
    center = (num_taps - 1) / 2.0
    for i in range(num_taps):
        t = (i - center) / sps
        if t == 0.0:
            taps[i] = (1.0 + alpha * (4.0 / np.pi - 1.0))
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
    # Normalize
    taps /= np.sqrt(np.sum(taps ** 2))
    return taps.astype(np.float32)

# load i-q samples from the .bin to perfoem autocorrelation with access code and proceed further
fileID = fopen('sync_samples.bin', 'r');

raw_data = fread(fileID, 'float32');
fclose(fileID);

I = raw_data(1:2:end);
Q = raw_data(2:2:end);

iq_samples = complex(I, Q);

plt.plot(iq_samples);
