"""
plots for packet error analysis
ErrorDist: errors per packet distribution
ErrorPos: frequency of errors at an index
"""
import numpy as np
import matplotlib.pyplot as plt
import os

scheme = 'DiZeT-CFO-BCH'
FLAG = 'FS_CFO'
data_dir = 'results/ErrorAnalysis/Simulation'
plot_dir = f'results/ErrorAnalysis/Plots/{scheme}_{FLAG}'
os.makedirs(plot_dir, exist_ok=True)

def hist_stats(counts):
    counts = np.asarray(counts)
    N = counts.sum()
    k = np.arange(len(counts))
    p = counts/N

    mean = np.sum( k * p)
    var = np.sum((k - mean)**2 * p)
    mode = np.argmax(counts)
    cdf = np.cumsum(p)
    median = k[np.searchsorted(cdf, 0.5)]
    return mean, median, mode,np.sqrt(var)

Lh = np.concatenate(([1, 2, 3, 4, 5, 6], np.arange(10, 51, 5)), axis=None)
Lh = [1, 4, 6]
data = {}

for l in Lh:

    filepath = os.path.join(data_dir, f"{scheme}_Lh{l}_{FLAG}.npz")
    with np.load(filepath) as f:
        data[l] = {
            key: f[key].copy() for key in f.files            
        }

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 23,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})       

def plot_dist(ax, counts, xlabel, ylabel, stats_lines=True, limit=None, norm=None):
    norm = counts.sum() if norm is None else norm
    p = counts / norm
    k = np.arange(len(p))

    bars = ax.bar(k, p, width=1.0, edgecolor='k')
    labels = [f'{v:.4f}' if v > 0 else '' for v in p]
    ax.bar_label(bars, labels=labels, fontsize=7, rotation=90, padding=3)

    if stats_lines:
        mean, median, mode, _ = hist_stats(counts)      # note the order
        ax.axvline(mean,   color='r', ls='--', label=f'mean = {mean:.2f}')
        ax.axvline(median, color='g', ls=':',  label=f'median = {median:.1f}')
        ax.axvline(mode,   color='b', ls='-.', label=f'mode = {mode}')
    if limit is not None:
        ax.axvline(limit, color='purple', label='ACPC limit')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(p.max() * 1.25, 1e-3))           # headroom scales with the data
    ax.grid(True, alpha=0.4, ls='--')
    if stats_lines or limit is not None:
        ax.legend(fontsize=8)


for l in Lh:
    ErrorDist     = data[l]['ErrorDist']
    ErrorPosition = data[l]['ErrorPosition']
    noIter        = int(data[l]['noIter'])
    SNR_dB        = data[l]['SNR_dB']
    # os.makedirs(os.path.join(plot_dir, 'Combined'), exist_ok=True)

    for i, snr in enumerate(SNR_dB):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=600)

        plot_dist(ax1, ErrorDist[i], 'Errors per packet', 'Frequency', limit=2)
        plot_dist(ax2, ErrorPosition[i], 'Error position (index)',
                  'P(error at index)', stats_lines=False, norm=noIter)

        fig.suptitle(f'Lh = {l}, SNR = {snr} dB')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f'Lh{l}_SNR{snr}.jpeg'),
                    bbox_inches='tight')
        plt.close(fig)                                  # otherwise figures pile up in memory

    res = np.array([hist_stats(c) for c in ErrorDist])   # columns: mean, median, mode, std
    plt.figure(dpi=600)
    plt.plot(SNR_dB, res[:, 0], 'o-', label='mean')
    plt.plot(SNR_dB, res[:, 1], 's--', label='median')
    plt.plot(SNR_dB, res[:, 2], '^:', label='mode')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Errors per packet')
    plt.legend(loc='best') 
    plt.grid(True, alpha=0.4, ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'ErrorDist_Lh{l}.jpeg'), bbox_inches='tight')
    plt.close()

    # res = np.array([hist_stats(c) for c in ErrorPosition])   # columns: mean, median, mode, std
    # plt.figure(dpi=600)
    # plt.plot(SNR_dB, res[:, 0], 'o-', label='mean')
    # plt.plot(SNR_dB, res[:, 1], 's--', label='median')
    # plt.plot(SNR_dB, res[:, 2], '^:', label='mode')
    # plt.xlabel('SNR (dB)')
    # plt.ylabel('Error Position')
    # plt.legend(loc='best') 
    # plt.grid(True, alpha=0.4, ls='--')
    # plt.tight_layout()
    # plt.savefig(os.path.join(plot_dir, f'ErrorPosition_Lh{l}.jpeg'), bbox_inches='tight')
    # plt.close()