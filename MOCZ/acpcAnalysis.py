"""
BMOCZ-ACPC analysis in AWGN, CFO, flat fading channel 
and frequency selective channel
"""
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = "results/conSim/heuristicsJ/simulation"
plot_dir = "results/conSim/heuristicsJ/plotsACPC"

Lh = 1

os.makedirs(plot_dir, exist_ok=True)

schemes = [ 'AWGN', 'CFO', 'FS', 'FS_CFO']
# schemes = [ 'BMOCZ-ACPC', 'DiZeT-CFO', 'DiZeT-CFO-BCH', 'J-BMOCZ', 'MJ-BMOCZ' ]
colors = ["black", "tab:green", "tab:blue", "tab:purple", 'tab:orange']
linestyles = ["-", "--", "-.", ":", (0, (3, 1, 2, 4))]
markers = ["", "o", "s", "d", "1", "x", "*", "^", '+']
markerColor = ["None", "None", "tab:blue", "tab:purple"]

data = {}
for scheme in schemes:

    # filepath = os.path.join(data_dir, f"{scheme}_Lh{Lh}.npz")
    filepath = os.path.join(data_dir, f"BMOCZ-ACPC_Lh{Lh}_{scheme}.npz" )

    with np.load(filepath) as f:
        data[scheme] = {
            key: f[key].copy()
            for key in f.files
        }

# Metrics and plotting configurations
metrics = {
    'BER': {
        'ylabel': 'BER',
        'scale': 'log',
        'ylim': (1e-4, 1)
    },
    'PER': {
        'ylabel': 'PER',
        'scale': 'log',
        'ylim': (1e-4, 1)
    },
    'PAPR': {
        'ylabel': 'PAPR (dB)',
        'scale': 'linear',
        'ylim': None
    },
    'Goodput': {
        'ylabel': 'Throughput (bits/sec)',
        'scale': 'linear',
        'ylim': (-0.01, 1.01)
    },
    'Rotation': {
        'ylabel': 'NMAE',
        'scale': 'log',
        'ylim': (1e-4, 1)
    }
}

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 23,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Plot each metric
for metric, config in metrics.items():

    plt.figure(figsize=(8, 6), dpi=800)

    for i, scheme in enumerate(schemes):

        snr = data[scheme]['SNR_dB']
        values = data[scheme][metric]

        if config['scale'] == 'log':
            plt.semilogy(
                snr, values,
                linewidth=1.2,
                linestyle=linestyles[i],
                color=colors[i],
                marker=markers[i],
                markersize=8,
                label=scheme
            )
        else:
            plt.plot(
                snr, values,
                linewidth=1.2,
                linestyle=linestyles[i],
                color=colors[i],
                marker=markers[i],
                markersize=8,
                label=scheme
            )

    if config['ylim'] is not None:
        plt.ylim(*config['ylim'])

    plt.xlim(0, 40)
    plt.xlabel("SNR (dB)")
    plt.ylabel(config['ylabel'])
    plt.grid(True, alpha=0.6, linestyle='--')
    plt.legend(loc='best', framealpha=0.2, fontsize=20)
    plt.tight_layout()

    plt.savefig( 
        os.path.join(plot_dir, f"{metric.lower()}_Lh{Lh}.pdf"),
        bbox_inches="tight"
    )

    plt.close()

print("All comparison plots generated.")