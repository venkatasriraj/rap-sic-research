"""
plots
"""
import numpy as np
import os
import matplotlib.pyplot as plt

m, n = 20, 20
data_dir = f'results/simulation/U{m}_S{n}'
plot_dir = f'results/plots/U{m}_S{n}'
os.makedirs(plot_dir, exist_ok=True)

# schemes = [ 'BMOCZ-ACPC', 'DiZeT-CFO', 'DiZeT-CFO-BCH', 'J-BMOCZ', 'MJ-BMOCZ' ]
colors = ["tab:green", "tab:blue", "tab:purple", "black", 'tab:orange']
linestyles = [":", "--", "-.", "", (0, (3, 1, 2, 4))]
markers = ["x", "1", "*", "d", "o", "", "s", "^", '+']
markerColor = ["None", "None", "tab:blue", "tab:purple"]

schemes = ['BMOCZ', 'DPSK']
modes = ['SIC', 'noSIC', 'CSI']
cfoModes = ['CFO', 'noCFO']
cfo = 'CFO' 
Lh = 1
# xlabel = ['Load', 'snr]
xlabel = 'snr'

hyperParam = 'snr' if xlabel == 'Load' else 'load'

data = {}
for scheme in schemes:
    data_scheme = {}
    for mode in modes:
        filepath = os.path.join(data_dir, f'{scheme}_{mode}{xlabel}_{cfo}{Lh}.npz')

        with np.load(filepath) as f:
            data_scheme[mode] = { key: f[key].copy() for key in f.files }
    data[scheme] = data_scheme

metrics = {
    'BER': {
        'ylabel': 'BER',
        'ylim': (0, 1.05)
    },
    'PER': {
        'ylabel': 'BLER',
        'ylim': (0, 1.05)
    },
    # 'PAPR': {
    #     'ylabel': 'PAPR (dB)',
    #     'scale': 'linear',
    #     'ylim': None
    # },
    'THROUGHPUT': {
        'ylabel': 'Throughput',
        'scale': 'linear',
        'ylim': (0, 0.6)
    },
    # 'Rotation': {
    #     'ylabel': 'MNAE',
    #     'scale': 'log',
    #     'ylim': (1e-4, 1)
    # }
#-----  MAE should be plotted independently
    # 'MAEH': {
    #     'ylabel': 'MNAE',
    #     'scale': 'linear'
    # }
}

plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 26,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
    "pdf.fonttype": 42,
    'ps.fonttype': 42,
})

for metric, config in metrics.items():

    plt.figure(figsize=(8, 6), dpi=800)

    for i, scheme in enumerate(schemes):
        data_scheme = data[scheme]
        for j, mode in enumerate(modes):

            xlabel = data_scheme[mode]['xlabel']
            MODE = data_scheme[mode]['MODE']

            hyperParameter = data_scheme[mode][hyperParam]

            xlabelVal = data_scheme[mode]['xlabelVal']
            values = data_scheme[mode][metric]
            plt.plot(
                xlabelVal, values, linewidth=1.2, linestyle=linestyles[j],
                color=colors[i], marker=markers[j], markersize=8, label=f'{scheme}-{mode}'
            )
    if config['ylim'] is not None:
        plt.ylim(*config['ylim'])

    plt.xlabel(f'{xlabel}')
    plt.ylabel(config['ylabel'])
    plt.grid(True, alpha=0.6, linestyle='--')
    plt.legend(loc='best', framealpha=0.2, fontsize=20)
    plt.tight_layout()

    plt.savefig(
        os.path.join(plot_dir, f'{metric.lower()}_{xlabel}_{hyperParameter}{cfo}{Lh}.jpeg'),
        bbox_inches = 'tight'
    )

    plt.close()

print("All comparison plots generated.")