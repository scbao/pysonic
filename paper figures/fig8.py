import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.neurons import getNeuronsDict
from PySONIC.utils import logger, si_format, selectDirDialog, cm2inch


def getThresholdAmplitudes(root, neuron, a, Fdrive, tstim, PRF):
    subfolder = '{} {:.0f}nm {}Hz PRF{}Hz {}s'.format(
        neuron, a * 1e9,
        *si_format([Fdrive, PRF, tstim], 0, space='')
    )

    fname = 'log_ASTIM.xlsx'
    fpath = os.path.join(root, subfolder, fname)

    df = pd.read_excel(fpath, sheet_name='Data')
    DCs = df['Duty factor'].values
    Athrs = df['Adrive (kPa)'].values

    iDCs = np.argsort(DCs)
    DCs = DCs[iDCs]
    Athrs = Athrs[iDCs]

    return DCs, Athrs


def plotRheobasevsThresholdAmps(root, neuron, radii, freqs, PRF, tstim,
                                fs=10, colors=None, figsize=None):
    ''' Plot comparative threshold excitation amplitudes of a specific neuron determined by
        (1) quasi-steady approximation and (2) titration procedures, as a function
        of duty cycle, for various combinations of sonophore radius and US frequency.

        :param neuron: neuron name
        :param radii: list of sonophore radii (m)
        :param freqs: list US frequencies (Hz)
        :param PRF: pulse repetition frequency used for titration procedures (Hz)
        :param tstim: stimulus duration used for titration procedures
        :return: figure handle
    '''
    if figsize is None:
        figsize = cm2inch(8, 7)
    neuron = getNeuronsDict()[neuron]()
    linestyles = ['--', ':', '-.']
    assert len(freqs) <= len(linestyles), 'too many frequencies'
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('{} neuron'.format(neuron.name), fontsize=fs)
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.set_yscale('log')
    ax.set_xlim([0, 100])
    ax.set_ylim([10, 600])
    DCs = np.arange(1, 101) / 1e2
    icolor = 0
    for i, a in enumerate(radii):
        nbls = NeuronalBilayerSonophore(a, neuron)
        for j, Fdrive in enumerate(freqs):
            Arheobases, Aref = nbls.findRheobaseAmps(DCs, Fdrive, neuron.VT)
            if colors is None:
                color = 'C{}'.format(icolor)
            else:
                color = colors[icolor]
            lbl = 'rheobase {:.0f} nm radius sonophore, {}Hz'.format(
                a * 1e9, si_format(Fdrive, 0, space=' '))
            ax.plot(DCs * 1e2, Arheobases * 1e-3, '--', c=color, label=lbl)
            DCs2, Athrs = getThresholdAmplitudes(root, neuron.name, a, Fdrive, tstim, PRF)
            lbl = 'threshold {:.0f} nm radius sonophore, {}Hz, {}Hz PRF'.format(
                a * 1e9, *si_format([Fdrive, PRF], 0, space=' '))
            ax.plot(DCs2 * 1e2, Athrs, c=color, label=lbl)
            icolor += 1
    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    return fig


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-i', '--inputdir', type=str, help='Input directory')
    ap.add_argument('-f', '--figset', type=str, nargs='+', help='Figure set', default='all')
    ap.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save output figures as pdf')

    args = ap.parse_args()
    loglevel = logging.DEBUG if args.verbose is True else logging.INFO
    logger.setLevel(loglevel)
    inputdir = selectDirDialog() if args.inputdir is None else args.inputdir
    if inputdir == '':
        logger.error('No input directory chosen')
        return
    figset = args.figset
    if figset == 'all':
        figset = ['a', 'b', 'c']

    # Parameters
    radii = np.array([16, 32, 64]) * 1e-9  # m
    a = radii[1]
    freqs = np.array([20, 500, 4000]) * 1e3  # Hz
    Fdrive = freqs[1]
    PRFs = np.array([1e1, 1e2, 1e3])  # Hz
    PRF = PRFs[1]
    tstim = 1  # s

    colors = plt.get_cmap('tab20c').colors

    # Generate figures
    figs = []
    if 'a' in figset:
        fig = plotRheobasevsThresholdAmps(inputdir, 'RS', radii, [Fdrive], PRF, tstim,
                                          fs=12, colors=colors[:3][::-1])
        fig.canvas.set_window_title('fig8a')
        figs.append(fig)
    if 'b' in figset:
        fig = plotRheobasevsThresholdAmps(inputdir, 'RS', [a], freqs, PRF, tstim,
                                          fs=12, colors=colors[8:11][::-1])
        fig.canvas.set_window_title('fig8b')
        figs.append(fig)
    if 'c' in figset:
        fig = plotRheobasevsThresholdAmps(inputdir, 'LTS', radii, [Fdrive], PRF, tstim,
                                          fs=12, colors=colors[:3][::-1])
        fig.canvas.set_window_title('fig8c')
        figs.append(fig)
    if 'd' in figset:
        fig = plotRheobasevsThresholdAmps(inputdir, 'LTS', [a], freqs, PRF, tstim,
                                          fs=12, colors=colors[8:11][::-1])
        fig.canvas.set_window_title('fig8d')
        figs.append(fig)

    if args.save:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(inputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
