import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.utils import logger, selectDirDialog, ASTIM_filecode, Intensity2Pressure
from PySONIC.plt import plotFRProfile, plotBatch


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
    if figset is 'all':
        figset = ['a', 'b']

    # Parameters
    neuron = 'STN'
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    tstim = 1  # s
    PRF = 1e2
    DC = 1.

    # Range of intensities
    intensities = np.hstack((
        np.arange(10, 101, 10),
        np.arange(101, 131, 1),
        np.array([140])
    ))  # W/m2

    # remove levels corresponding to overwritten files (similar Adrive decimals)
    todelete = [np.argwhere(intensities == x)[0][0] for x in [108, 115, 122, 127]]
    intensities = np.delete(intensities, todelete)

    # convert to amplitudes and get filepaths
    amplitudes = np.array([Intensity2Pressure(I) for I in intensities])  # Pa
    fnames = ['{}.pkl'.format(ASTIM_filecode(neuron, a, Fdrive, A, tstim, PRF, DC, 'sonic'))
              for A in amplitudes]
    fpaths = [os.path.join(inputdir, 'STN', fn) for fn in fnames]

    # Generate figures
    figs = []
    if 'a' in figset:
        fig = plotFRProfile(fpaths, 'Qm', no_offset=True, no_first=False,
                            zref='A', zscale='lin', cmap='viridis')
        fig.canvas.set_window_title('fig9a')
        figs.append(fig)
    if 'b' in figset:
        # Levels depicted with individual traces
        itraces = [np.argwhere(intensities == x)[0][0] for x in [105, 107, 123]]
        figtraces = plotBatch([fpaths[i] for i in itraces], vars_dict={'Q_m': ['Qm']})
        for i, fig in zip(itraces, figtraces):
            fig.canvas.set_window_title('fig9b {:.2f} kPa'.format(amplitudes[i] * 1e-3))
            figs.append(fig)

    if args.save:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(inputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
