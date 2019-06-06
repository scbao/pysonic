# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-12-09 12:06:01
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-06 21:29:48

''' Sub-panels of SONIC model validation on an STN neuron (response to CW sonication). '''

import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.neurons import OtsukaSTN
from PySONIC.utils import logger, selectDirDialog, Intensity2Pressure
from PySONIC.plt import plotFRProfile, SchemePlot


# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
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

    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Parameters
    neuron = OtsukaSTN()
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    tstim = 1  # s
    toffset = 0.  # s
    PRF = 1e2
    DC = 1.
    nbls = NeuronalBilayerSonophore(a, neuron)

    # Range of intensities
    intensities = neuron.getLowIntensities()  # W/m2

    # Levels depicted with individual traces
    subset_intensities = [112, 114, 123]  # W/m2

    # convert to amplitudes and get filepaths
    amplitudes = Intensity2Pressure(intensities)  # Pa
    fnames = ['{}.pkl'.format(nbls.filecode(Fdrive, A, tstim, toffset, PRF, DC, 'sonic'))
              for A in amplitudes]
    fpaths = [os.path.join(inputdir, 'STN', fn) for fn in fnames]

    # Generate figures
    figs = []
    if 'a' in figset:
        fig = plotFRProfile(fpaths, 'Qm', no_offset=True, no_first=False,
                            zref='A', zscale='lin', cmap='Oranges')
        fig.canvas.set_window_title(figbase + 'a')
        figs.append(fig)
    if 'b' in figset:
        isubset = [np.argwhere(intensities == x)[0][0] for x in subset_intensities]
        subset_amplitudes = amplitudes[isubset]
        titles = ['{:.2f} kPa ({:.0f} W/m2)'.format(A * 1e-3, I)
                  for A, I in zip(subset_amplitudes, subset_intensities)]
        print(titles)
        figtraces = SchemePlot([fpaths[i] for i in isubset], pltscheme={'Q_m': ['Qm']})()
        for fig, title in zip(figtraces, titles):
            fig.axes[0].set_title(title)
            fig.canvas.set_window_title(figbase + 'b {}'.format(title))
            figs.append(fig)

    if args.save:
        for fig in figs:
            s = fig.canvas.get_window_title()
            s = s.replace('(', '- ').replace('/', '_').replace(')', '')
            figname = '{}.pdf'.format(s)
            fig.savefig(os.path.join(inputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
