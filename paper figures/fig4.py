#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-15 15:59:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-02-27 17:19:42

''' Sub-panels of the effective variables figure. '''

import os
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging

from PySONIC.plt import plotEffectiveVariables
from PySONIC.utils import logger, selectDirDialog
from PySONIC.neurons import getNeuronsDict

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-o', '--outdir', type=str, help='Output directory')
    ap.add_argument('-f', '--figset', type=str, nargs='+', help='Figure set', default='all')
    ap.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save output figures as pdf')

    args = ap.parse_args()
    loglevel = logging.DEBUG if args.verbose is True else logging.INFO
    logger.setLevel(loglevel)
    figset = args.figset
    if figset == 'all':
        figset = ['a', 'b', 'c']

    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Parameters
    neuron_str = 'RS'
    neuron = getNeuronsDict()[neuron_str]()
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    Adrive = 50e3  # Pa

    # Generate figures
    figs = []
    if 'a' in figset:
        fig = plotEffectiveVariables(neuron, a=a, Fdrive=Fdrive, cmap='Oranges', zscale='log')
        fig.canvas.set_window_title(figbase + 'a')
        figs.append(fig)
    if 'b' in figset:
        fig = plotEffectiveVariables(neuron, a=a, Adrive=Adrive, cmap='Greens', zscale='log')
        fig.canvas.set_window_title(figbase + 'b')
        figs.append(fig)
    if 'c' in figset:
        fig = plotEffectiveVariables(neuron, Fdrive=Fdrive, Adrive=Adrive, cmap='Blues', zscale='log')
        fig.canvas.set_window_title(figbase + 'c')
        figs.append(fig)

    if args.save:
        outdir = selectDirDialog() if args.outdir is None else args.outdir
        if outdir == '':
            logger.error('No input directory chosen')
            return
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(outdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
