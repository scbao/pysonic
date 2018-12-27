#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-12-27 18:57:43

''' Plot temporal profiles of specific simulation output variables. '''

import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from PySONIC.utils import logger, OpenFilesDialog
from PySONIC.plt import plotComp, plotBatch

# Set logging level
logger.setLevel(logging.INFO)

default_comp = 'Qm'
defaults_batch = {'Q_m': ['Qm'], 'V_m': ['Vm']}
# defaults_batch = {'Pac': ['Pac'], 'Z': ['Z'], 'Cm': ['Cm'], 'Vm': ['Vm'], 'Qm': ['Qm']}


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('--hide', default=False, action='store_true', help='Hide output')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')
    ap.add_argument('-c', '--compare', default=False, action='store_true', help='Comparative graph')
    ap.add_argument('-s', '--save', default=False, action='store_true', help='Save output')
    ap.add_argument('--vars', type=str, nargs='+', default=None, help='Variables to plot')
    ap.add_argument('-f', '--frequency', type=int, default=1, help='Sampling frequency for plot')

    # Parse arguments
    args = ap.parse_args()

    loglevel = logging.DEBUG if args.verbose is True else logging.INFO
    logger.setLevel(loglevel)

    # Select data files
    pkl_filepaths, _ = OpenFilesDialog('pkl')
    if not pkl_filepaths:
        logger.error('No input file')
        return

    # Comparative plot
    if args.compare:
        varname = default_comp if args.vars is None else args.vars[0]
        plotComp(pkl_filepaths, varname=varname)
    else:
        vars_dict = defaults_batch if args.vars is None else {key: [key] for key in args.vars}
        plotBatch(pkl_filepaths, title=True, vars_dict=vars_dict, directory=args.outputdir,
                  plt_save=args.save, ask_before_save=not args.save)

    if not args.hide:
        plt.show()


if __name__ == '__main__':
    main()
