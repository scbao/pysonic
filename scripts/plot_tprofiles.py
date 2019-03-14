#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-14 23:36:39

''' Plot temporal profiles of specific simulation output variables. '''

import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from PySONIC.utils import logger, OpenFilesDialog, selectDirDialog
from PySONIC.plt import plotComp, plotBatch

# Set logging level
logger.setLevel(logging.INFO)


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('--hide', default=False, action='store_true', help='Hide output')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')
    ap.add_argument('-c', '--compare', default=False, action='store_true', help='Comparative graph')
    ap.add_argument('-s', '--save', default=False, action='store_true', help='Save output')
    ap.add_argument('-p', '--plot', type=str, nargs='+', default=None, help='Variables to plot')
    ap.add_argument('-f', '--frequency', type=int, default=1, help='Sampling frequency for plot')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    logger.setLevel(logging.DEBUG if args['verbose'] else logging.INFO)

    # Select data files
    pkl_filepaths, _ = OpenFilesDialog('pkl')
    if not pkl_filepaths:
        logger.error('No input file')
        return

    # Plot appropriate graph
    if args['compare']:
        plotComp(pkl_filepaths, varname=args.get('plot', [None])[0])
    else:
        pltscheme = {key: [key] for key in args['plot']} if 'plot' in args else None
        if 'outputdir' not in args:
            args['outputdir'] = selectDirDialog() if args['save'] else None
        plotBatch(pkl_filepaths, title=True, pltscheme=pltscheme, directory=args['outputdir'],
                  plt_save=args['save'], ask_before_save=not args['save'])
    if not args['hide']:
        plt.show()


if __name__ == '__main__':
    main()
