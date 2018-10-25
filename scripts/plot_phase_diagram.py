#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-25 14:49:28

''' Plot temporal profiles of specific simulation output variables. '''

import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from PySONIC.utils import logger, OpenFilesDialog
from PySONIC.plt import plotPhasePlane

# Set logging level
logger.setLevel(logging.INFO)


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-i', '--inputfile', type=str, default=None, help='Input file')
    ap.add_argument('--var', type=str, default='Vm', help='Variable to plot')
    ap.add_argument('--nooffset', default=False, action='store_true',
                    help='Discard post-offset spikes')
    ap.add_argument('--nofirst', default=False, action='store_true',
                    help='Discard first spike')

    # Parse arguments
    args = ap.parse_args()

    if args.inputfile is None:
        pkl_filepaths, _ = OpenFilesDialog('pkl')
        if not pkl_filepaths:
            logger.error('No input file')
            return
        if len(pkl_filepaths) > 1:
            logger.error('Multiple input files')
            return
        filepath = pkl_filepaths[0]
    else:
        filepath = args.inputfile

    loglevel = logging.DEBUG if args.verbose is True else logging.INFO
    logger.setLevel(loglevel)

    # Plot phase-plane diagram
    plotPhasePlane(filepath, args.var, no_offset=args.nooffset, no_first=args.nofirst)
    plt.show()


if __name__ == '__main__':
    main()
