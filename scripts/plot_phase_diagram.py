#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-25 22:36:33

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
    ap.add_argument('-i', '--inputfiles', type=str, nargs='+', default=None, help='Input files')
    ap.add_argument('--var', type=str, default='Vm', help='Variable to plot')
    ap.add_argument('--nooffset', default=False, action='store_true',
                    help='Discard post-offset spikes')
    ap.add_argument('--nofirst', default=False, action='store_true',
                    help='Discard first spike')
    ap.add_argument('--tbounds', type=float, nargs='+', default=None, help='Spike interval bounds')
    ap.add_argument('-l', '--labels', type=str, nargs='+', default=None, help='Labels')

    # Parse arguments
    args = ap.parse_args()

    if args.inputfiles is None:
        filepaths, _ = OpenFilesDialog('pkl')
        if not filepaths:
            logger.error('No input file')
            return
    else:
        filepaths = args.inputfiles

    loglevel = logging.DEBUG if args.verbose is True else logging.INFO
    logger.setLevel(loglevel)

    # Plot phase-plane diagram
    print(args.var)
    plotPhasePlane(filepaths, args.var, no_offset=args.nooffset, no_first=args.nofirst,
                   tbounds=args.tbounds, labels=args.labels)
    plt.show()


if __name__ == '__main__':
    main()
