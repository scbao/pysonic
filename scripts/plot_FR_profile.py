# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-13 12:41:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-14 08:15:03

''' Plot firing rate temporal profile of specific simulation outputs. '''

import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from PySONIC.utils import logger, OpenFilesDialog
from PySONIC.plt import plotFRProfile

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
    ap.add_argument('--log', action='store_true', default=False,
                    help='Log color scale')
    ap.add_argument('-c', '--cmap', type=str, default=None,
                    help='Colormap name')
    ap.add_argument('--ref', type=str, default='A',
                    help='Color code reference')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}

    zscale = 'log' if args['log'] else 'lin'
    cmap = args.get('cmap', None)
    zref = args.get('ref', None)

    try:
        filepaths = args['inputfiles'] if 'inputfiles' in args else OpenFilesDialog('pkl')[0]
    except ValueError as err:
        logger.error(err)
        return

    loglevel = logging.DEBUG if args['verbose'] else logging.INFO
    logger.setLevel(loglevel)

    # Plot phase-plane diagram
    plotFRProfile(filepaths, args['var'], no_offset=args['nooffset'], no_first=args['nofirst'],
                  zref=zref, zscale=zscale, cmap=cmap)
    plt.show()


if __name__ == '__main__':
    main()
