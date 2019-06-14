# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-14 11:30:00

''' Run E-STIM simulations of a specific point-neuron. '''

import matplotlib.pyplot as plt

from PySONIC.core import Batch
from PySONIC.utils import logger
from PySONIC.plt import SchemePlot
from PySONIC.parsers import EStimParser


def main():
    # Parse command line arguments
    parser = EStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Run E-STIM batch
    logger.info("Starting E-STIM simulation batch")
    pkl_filepaths = []
    inputs = [args[k] for k in ['amp', 'tstim', 'toffset', 'PRF', 'DC']]
    for pneuron in args['neuron']:
        queue = pneuron.simQueue(*inputs, outputdir=args['outputdir'])
        batch = Batch(pneuron.runAndSave, queue)
        pkl_filepaths += batch(mpi=args['mpi'], loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        scheme_plot = SchemePlot(pkl_filepaths, pltscheme=args['pltscheme'])
        scheme_plot.render(mark_spikes=args['markspikes'])
        plt.show()


if __name__ == '__main__':
    main()
