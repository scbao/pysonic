# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-13 18:16:09
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-26 19:05:02

''' Run A-STIM simulations of a specific point-neuron. '''

import matplotlib.pyplot as plt

from PySONIC.core import NeuronalBilayerSonophore, Batch
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries
from PySONIC.parsers import AStimParser


def main():
    # Parse command line arguments
    parser = AStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Run A-STIM batch
    logger.info("Starting A-STIM simulation batch")
    pkl_filepaths = []
    inputs = [args[k] for k in ['freq', 'amp', 'tstim', 'toffset', 'PRF', 'DC', 'fs', 'method']]
    for a in args['radius']:
        for pneuron in args['neuron']:
            nbls = NeuronalBilayerSonophore(a, pneuron)
            queue = nbls.simQueue(*inputs, outputdir=args['outputdir'])
            batch = Batch(nbls.runAndSave, queue)
            pkl_filepaths += batch(mpi=args['mpi'], loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        scheme_plot = GroupedTimeSeries(pkl_filepaths, pltscheme=args['pltscheme'])
        scheme_plot.render(spikes=args['spikes'])
        plt.show()


if __name__ == '__main__':
    main()
