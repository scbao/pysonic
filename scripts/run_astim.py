#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-06 18:20:17

''' Run A-STIM simulations of a specific point-neuron. '''

import matplotlib.pyplot as plt

from PySONIC.core import NeuronalBilayerSonophore, Batch
from PySONIC.utils import logger
from PySONIC.plt import SchemePlot
from PySONIC.parsers import AStimParser


def main():
    # Parse command line arguments
    parser = AStimParser()
    parser.addOutputDir()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    args['outputdir'] = parser.parseOutputDir(args)

    # Run A-STIM batch
    logger.info("Starting A-STIM simulation batch")
    pkl_filepaths = []
    for a in args['radius']:
        for neuron in args['neuron']:
            nbls = NeuronalBilayerSonophore(a, neuron)
            queue = nbls.simQueue(
                args['freq'],
                args['amp'],
                args['tstim'],
                args['toffset'],
                args['PRF'],
                args['DC'],
                args['method'][0]
            )
            for item in queue:
                item.insert(0, args['outputdir'])
            batch = Batch(nbls.runAndSave, queue)
            pkl_filepaths += batch(mpi=args['mpi'], loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        SchemePlot(pkl_filepaths, pltscheme=parser.parsePltScheme(args))()
        plt.show()


if __name__ == '__main__':
    main()
