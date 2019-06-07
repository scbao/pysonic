# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-07 14:57:30

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
    for neuron in args['neuron']:
        queue = neuron.simQueue(
            args['amp'],
            args['tstim'],
            args['toffset'],
            args['PRF'],
            args['DC'],
        )
        for item in queue:
            item.insert(0, args['outputdir'])
        batch = Batch(neuron.runAndSave, queue)
        pkl_filepaths += batch(mpi=args['mpi'], loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        SchemePlot(pkl_filepaths, pltscheme=parser.parsePltScheme(args))()
        plt.show()


if __name__ == '__main__':
    main()
