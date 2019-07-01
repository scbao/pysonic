# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-13 18:16:09
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-01 15:50:03

''' Run A-STIM simulations of a specific point-neuron. '''

from PySONIC.core import NeuronalBilayerSonophore, Batch
from PySONIC.utils import logger
from PySONIC.parsers import AStimParser


def main():
    # Parse command line arguments
    parser = AStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Run A-STIM batch
    logger.info("Starting A-STIM simulation batch")
    output = []
    queue = NeuronalBilayerSonophore.simQueue(*parser.parseSimInputs(args),
                                              outputdir=args['outputdir'])
    for a in args['radius']:
        for pneuron in args['neuron']:
            nbls = NeuronalBilayerSonophore(a, pneuron)
            batch = Batch(nbls.simAndSave if args['save'] else nbls.simulate, queue)
            output += batch(mpi=args['mpi'], loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        parser.parsePlot(args, output)

if __name__ == '__main__':
    main()
