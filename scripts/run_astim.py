# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-13 18:16:09
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-06 15:20:46

''' Run A-STIM simulations of a specific point-neuron. '''

from PySONIC.core import NeuronalBilayerSonophore, Batch
from PySONIC.utils import logger
from PySONIC.parsers import AStimParser


def main():
    # Parse command line arguments
    parser = AStimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Create simulation queue
    queue = NeuronalBilayerSonophore.simQueue(
        *parser.parseSimInputs(args), outputdir=args['outputdir'], overwrite=args['overwrite'])

    # If QSS method selected, add QSS vars (if any) to simulation parameters
    for item in queue:
        if item[-1] == 'qss':
            item.append(args['qss'])

    # Run A-STIM batch
    logger.info("Starting A-STIM simulation batch")
    output = []
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
