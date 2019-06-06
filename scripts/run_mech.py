#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-21 10:46:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-06 18:21:13

''' Run simulations of the NICE mechanical model. '''

import matplotlib.pyplot as plt

from PySONIC.core import BilayerSonophore, Batch
from PySONIC.utils import logger
from PySONIC.plt import SchemePlot
from PySONIC.parsers import MechSimParser


def main():
    # Parse command line arguments
    parser = MechSimParser()
    parser.addOutputDir()
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    args['outputdir'] = parser.parseOutputDir(args)

    # Run MECH batch
    logger.info("Starting mechanical simulation batch")
    pkl_filepaths = []
    for a in args['radius']:
        for d in args['embedding']:
            for Cm0 in args['Cm0']:
                for Qm0 in args['Qm0']:
                    bls = BilayerSonophore(a, Cm0, Qm0, embedding_depth=d)
                    queue = bls.simQueue(args['freq'], args['amp'], args['charge'])
                    for item in queue:
                        item.insert(0, args['outputdir'])
                    batch = Batch(bls.runAndSave, queue)
                    pkl_filepaths += batch(mpi=args['mpi'], loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        SchemePlot(pkl_filepaths, pltscheme=parser.parsePltScheme(args))()
        plt.show()


if __name__ == '__main__':
    main()
