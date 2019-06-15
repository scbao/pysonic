# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2016-11-21 10:46:56
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-15 13:06:28

''' Run simulations of the NICE mechanical model. '''

import matplotlib.pyplot as plt

from PySONIC.core import BilayerSonophore, Batch
from PySONIC.utils import logger
from PySONIC.plt import SchemePlot
from PySONIC.parsers import MechSimParser


def main():
    # Parse command line arguments
    parser = MechSimParser()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Run MECH batch
    logger.info("Starting mechanical simulation batch")
    pkl_filepaths = []
    inputs = [args[k] for k in ['freq', 'amp', 'charge']]
    for a in args['radius']:
        for d in args['embedding']:
            for Cm0 in args['Cm0']:
                for Qm0 in args['Qm0']:
                    bls = BilayerSonophore(a, Cm0, Qm0, embedding_depth=d)
                    queue = bls.simQueue(*inputs, outputdir=args['outputdir'])
                    batch = Batch(bls.runAndSave, queue)
                    pkl_filepaths += batch(mpi=args['mpi'], loglevel=args['loglevel'])

    # Plot resulting profiles
    if args['plot'] is not None:
        SchemePlot(pkl_filepaths, pltscheme=args['pltscheme'])()
        plt.show()


if __name__ == '__main__':
    main()
