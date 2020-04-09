# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-09 10:52:49
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-09 16:17:20

''' Example script showing how to build an neuron activation map. '''

import logging
import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import logger, selectDirDialog
from PySONIC.neurons import getPointNeuron
from PySONIC.plt import ActivationMap

logger.setLevel(logging.INFO)


def main():
    ''' The code must be wrappped inside a main function in order to allow MPI usage. '''
    # Parameters
    root = selectDirDialog()
    pneuron = getPointNeuron('TC')
    a = 32e-9  # m
    f = 500e3  # Hz
    tstim = 100e-3  # s
    PRF = 100  # Hz
    amps = np.logspace(np.log10(10.), np.log10(600.), 10) * 1e3  # Pa
    DCs = np.linspace(5, 100, 20) * 1e-2  # (-)

    # Create activation map object
    actmap = ActivationMap(root, pneuron, a, f, tstim, PRF, amps, DCs)

    # Run simulations for populate the 2D map
    actmap.run(mpi=True)

    # Render the 2D map
    actmap.render(interactive=True, FRbounds=(1e0, 1e3))
    plt.show()


if __name__ == '__main__':
    main()
