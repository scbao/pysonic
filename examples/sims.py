# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-17 16:09:42
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-17 20:05:44

''' Example script showing how to simulate a point-neuron model upon application
    of both electrical and ultrasonic stimuli, with various temporal protocols.
'''

import logging
import matplotlib.pyplot as plt

from PySONIC.utils import logger
from PySONIC.neurons import getPointNeuron
from PySONIC.core import NeuronalBilayerSonophore, ElectricDrive, AcousticDrive
from PySONIC.core.protocols import *
from PySONIC.plt import GroupedTimeSeries

# Set logging level
logger.setLevel(logging.INFO)

# Define point-neuron model and corresponding neuronal bilayer sonophore model
pneuron = getPointNeuron('RS')
a = 32e-9  # sonophore radius (m)
nbls = NeuronalBilayerSonophore(a, pneuron)

# Define electric and ultrasonic drives
ELdrive = ElectricDrive(20.)  # mA/m2
USdrive = AcousticDrive(
    500e3,  # Hz
    100e3)  # Pa

# Pulsing parameters
tburst = 100e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -
BRF = 1.0        # Hz
nbursts = 3      # -

# Protocols
protocols = [
    TimeProtocol(tburst, 1 / BRF - tburst),
    PulsedProtocol(tburst, 1 / BRF - tburst, PRF=PRF, DC=DC),
    BurstProtocol(tburst, PRF=PRF, DC=DC, BRF=BRF, nbursts=nbursts)
]

# For each protocol
for p in protocols:
    # Run simulation upon electrical stimulation, and plot results
    data, meta = pneuron.simulate(ELdrive, p)
    GroupedTimeSeries([(data, meta)]).render()

    # Run simulation upon ultrasonic stimulation, and plot results
    data, meta = nbls.simulate(USdrive, p)
    GroupedTimeSeries([(data, meta)]).render()

# Show figures
plt.show()
