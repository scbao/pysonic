# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-04 15:26:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-04 15:29:52

import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, NeuronalBilayerSonophore, ElectricDrive, AcousticDrive
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
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

# Set pulsing protocol
tstim = 250e-3   # s
toffset = 50e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -
pp = PulsedProtocol(tstim, toffset, PRF, DC)

# Run simulation upon electrical stimulation, and plot results
data, meta = pneuron.simulate(ELdrive, pp)
GroupedTimeSeries([(data, meta)]).render()

# Run simulation upon ultrasonic stimulation, and plot results
data, meta = nbls.simulate(USdrive, pp)
GroupedTimeSeries([(data, meta)]).render()

# Show figures
plt.show()
