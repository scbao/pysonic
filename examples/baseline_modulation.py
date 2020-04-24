# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-24 10:22:56
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-24 11:14:47

import logging
import matplotlib.pyplot as plt

from PySONIC.utils import logger
from PySONIC.neurons import getPointNeuron
from PySONIC.core import DrivenNeuronalBilayerSonophore, AcousticDrive
from PySONIC.core.protocols import *
from PySONIC.plt import CompTimeSeries

# Set logging level
logger.setLevel(logging.INFO)

# Point-neuron model and sonophore radius
pneuron = getPointNeuron('TC')
a = 32e-9  # sonophore radius (m)

# Ultrasonic drive
USdrive = AcousticDrive(
    500e3,  # Hz
    100e3)  # Pa

# Bursting protocol
tstart = 500e-3  # s
tburst = 100e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -
BRF = 1.0        # Hz
nbursts = 3      # -
p = BurstProtocol(tburst, PRF=PRF, DC=DC, BRF=BRF, nbursts=nbursts, tstart=tstart)

# Simulate for a bunch of driving currents
Idrives = [-3.0, -1.5, 0., 4.0, 5.0]  # mA/m2
outputs = []
for Idrive in Idrives:
    nbls = DrivenNeuronalBilayerSonophore(Idrive, a, pneuron)
    outputs.append(nbls.simulate(USdrive, p))

# Plot comparative profiles
labels = [f'Idrive = {x:.1f} mA/m2' for x in Idrives]
CompTimeSeries(outputs, 'Qm').render(labels=labels)

plt.show()
