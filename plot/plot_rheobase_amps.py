# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-04-30 21:06:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-23 15:28:02

''' Plot neuron-specific rheobase acoustic amplitudes for various duty cycles. '''

import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from PySONIC.utils import logger, InputError, getNeuronsDict, si_format
from PySONIC.core import NeuronalBilayerSonophore

# Set logging level
logger.setLevel(logging.INFO)

# Set plot parameters
fs = 15  # font size
ps = 100  # scatter point size
lw = 2  # linewidth

# Define input parameters
Fdrive = 500e3  # Hz
a = 32e-9  # m
DCs = np.arange(1, 101) / 1e2
neurons = ['RS', 'FS', 'LTS', 'RE', 'TC']

# Initialize figure
fig, ax = plt.subplots()
ax.set_xlabel('Duty cycle (%)', fontsize=fs)
ax.set_ylabel('Rheobase amplitude (kPa)', fontsize=fs)
for item in ax.get_xticklabels() + ax.get_yticklabels():
    item.set_fontsize(fs)
ax.set_yscale('log')
ax.set_ylim([10, 600])

# Loop through neuron types
for n in neurons:
    neuron = getNeuronsDict()[n]()
    try:
        # Find and plot rheobase amplitudes for duty cycles
        nbls = NeuronalBilayerSonophore(a, neuron)
        logger.info('Computing %s neuron rheobase amplitudes at %sHz', neuron.name, si_format(Fdrive))
        Athrs = nbls.findRheobaseAmps(Fdrive, DCs, neuron.VT)
        ax.plot(DCs * 1e2, Athrs * 1e-3, label='{} neuron'.format(neuron.name))

    except InputError as err:
        logger.error(err)
        sys.exit(1)

ax.legend()
fig.tight_layout()

plt.show()
