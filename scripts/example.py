import logging
import matplotlib.pyplot as plt

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.neurons import CorticalRS
from PySONIC.utils import logger
from PySONIC.plt import SchemePlot

logger.setLevel(logging.INFO)

# Point-neuron model
pneuron = CorticalRS()

# Stimulation parameters
a = 32e-9        # m
Fdrive = 500e3   # Hz
Adrive = 100e3   # Pa
tstim = 250e-3   # s
toffset = 50e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -

# Integration method ('sonic', 'full' or 'hybrid')
method = 'sonic'

# Initialize model and run simulation
nbls = NeuronalBilayerSonophore(a, pneuron)
args = (Fdrive, Adrive, tstim, toffset, PRF, DC, method)
data, tcomp = nbls.simulate(*args)
meta = nbls.meta(*args)

# Plot results
scheme_plot = SchemePlot([(nbls.simkey, data, meta)])
fig = scheme_plot.render()

plt.show()
