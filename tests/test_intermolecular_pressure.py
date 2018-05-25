import logging
import numpy as np
from PointNICE.utils import logger, si_format, PmCompMethod, rmse, rsquared
from PointNICE.neurons import CorticalRS
from PointNICE.solvers import BilayerSonophore
from PointNICE.constants import *

# Set logging level
logger.setLevel(logging.INFO)

# Create standard bls object
neuron = CorticalRS()
A = 100e3
f = 500e3
a = 32e-9
Cm0 = neuron.Cm0
Qm0 = neuron.Vm0 * Cm0 * 1e-3
Qm = Qm0

# Create sonophore object
bls = BilayerSonophore(a, f, Cm0, Qm0)

# Run simulation with integrated intermolecular pressure
_, y, _ = bls.run(f, A, Qm, Pm_comp_method=PmCompMethod.direct)
Z1, _ = y[:, -NPC_FULL:]
deltaZ1 = Z1.max() - Z1.min()
logger.info('simulation with standard Pm: Zmin = %.2f nm, Zmax = %.2f nm, dZ = %.2f nm',
            Z1.min() * 1e9, Z1.max() * 1e9, deltaZ1 * 1e9)

# Run simulation with predicted intermolecular pressure
_, y, _ = bls.run(f, A, Qm, Pm_comp_method=PmCompMethod.predict)
Z2, _ = y[:, -NPC_FULL:]
deltaZ2 = Z2.max() - Z2.min()
logger.info('simulation with predicted Pm: Zmin = %.2f nm, Zmax = %.2f nm, dZ = %.2f nm',
            Z2.min() * 1e9, Z2.max() * 1e9, deltaZ2 * 1e9)

error_Z = rmse(Z1, Z2)
r2_Z = rsquared(Z1, Z2)
logger.info('Z-error: R2 = %.4f, RMSE = %.4f nm (%.4f%% dZ)',
            r2_Z, error_Z * 1e9, error_Z / deltaZ1 * 1e2)
