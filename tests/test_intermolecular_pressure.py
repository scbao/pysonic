import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PySONIC.utils import logger, si_format, PmCompMethod, rmse, rsquared
from PySONIC.plt import cm2inch
from PySONIC.neurons import CorticalRS
from PySONIC.solvers import BilayerSonophore
from PySONIC.constants import *

# Set logging level
logger.setLevel(logging.INFO)

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

fs = 8  # font size
lw = 2  # linewidth
ps = 15  # scatter point size

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

# Compare profiles of direct and approximated intermolecular pressures along Z
Z = np.linspace(-0.4 * bls.Delta_, bls.a, 1000)
Pm_direct = bls.v_PMavg(Z, bls.v_curvrad(Z), bls.surface(Z))
Pm_approx = bls.PMavgpred(Z)
fig, ax = plt.subplots(figsize=cm2inch(6, 5))
for skey in ['right', 'top']:
    ax.spines[skey].set_visible(False)
ax.set_xlabel('Z (nm)', fontsize=fs)
ax.set_ylabel('Pressure (kPa)', fontsize=fs)
ax.set_xticks([0, bls.a * 1e9])
ax.set_xticklabels(['0', 'a'])
ax.set_yticks([-10, 0, 40])
ax.set_ylim([-10, 50])
for item in ax.get_xticklabels() + ax.get_yticklabels():
    item.set_fontsize(fs)
ax.plot(Z * 1e9, Pm_direct * 1e-3, label='$\mathregular{P_m}$')
ax.plot(Z * 1e9, Pm_approx * 1e-3, label='$\mathregular{P_{m,approx}}$')
ax.axhline(y=0, color='k')
ax.legend(fontsize=fs, frameon=False)
fig.tight_layout()

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

plt.show()
