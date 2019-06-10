## Description

This package is a Python implementation of the **multi-Scale Optimized Neuronal Intramembrane Cavitation** (SONIC) model [1] to compute individual neural responses to acoustic stimuli, as predicted by the *intramembrane cavitation* hypothesis.

This package contains three core model classes:
- `BilayerSonophore` defines the underlying biomechanical model of intramembrane cavitation.
- `PointNeuron` defines an abstract generic interface to *Hodgkin-Huxley* point-neuron models. It is inherited by classes defining the different neuron types with specific membrane dynamics.
- `NeuronalBilayerSonophore` defines the full electromechanical model for a particular neuron type. To do so, it inherits from `BilayerSonophore` and receives a specific `PointNeuron` child instance at initialization.

These three classes contain a `simulate` method to simulate the underlying model's behavior for a given set of stimulation and pyhsiological parameters. The `NeuronalBilayerSonophore.simulate` method contains an additional `method` argument defining whether to perform a detailed (`full`), coarse-grained (`sonic`) or hybrid (`hybrid`) integration of the differential system.

Numerical integration routines are implemented outside the models, in separate `Simulator` classes.

The package also contains modules for graphing utilities, multiprocessing, results post-processing and command line parsing.

## Requirements

- Python 3.6 or more

## Installation

- Open a terminal.

- Activate a Python3 environment if needed, e.g. on the tnesrv5 machine:

```$ source /opt/apps/anaconda3/bin activate```

- Check that the appropriate version of pip is activated:

```$ pip --version```

- Go to the package directory (where the setup.py file is located):

```$ cd <path_to_directory>```

- Intsall the package:

```$ pip install -e .```

*PySONIC* and all its dependencies will be installed.

## Usage

### Example script

The script below shows how to:
1. create a `NeuronalBilayerSonophore` model of a point-like cortical regular spiking (`CorticalRS`) neuron
2. simulate the model with specific ultrasound parameters
3. plot the results

```python
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
meta = nbls.meta(*args)  # meta-information dictionary
data, tcomp = nbls.simulate(*args)
logger.info('completed in %.0f ms', tcomp * 1e3)

# Plot results
scheme_plot = SchemePlot([(nbls.simkey, data, meta)])
fig = scheme_plot.render()

plt.show()
```

### From the command line

You can easily run simulations of all 3 model types using the dedicated command line scripts. To do so, open a terminal in the *scripts* directory.

- Use `run_mech.py` for simulations of the **mechanical model** upon **sonication** (until periodic stabilization). For instance, a 32 nm radius bilayer sonophore sonicated at 500 kHz and 100 kPa:

```$ python run_mech.py -a 32 -f 500 -A 100```

- Use `run_estim.py` for simulations of **point-neuron models** upon **electrical stimulation**. For instance, a *regular-spiking neuron* injected with 10 mA/m2 intracellular current for 30 ms:

```$ python run_estim.py -n RS -A 10 --tstim 30```

Use `run_astim.py` for simulations of **point-neuron models** upon **sonication**. For instance, a 32 nm radius bilayer sonophore within a *regular-spiking neuron* membrane, sonicated at 500 kHz and 100 kPa for 150 ms:

```$ python run_astim.py -n RS -a 32 -f 500 -A 100 --tstim 150```

You can also easily run batches of simulations by specifying more than one value for any given stimulation parameter (e.g. `-A 100 200` for sonication with 100 and 200 kPa respectively). These batches can be parallelized using multiprocessing to optimize performance, with the extra argument `--mpi`.

The simulation results are saved in `.pkl` files. To view these results directly upon simulation completion, you can use the `-p [xxx]` option, where [xxx] can be "all" or a given variable name (e.g. "Vm" for membrane potential, "Qm" for membrane charge density).

Several more options are available. To view them, type in:

```$ python <script_name> -h```


## References

[1] Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019). *Understanding ultrasound neuromodulation using a computationally efficient and interpretable model of intramembrane cavitation*. J. Neural Eng.
