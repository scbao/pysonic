# Description

This package is a Python implementation of the **multi-Scale Optimized Neuronal Intramembrane Cavitation (SONIC) model [1]**, a computationally efficient and interpretable model of neuronal intramembrane cavitation. It allows to simulate the responses of various neuron types to ultrasonic (and electrical) stimuli.

This package contains three core model classes:
- `BilayerSonophore` defines the underlying **biomechanical model of intramembrane cavitation**.
- `PointNeuron` defines an abstract generic interface to **conductance-based point-neuron electrical models**. It is inherited by classes defining the different neuron types with specific membrane dynamics.
- `NeuronalBilayerSonophore` defines the **full electromechanical model for any given neuron type**. To do so, it inherits from `BilayerSonophore` and receives a specific `PointNeuron` object at initialization.

All three classes contain a `simulate` method to simulate the underlying model's behavior for a given set of stimulation and physiological parameters. The `NeuronalBilayerSonophore.simulate` method contains an additional `method` argument defining whether to perform a detailed (`full`), coarse-grained (`sonic`) or hybrid (`hybrid`) integration of the differential system.

Numerical integration routines are implemented outside the models, in separate `Simulator` classes.

The package also contains modules for graphing utilities, multiprocessing, results post-processing and command line parsing.

# Requirements

- Python 3.6 or more

# Installation

- Open a terminal.

- Activate a Python3 environment if needed, e.g. on the tnesrv5 machine:

```$ source /opt/apps/anaconda3/bin activate```

- Check that the appropriate version of pip is activated:

```$ pip --version```

- Go to the package directory (where the setup.py file is located):

```$ cd <path_to_directory>```

- Insall the package and all its dependencies:

```$ pip install -e .```

# Usage

This package contains conductance-based point-neuron implementations of several generic neuron types, including:
- cortical regular spiking (RS) neuron
- cortical fast spiking (FS) neuron
- cortical low-threshold spiking (LTS) neuron
- cortical intrinsically bursting (IB) neuron
- thalamic reticular (RE) neuron
- thalamo-cortical (TC) neuron
- subthalamic nucleus (STN) neuron


## Python scripts

You can easily run simulations of any implemented point-neuron model under both electrical and ultrasonic stimuli, and visualize the simulation results, in just a few lines of code:

```
import logging
import matplotlib.pyplot as plt

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
from PySONIC.plt import SchemePlot

logger.setLevel(logging.INFO)

# Stimulation parameters
a = 32e-9        # m
Fdrive = 500e3   # Hz
Adrive = 100e3   # Pa
Astim = 10.      # mA/m2
tstim = 250e-3   # s
toffset = 50e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -

# Point-neuron model and corresponding neuronal intramembrane cavitation model
pneuron = getPointNeuron('RS')
nbls = NeuronalBilayerSonophore(a, pneuron)

# Run simulation upon electrical stimulation, and plot results
elec_args = (Astim, tstim, toffset, PRF, DC)
data, tcomp = pneuron.simulate(*elec_args)
logger.info('completed in %.0f ms', tcomp * 1e3)
scheme_plot = SchemePlot([(pneuron.simkey, data, pneuron.meta(*elec_args))])
fig1 = scheme_plot.render()

# Run simulation upon ultrasonic stimulation, and plot results
US_int_method = 'sonic'  # Integration method ('sonic', 'full' or 'hybrid')
US_args = (Fdrive, Adrive, tstim, toffset, PRF, DC, US_int_method)
data, tcomp = nbls.simulate(*US_args)
logger.info('completed in %.0f ms', tcomp * 1e3)
scheme_plot = SchemePlot([(nbls.simkey, data, nbls.meta(*US_args))])
fig2 = scheme_plot.render()

plt.show()
```

## From the command line

You can easily run simulations of all 3 model types using the dedicated command line scripts. To do so, open a terminal in the `scripts` directory.

- Use `run_mech.py` for simulations of the **mechanical model** upon **ultrasonic stimulation**. For instance, for a 32 nm radius bilayer sonophore sonicated at 500 kHz and 100 kPa:

```$ python run_mech.py -a 32 -f 500 -A 100 -p Z```

- Use `run_estim.py` for simulations of **point-neuron models** upon **intracellular electrical stimulation**. For instance, a regular-spiking (RS) neuron injected with 10 mA/m2 intracellular current for 30 ms:

```$ python run_estim.py -n RS -A 10 --tstim 30 -p Vm```

- Use `run_astim.py` for simulations of **point-neuron models** upon **ultrasonic stimulation**. For instance, for a coarse-grained simulation of a 32 nm radius bilayer sonophore within a regular-spiking (RS) neuron membrane, sonicated at 500 kHz and 100 kPa for 150 ms:

```$ python run_astim.py -n RS -a 32 -f 500 -A 100 --tstim 150 --method sonic -p Qm```

The simulation results are saved in `.pkl` files. To view these results directly upon simulation completion, you can use the `-p [xxx]` option, where `[xxx]` can be `all` or a given variable name (e.g. `Z` for membrane deflection, `Vm` for membrane potential, `Qm` for membrane charge density).

You can also easily run batches of simulations by specifying more than one value for any given stimulation parameter (e.g. `-A 100 200` for sonication with 100 and 200 kPa respectively). These batches can be parallelized using multiprocessing to optimize performance, with the extra argument `--mpi`.

Several more options are available. To view them, type in:

```$ python <script_name> -h```


# Extend the package

## Add other neuron types

You can easily add other neuron types into the package, providing their ion channel populations and underlying voltage-gated dynamics equations are known.

To add a new point-neuron model, follow this procedure:

1. Create a new file, and save it in the `neurons` sub-folder, with an explicit name (e.g. `my_neuron.py`)
2. Copy-paste the content of the `template.py` file (also located in the `neurons` sub-folder) into your file
3. In your file, change the **class name** from `TemplateNeuron` to something more explicit (e.g. `MyNeuron`), and change the **neuron name** accordingly (e.g. `myneuron`). This name is a keyword used to refer to the model from outside the class
4. Modify/add **biophysical parameters** of your model (resting parameters, reversal potentials, channel conductances, ionic concentrations, temperatures, diffusion constants, etc...) as class attributes
5. Specify a **tuple of names of your different differential states** (i.e. all the differential variables of your model, except for the membrane potential), in the order you want them to appear in the solution.
6. Modify/add **gating states kinetics** (`alphax` and `betax` methods) that define the voltage-dependent activation and inactivation rates of the different ion channnels gates of your model. Those methods take the membrane potential `Vm` as input and return a rate in `s-1`. **You also need to modify the docstring accordingly, as this information is used by the package**. Alternatively, your can use steady-state open-probabilties (`xinf`) and adaptation time constants (`taux`) methods, but you will need to modify the other class methods accordingly.
7. Modify/add **states derivatives** (`derX` methods) that define the derivatives of your different state variables. Those methods must return a derivative in `<state_unit>/s`. **You also need to modify the docstring accordingly, as this information is used by the package**
8. Modify/add **steady-states** (`Xinf` methods) that define the steady-state values of your different state variables. Those methods must return a steady-state value in `<state_unit>`.
9. Modify/add **membrane currents** (`iXX` methods) of your model. Those methods take relevant gating states and the membrane potential `Vm` as inputs, and must return a current density in `mA/m2`. **You also need to modify the docstring accordingly, as this information is used by the package**
10. Modify the other required methods of the class:
  - The `currents` method that takes a membrane potential value `Vm` and a states vector as inputs, and returns a dictionary of membrane currents
  - The `derStates` method that takes the membrane potential `Vm` and a states vector as inputs, and returns a dictionary of states derivatives
  - The `derEffStates` method that takes a membrane charge density value `Qm`, a states vector, and a lookup dictionary as inputs, and returns a dictionary of effective states derivatives
  - The `steadyStates` method that takes a membrane potential value `Vm` as input, and returns a dictionary of steady-states
  - The `computeEffRates` method that takes a membrane potential array `Vm` as input, and returns a dictionary of effective (i.e. averaged over the `Vm` array) voltage-gated states
11. Add the neuron class to the package, by importing it in the `__init__.py` file of the `neurons` sub-folder:

```from .my_neuron import MyNeuron```

12. Verify your point-neuron model by running simulations under various electrical stimuli and comparing the output to the neurons's expected behavior. Implemented required corrections if any.
13. Pre-compute lookup tables required to run coarse-grained  simulations of the neuron model upon ultrasonic stimulation. To do so, go to the `scripts` directory and run the `run_lookups.py` script with the neuron's name as command line argument, e.g.:

```$ python run_lookups.py -n myneuron --mpi```

If possible, use the `--mpi` argument to enable multiprocessing, as lookups pre-computation greatly benefits from parallelization.

14. That's it! You can now run simulations of your point-neuron model upon ultrasonic stimulation.

# References

[1] Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019). Understanding ultrasound neuromodulation using a computationally efficient and interpretable model of intramembrane cavitation. J. Neural Eng.
