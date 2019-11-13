# Description

`PySONIC` is a Python implementation of the **multi-Scale Optimized Neuronal Intramembrane Cavitation (SONIC) model [1]**, a computationally efficient and interpretable model of neuronal intramembrane cavitation. It allows to simulate the responses of various neuron types to ultrasonic (and electrical) stimuli.

## Content of repository

### Models

The package contains four model classes:
- `Model` defines the generic interface of a model, including mandatory attributes and methods for simulating it.
- `BilayerSonophore` defines the underlying **biomechanical model of intramembrane cavitation**.
- `PointNeuron` defines an abstract generic interface to **conductance-based point-neuron electrical models**. It is inherited by classes defining the different neuron types with specific membrane dynamics.
- `NeuronalBilayerSonophore` defines the **full electromechanical model for any given neuron type**. To do so, it inherits from `BilayerSonophore` and receives a specific `PointNeuron` object at initialization.

All model classes contain a `simulate` method to simulate the underlying model's behavior for a given set of stimulation and physiological parameters. The `NeuronalBilayerSonophore.simulate` method contains an additional `method` argument defining whether to perform a detailed (`full`), coarse-grained (`sonic`) or hybrid (`hybrid`) integration of the differential system.

### Simulators

Numerical integration routines are implemented outside the models, in separate `Simulator` classes:
- `PeriodicSimulator` integrates a differential system periodically until a stable periodic behavior is detected.
- `PWSimulator` integrates a differential system given a specific temporal stimulation pattern (pulse repetition frequency, stimulus duty cycle and post-stimulus offset), using different derivative functions for "ON" (with stimulus) and "OFF" (without stimulus) periods
- `HybridSimulator` inherits from both `PeriodicSimulator`and `PWSimulator`. It integrates a differential system using a hybrid scheme inside each "ON" or "OFF" period:
  1. The full ODE system is integrated for a few cycles with a dense time granularity until a periodic stabilization detection
  2. The profiles of all variables over the last cycle are resampled to a far lower (i.e. sparse) sampling rate
  3. A subset of the ODE system is integrated with a sparse time granularity, while the remaining variables are periodically expanded from their last cycle profile, until the end of the period or that of an predefined update interval.
  4. The process is repeated from step 1

### Neurons

Several conductance-based point-neuron models are implemented that inherit from the `PointNeuron` generic interface:
- `CorticalRS`: cortical regular spiking (`RS`) neuron
- `CorticalFS`: cortical fast spiking (`FS`) neuron
- `CorticalLTS`: cortical low-threshold spiking (`LTS`) neuron
- `CorticalIB`: cortical intrinsically bursting (`IB`) neuron
- `ThalamicRE`: thalamic reticular (`RE`) neuron
- `ThalamoCortical`: thalamo-cortical (`TC`) neuron
- `OstukaSTN`: subthalamic nucleus (`STN`) neuron
- `FrankenhaeuserHuxley`: Xenopus myelinated fiber node (`FH`)

### Other modules

- `batches`: a generic interface to run simulation batches with or without multiprocessing
- `parsers`: command line parsing utilities
- `plt`: graphing utilities
- `postpro`: post-processing utilities (mostly signal features detection)
- `constants`: algorithmic constants used across modules and classes
- `utils`: generic utilities

# Requirements

- Python 3.6+
- Package dependencies (numpy, scipy, ...) are installed automatically upon installation of the package.

# Installation

- Open a terminal.

- Activate a Python3 environment if needed, e.g. on the tnesrv5 machine:

```source /opt/apps/anaconda3/bin activate```

- Check that the appropriate version of pip is activated:

```pip --version```

- Clone the repository and install the python package:

```git clone https://c4science.ch/diffusion/4670/pysonic.git```

```cd pysonic```

```pip install -e .```

# Usage

## Python scripts

You can easily run simulations of any implemented point-neuron model under both electrical and ultrasonic stimuli, and visualize the simulation results, in just a few lines of code:

```python
import logging
import matplotlib.pyplot as plt

from PySONIC.core import PulsedProtocol, NeuronalBilayerSonophore
from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger
from PySONIC.plt import GroupedTimeSeries

logger.setLevel(logging.INFO)

# Stimulation parameters
a = 32e-9        # m
Fdrive = 500e3   # Hz
Adrive = 100e3   # Pa
Astim = 10.      # mA/m2

# Pulsing parameters
tstim = 250e-3   # s
toffset = 50e-3  # s
PRF = 100.       # Hz
DC = 0.5         # -
pp = PulsedProtocol(tstim, toffset, PRF, DC)

# Point-neuron model and corresponding neuronal intramembrane cavitation model
pneuron = getPointNeuron('RS')
nbls = NeuronalBilayerSonophore(a, pneuron)

# Run simulation upon electrical stimulation, and plot results
data, meta = pneuron.simulate(Astim, pp)
fig1 = GroupedTimeSeries([(data, meta)]).render()

# Run simulation upon ultrasonic stimulation, and plot results
data, meta = nbls.simulate(Fdrive, Adrive, pp)
fig2 = GroupedTimeSeries([(data, meta)]).render()

plt.show()
```

## From the command line

### Running simulations and batches

You can easily run simulations of all 3 model types using the dedicated command line scripts. To do so, open a terminal in the `scripts` directory.

- Use `run_mech.py` for simulations of the **mechanical model** upon **ultrasonic stimulation**. For instance, for a 32 nm radius bilayer sonophore sonicated at 500 kHz and 100 kPa:

```python run_mech.py -a 32 -f 500 -A 100 -p Z```

- Use `run_estim.py` for simulations of **point-neuron models** upon **intracellular electrical stimulation**. For instance, a regular-spiking (RS) neuron injected with 10 mA/m2 intracellular current for 30 ms:

```python run_estim.py -n RS -A 10 --tstim 30 -p Vm```

- Use `run_astim.py` for simulations of **point-neuron models** upon **ultrasonic stimulation**. For instance, for a coarse-grained simulation of a 32 nm radius bilayer sonophore within a regular-spiking (RS) neuron membrane, sonicated at 500 kHz and 100 kPa for 150 ms:

```python run_astim.py -n RS -a 32 -f 500 -A 100 --tstim 150 --method sonic -p Qm```

Additionally, you can run batches of simulations by specifying more than one value for any given stimulation parameter (e.g. `-A 100 200` for sonication with 100 and 200 kPa respectively). These batches can be parallelized using multiprocessing to optimize performance, with the extra argument `--mpi`.

### Saving and visualizing results

By default, simulation results are neither shown, nor saved.

To view results directly upon simulation completion, you can use the `-p [xxx]` option, where `[xxx]` can be `all` (to plot all resulting variables) or a given variable name (e.g. `Z` for membrane deflection, `Vm` for membrane potential, `Qm` for membrane charge density).

To save simulation results in binary `.pkl` files, you can use the `-s` option. You will be prompted to choose an output directory, unless you also specify it with the `-o <output_directory>` option. Output files are automatically named from model and simulation parameters to avoid ambiguity.

When running simulation batches, it is highly advised to specify the `-s` option in order to save results of each simulation. You can then visualize results at a later stage.

To visualize results, use the `plot_timeseries.py` script. You will be prompted to select the output files containing the simulation(s) results. By default, separate figures will be created for each simulation, showing the time profiles of all resulting variables. Here again, you can choose to show only a subset of variables using the `-p [xxx]` option. Moreover, if you select a subset of variables, you can visualize resulting profiles across simulations in comparative figures wih the `--compare` option.

Several more options are available. To view them, type in:

```python <script_name> -h```


# Extend the package

## Add other neuron types

You can easily add other neuron types into the package, providing their ion channel populations and underlying voltage-gated dynamics equations are known.

To add a new point-neuron model, follow this procedure:

1. Create a new file, and save it in the `neurons` sub-folder, with an explicit name (e.g. `my_neuron.py`).
2. Copy-paste the content of the `template.py` file (also located in the `neurons` sub-folder) into your file.
3. In your file, change the **class name** from `TemplateNeuron` to something more explicit (e.g. `MyNeuron`), and change the **neuron name** accordingly (e.g. `myneuron`). This name is a keyword used to refer to the model from outside the class.
4. Modify/add **biophysical parameters** of your model (resting parameters, reversal potentials, channel conductances, ionic concentrations, temperatures, diffusion constants, etc...) as class attributes. If some parameters are not fixed and must be computed, assign them to the class inside a  `__new__` method, taking the class (`cls`) as sole attribute.
5. Specify a **dictionary of names:descriptions of your different differential states** (i.e. all the differential variables of your model, except for the membrane potential).
6. Modify/add **gating states kinetics** (`alphax` and `betax` methods) that define the voltage-dependent activation and inactivation rates of the different ion channnels gates of your model. Those methods take the membrane potential `Vm` as input and return a rate in `s-1`. Alternatively, your can use steady-state open-probabilties (`xinf`) and adaptation time constants (`taux`) methods.
7. Modify the `derStates` method that defines the **derivatives of your different state variables**. These derivatives are defined inside a dictionary, where each state key is paired to a lambda function that takes the membrane potential `Vm` and a states vector `x` as inputs, and returns the associated state derivative (in `<state_unit>/s`).
8. Modify the `steadyStates` method that defines the **steady-state values of your different state variables**. These steady-states are defined inside a dictionary, where each state key is paired to a lambda function that takes the membrane potential `Vm` as only input, and returns the associated steady-state value (in `<state_unit>`). If some steady-states depend on the values of other-steady states, you can proceed as follows:
   - define all independent steady-states functions in a dictionary called `lambda_dict`
   - add dependent steady-state functions to the dictionary, calling `lambda_dict[k](Vm)` for each state `k` whose value is required.
9. Modify/add **membrane currents** (`iXX` methods) of your model. Those methods take relevant gating states and the membrane potential `Vm` as inputs, and must return a current density in `mA/m2`. **You also need to modify the docstring accordingly, as this information is used by the package**.
10. Modify the `currents` method that defines the **membrane currents of your model**. These currents are defined inside a dictionary, where each current key is paired to a lambda function that takes the membrane potential `Vm` and a states vector `x` as inputs, and returns the associated current (in `mA/m2`).

**The `derStates`, `steadyStates` and `currents` methods are automatically parsed by the package to adapt neuron models to US stimulation. Hence, make sure to**:
   - **keep them as class methods**
   - **check that all calls to functions that depend solely on `Vm` appear directly in the methods' lambda expressions and are not hidden inside nested function calls.**

11. Add the neuron class to the package, by importing it in the `__init__.py` file of the `neurons` sub-folder:

```python
from .my_neuron import MyNeuron
```

12. Verify your point-neuron model by running simulations under various electrical stimuli and comparing the output to the neurons's expected behavior. Implemented required corrections if any.
13. Pre-compute lookup tables required to run coarse-grained  simulations of the neuron model upon ultrasonic stimulation. To do so, go to the `scripts` directory and run the `run_lookups.py` script with the neuron's name as command line argument, e.g.:

```python run_lookups.py -n myneuron --mpi```

If possible, use the `--mpi` argument to enable multiprocessing, as lookups pre-computation greatly benefits from parallelization.

That's it! You can now run simulations of your point-neuron model upon ultrasonic stimulation.

## Future developments

Here is a list of future developments:

- [ ] Add quasi-steady state analysis module
- [x] Integration within the [NEURON simulation environment](https://www.neuron.yale.edu/neuron/)
- [x] Spatial expansion into nanoscale multicompartmental model
- [ ] Spatial expansion into morphological realistic fiber models
- [ ] Model validation against experimental data (leech neurons)

# Authors

Code written and maintained by Theo Lemaire (theo.lemaire@epfl.ch).

# License

This project is licensed under the MIT License - see the LICENSE file for details.


# References

[1] Lemaire, T., Neufeld, E., Kuster, N., and Micera, S. (2019). Understanding ultrasound neuromodulation using a computationally efficient and interpretable model of intramembrane cavitation. J. Neural Eng.
