## Description

Python implementation of the **multi-Scale Optimized Neuronal Intramembrane Cavitation** (SONIC) model to compute individual neural responses to acoustic stimuli, as predicted by the *intramembrane cavitation* hypothesis.

This package contains several core modules:
- `bls` defines the underlying biomechanical model of intramembrane cavitation (`BilayerSonophore` class), and provides an integration method to predict compute the mechanical oscillations of the plasma membrane subject to a periodic acoustic perturbation.
- `pneuron` defines a generic `PointNeuron` class that contains methods to simulate a *Hodgkin-Huxley* point-neuron model.
- `neurons` contains classes that inherit from `PointNeuron` and define intrinsic channels mechanisms of several neurons types.
- `sonic` defines a `SonicNeuron` class that inherits from `BilayerSonophore` and receives a specific `PointNeuron` child instance at initialization, and implements the bi-directional link between the mechanical and electrical parts of the NICE model. It also provides several integration methods (detailed below) to simulate the full electro-mechanical model behavior upon sonication:
	- `runFull` solves all variables for the entire duration of the simulation. This method uses very small time steps and is computationally expensive (simulation time: several hours)
	- `runHybrid` integrates the system by consecutive “slices” of time, during which the full system is solved until mechanical stabilization, at which point the electrical system is solely solved with predicted mechanical variables until the end of the slice. This method is more efficient (simulation time: several minutes) and provides accurate results.
	- `runSONIC` integrates a coarse-grained, effective differential system to grasp the net effect of the acoustic stimulus on the electrical system. This method allows to run simulations of the electrical system in only a few seconds, with very accurate results of the net membrane charge density evolution, but requires the pre-computation of lookup tables.

As well as some additional modules:
- `plt` defines graphing utilities to load results of several simulations and display/compare temporal profiles of multiple variables of interest across simulations.
- `utils` defines generic utilities used across the different modules


## Installation

Install Python 3 if not already done.

Open a terminal.

Activate a Python3 environment if needed, e.g. on the tnesrv5 machine:

```$ source /opt/apps/anaconda3/bin activate```

Check that the appropriate version of pip is activated:

```$ pip --version```

Go to the package directory (where the setup.py file is located) and install it:

```
$ cd <path_to_directory>
$ pip install -e .
```

*PySONIC* and all its dependencies will be installed.

## Usage

### Command line scripts

Open a terminal at the package root directory.

Use `ESTIM.py` to simulate a point-neuron model upon electrical stimulation, e.g. for a *regular-spiking neuron* injected with 10 mA/m2 intracellular current for 30 ms:

```python ESTIM.py -n=RS -A=10 -d=30```

Use `MECH.py` to simulate mechanical model upon sonication (until periodic stabilization), e.g. for a 32 nm diameter sonophore sonicated at 500 kHz and 100 kPa:

```python MECH.py -a=32 -f=500 -A=100```

Use `ASTIM.py` to simulate the full electro-mechanical model of a given neuron type upon sonication, e.g. for a 32 nm diameter sonophore within a *regular-spiking neuron* sonicated at 500 kHz and 100 kPa for 150 ms:

```python ASTIM.py -n=RS -a=32 -f=500 -A=100 -d=150```

If several values are defined for a given parameter, a batch of simulations is run (for every value of the parameter sweep).
You can also specify these values from within the script (*defaults* dictionary)

The simulation results will be save in an output PKL file. To view these results, you can use the `-p` option

Several more options are available. To view them, type in

```python <script_name> -h```

