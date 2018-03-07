Description
============

PointNICE is a Python implementation of the **Neuronal Intramembrane Cavitation Excitation** (NICE) model introduced by Plaksin et. al in 2014 and initially developed in MATLAB by its authors. It contains optimized methods to predict the electrical response of point-neuron models to both acoustic and electrical stimuli.

This package contains several core modules:
	- **bls** defines the underlying biomechanical model of intramembrane cavitation (**BilayerSonophore** class), and provides an integration method to predict compute the mechanical oscillations of the plasma membrane subject to a periodic acoustic perturbation.
	- **solvers** contains a simple solver for electrical stimuli (**SolverElec** class) as well as a tailored solver for acoustic stimuli (**SolverUS** class). The latter directly inherits from the BilayerSonophore class upon instantiation, and is hooked to a specific "channel mechanism" in order to link the mechanical model to an electrical model of membrane dynamics. It also provides several integration methods (detailed below) to compute the behaviour of the full electro-mechanical model subject to a continuous or pulsed ultrasonic stimulus.
	- **channels** contains the definitions of the different channels mechanisms inherent to specific neurons, including several types of **cortical** and **thalamic** neurons.
	- **plt** defines plotting utilities to load results of several simulations and display/compare temporal profiles of multiple variables of interest across simulations.
	- **utils** defines generic utilities used across the different modules

The **SolverUS** class incorporates optimized numerical integration methods to perform dynamic simulations of the model subject to acoustic perturbation, and compute the evolution of its mechanical and electrical variables:
	- a **classic** method that solves all variables for the entire duration of the simulation. This method uses very small time steps and is computationally expensive (simulation time: several hours)
	- a **hybrid** method (initially developed by Plaskin et al.) in which integration is performed in consecutive “slices” of time, during which the full system is solved until mechanical stabilization, at which point the electrical system is solely solved with predicted mechanical variables until the end of the slice. This method is more efficient (simulation time: several minutes) and provides accurate results.
	- a newly developed **effective** method that neglects the high amplitude oscillations of mechanical and electrical variables during each acoustic cycle, to instead grasp the net effect of the acoustic stimulus on the electrical system. To do so, the sole electrical system is solved using pre-computed coefficients that depend on membrane charge and acoustic amplitude. This method allows to run simulations of the electrical system in only a few seconds, with very accurate results of the net membrane charge density evolution.

This package is meant to be easy to use as a predictive and comparative tool for researchers investigating ultrasonic and/or electrical neuro-stimulation experimentally.


Installation
==================

Install Python 3 if not already done.

Open a terminal.

Activate a Python3 environment if needed, e.g. on the tnesrv5 machine:

	source /opt/apps/anaconda3/bin activate

Check that the appropriate version of pip is activated:

	pip --version

Go to the PointNICE directory (where the setup.py file is located) and install it as a package:

	cd <path_to_directory>
	pip install -e .

PointNICE and all its dependencies will be installed.


Usage
=======

Command line scripts
---------------------

To run single simulations of a given point-neuron model under specific stimulation parameters, you can use the `ASTIM_run.py` and `ESTIM_run.py` command-line scripts provided by the package.

For instance, to simulate a regular-spiking neuron under continuous wave ultrasonic stimulation at 500kHz and 100kPa, for 150 ms:

	python ASTIM_run.py -n=RS -f=500 -A=100 -t=150

Similarly, to simulate the electrical stimulation of a thalamo-cortical neuron at 10 mA/m2 for 150 ms:

	python ESTIM_run.py -n=TC -A=10 -t=150

The simulation results will be save in an output PKL file in the current working directory. To view these results, you can use the dedicated


Batch scripts
---------------

To run a batch of simulations on different neuron types and spanning ranges of several stimulation parameters, you can run the `ASTIM_batch.py` and `ESTIM_batch.py` scripts. To do so, simply modify the **stim_params** and **neurons** variables with your own neuron types and parameter sweeps, and then run the scripts (without command-line arguments).


