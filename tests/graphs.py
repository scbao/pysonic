#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-14 18:37:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-21 16:12:32

''' Test the basic functionalities of the package and output graphs of the call flows. '''

import logging
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

from PySONIC.utils import logger
from PySONIC import BilayerSonophore, SolverUS
from PySONIC.neurons import CorticalRS


# Set logging level
logger.setLevel(logging.DEBUG)

# Create Graphviz output object
graphviz = GraphvizOutput()


def graph_BLS():
    logger.info('Graph 1: BLS initialization')
    a = 32e-9  # nm
    Fdrive = 3.5e5  # Hz
    Cm0 = 1e-2  # membrane resting capacitance (F/m2)
    Qm0 = -80e-5  # membrane resting charge density (C/m2)
    graphviz.output_file = 'graphs/bls_init.png'
    with PyCallGraph(output=graphviz):
        bls = BilayerSonophore(a, Fdrive, Cm0, Qm0)
    logger.info('Graph 2: Mechanical simulation')
    Adrive = 1e5  # Pa
    graphviz.output_file = 'graphs/MECH_sim.png'
    with PyCallGraph(output=graphviz):
        bls.run(Fdrive, Adrive, Qm0)


def graph_neuron_init():
    logger.info('Graph 1: Channels mechanism initialization')
    graphviz.output_file = 'graphs/RS_neuron_init.png'
    with PyCallGraph(output=graphviz):
        CorticalRS()


def graph_ESTIM():

    logger.info('Graph 1: Neuron initialization')
    graphviz.output_file = 'graphs/ESTIM_solver_init.png'
    with PyCallGraph(output=graphviz):
        neuron = CorticalRS()

    logger.info('Graph 2: E-STIM simulation')
    Astim = 1.0  # mA/m2
    tstim = 1e-3  # s
    toffset = 1e-3  # s
    graphviz.output_file = 'graphs/ESTIM_sim.png'
    with PyCallGraph(output=graphviz):
        neuron.run(Astim, tstim, toffset)


def graph_ASTIM():
    rs_neuron = CorticalRS()

    a = 32e-9  # nm
    Fdrive = 3.5e5  # Hz
    Adrive = 1e5  # Pa

    logger.info('Graph 1: SolverUS initialization')
    graphviz.output_file = 'graphs/ASTIM_solver_init.png'
    with PyCallGraph(output=graphviz):
        solver = SolverUS(a, rs_neuron, Fdrive)

    logger.info('Graph 2: A-STIM classic simulation')
    tstim = 1e-6  # s
    toffset = 0.0  # s
    graphviz.output_file = 'graphs/ASTIM_sim_classic.png'
    with PyCallGraph(output=graphviz):
        solver.run(rs_neuron, Fdrive, Adrive, tstim, toffset, sim_type='classic')

    logger.info('Graph 3: A-STIM sonic simulation')
    tstim = 1e-3  # s
    toffset = 0.0  # s
    graphviz.output_file = 'graphs/ASTIM_sim_sonic.png'
    with PyCallGraph(output=graphviz):
        solver.run(rs_neuron, Fdrive, Adrive, tstim, toffset, sim_type='sonic')

    logger.info('Graph 4: A-STIM hybrid simulation')
    tstim = 1e-3  # s
    toffset = 0.0  # s
    graphviz.output_file = 'graphs/ASTIM_sim_hybrid.png'
    with PyCallGraph(output=graphviz):
        solver.run(rs_neuron, Fdrive, Adrive, tstim, toffset, sim_type='hybrid')


if __name__ == '__main__':
    logger.info('Starting graphs')
    graph_BLS()
    graph_neuron_init()
    graph_ESTIM()
    graph_ASTIM()
    logger.info('All graphs successfully created')
