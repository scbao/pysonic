#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-14 18:37:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-28 15:20:18

''' Test the basic functionalities of the package and output graphs of the call flows. '''

import logging
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

from PointNICE import BilayerSonophore, SolverUS, SolverElec
from PointNICE.utils import load_BLS_params
from PointNICE.channels import CorticalRS


# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# Create Graphviz output object
graphviz = GraphvizOutput()

# Geometry parameters of BLS structure
geom = {"a": 32e-9, "d": 0.0e-6}
params = load_BLS_params()


def graph_BLS():
    logger.info('Graph 1: BLS initialization')
    Fdrive = 3.5e5  # Hz
    Cm0 = 1e-2  # membrane resting capacitance (F/m2)
    Qm0 = -80e-5  # membrane resting charge density (C/m2)
    graphviz.output_file = 'graphs/bls_init.png'
    with PyCallGraph(output=graphviz):
        bls = BilayerSonophore(geom, params, Fdrive, Cm0, Qm0)
    logger.info('Graph 2: Mechanical simulation')
    Adrive = 1e5  # Pa
    graphviz.output_file = 'graphs/MECH_sim.png'
    with PyCallGraph(output=graphviz):
        bls.runMech(Fdrive, Adrive, Qm0)


def graph_neuron_init():
    logger.info('Graph 1: Channels mechanism initialization')
    graphviz.output_file = 'graphs/RS_neuron_init.png'
    with PyCallGraph(output=graphviz):
        CorticalRS()


def graph_ESTIM():
    rs_neuron = CorticalRS()

    logger.info('Graph 1: SolverElec initialization')
    graphviz.output_file = 'graphs/ESTIM_solver_init.png'
    with PyCallGraph(output=graphviz):
        solver = SolverElec()

    logger.info('Graph 2: E-STIM simulation')
    Astim = 1.0  # mA/m2
    tstim = 1e-3  # s
    toffset = 1e-3  # s
    graphviz.output_file = 'graphs/ESTIM_sim.png'
    with PyCallGraph(output=graphviz):
        solver.run(rs_neuron, Astim, tstim, toffset)


def graph_ASTIM():
    rs_neuron = CorticalRS()

    Fdrive = 3.5e5  # Hz
    Adrive = 1e5  # Pa

    logger.info('Graph 1: SolverUS initialization')
    graphviz.output_file = 'graphs/ASTIM_solver_init.png'
    with PyCallGraph(output=graphviz):
        solver = SolverUS(geom, params, rs_neuron, Fdrive)

    logger.info('Graph 2: A-STIM classic simulation')
    tstim = 1e-6  # s
    toffset = 0.0  # s
    graphviz.output_file = 'graphs/ASTIM_sim_classic.png'
    with PyCallGraph(output=graphviz):
        solver.run(rs_neuron, Fdrive, Adrive, tstim, toffset, sim_type='classic')

    logger.info('Graph 3: A-STIM effective simulation')
    tstim = 1e-3  # s
    toffset = 0.0  # s
    graphviz.output_file = 'graphs/ASTIM_sim_effective.png'
    with PyCallGraph(output=graphviz):
        solver.run(rs_neuron, Fdrive, Adrive, tstim, toffset, sim_type='effective')

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
