#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-14 18:37:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-07-18 15:54:31

''' Test the basic functionalities of the package and output graphs of the call flows. '''

import logging
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

import PointNICE
from PointNICE.utils import LoadParams
from PointNICE.channels import CorticalRS


# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# Create Graphviz output object
graphviz = GraphvizOutput()

# Set geometry of NBLS structure
geom = {"a": 32e-9, "d": 0.0e-6}

# Loading model parameters
params = LoadParams()

# Defining general stimulation parameters
Fdrive = 3.5e5  # Hz
Adrive = 1e5  # Pa
PRF = 1.5e3  # Hz
DF = 1


logger.info('Graph 1: BLS initialization')
Cm0 = 1e-2  # membrane resting capacitance (F/m2)
Qm0 = -89e-5  # membrane resting charge density (C/m2)
graphviz.output_file = 'graphs/bls_init.png'
with PyCallGraph(output=graphviz):
    bls = PointNICE.BilayerSonophore(geom, params, Fdrive, Cm0, Qm0)


logger.info('Graph 2: Channels mechanism initialization')
graphviz.output_file = 'graphs/sim_mech.png'
with PyCallGraph(output=graphviz):
    bls.runMech(Fdrive, 2e4, Qm0)


logger.info('Graph 3: Channels mechanism initialization')
graphviz.output_file = 'graphs/channel_init.png'
with PyCallGraph(output=graphviz):
    rs_mech = CorticalRS()


logger.info('Graph 4: SolverUS initialization')
graphviz.output_file = 'graphs/solver_init.png'
with PyCallGraph(output=graphviz):
    solver = PointNICE.SolverUS(geom, params, rs_mech, Fdrive)


logger.info('Graph 5: classic simulation')
tstim = 1e-3  # s
toffset = 1e-3  # s
graphviz.output_file = 'graphs/sim_classic.png'
with PyCallGraph(output=graphviz):
    solver.runSim(rs_mech, Fdrive, Adrive, tstim, toffset, PRF, DF, 'classic')


logger.info('Graph 6: effective simulation')
tstim = 30e-3  # s
toffset = 10e-3  # s
graphviz.output_file = 'graphs/sim_effective.png'
with PyCallGraph(output=graphviz):
    solver.runSim(rs_mech, Fdrive, Adrive, tstim, toffset, PRF, DF, 'effective')


logger.info('Graph 7: hybrid simulation')
tstim = 10e-3  # s
toffset = 1e-3  # s
graphviz.output_file = 'graphs/sim_hybrid.png'
with PyCallGraph(output=graphviz):
    solver.runSim(rs_mech, Fdrive, Adrive, tstim, toffset, PRF, DF, 'hybrid')



