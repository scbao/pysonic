# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-14 13:49:25
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-16 12:17:48

import numpy as np

from .protocols import TimeProtocol
from .model import Model
from .pneuron import PointNeuron
from .solvers import EventDrivenSolver
from .drives import Drive, VoltageDrive
from ..constants import *
from ..utils import *
from ..neurons import getPointNeuron


class VoltageClamp(Model):

    tscale = 'ms'  # relevant temporal scale of the model
    simkey = 'VCLAMP'  # keyword used to characterize simulations made with this model

    def __init__(self, pneuron):
        ''' Constructor of the class.

            :param pneuron: point-neuron model
        '''
        # Check validity of input parameters
        if not isinstance(pneuron, PointNeuron):
            raise ValueError(
                f'Invalid neuron type: "{pneuron.name}" (must inherit from PointNeuron class)')
        self.pneuron = pneuron

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pneuron})'

    def copy(self):
        return self.__class__(self.pneuron)

    @property
    def meta(self):
        return {'neuron': self.pneuron.name}

    @classmethod
    def initFromMeta(cls, meta):
        return cls(getPointNeuron(meta['neuron']))

    def params(self):
        return self.pneuron.params()

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        return self.pneuron.getPltVars(wrapleft, wrapright)

    @property
    def pltScheme(self):
        return self.pneuron.pltScheme

    def filecode(self, *args):
        return Model.filecode(self, *args)

    @staticmethod
    def inputs():
        return VoltageDrive.inputs()

    def filecodes(self, drive, tp):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            **drive.filecodes,
            **tp.filecodes
        }

    @classmethod
    @Model.checkOutputDir
    def simQueue(cls, holds, steps, durations, offsets, **kwargs):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps.

            :param holds: list (or 1D-array) of held membrane potentials
            :param steps: list (or 1D-array) of step membrane potentials
            :param durations: list (or 1D-array) of stimulus durations
            :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
            :return: list of parameters (list) for each simulation
        '''
        drives = VoltageDrive.createQueue(holds, steps)
        protocols = TimeProtocol.createQueue(durations, offsets)
        queue = []
        for drive in drives:
            for tp in protocols:
                queue.append([drive, tp])
        return queue

    @staticmethod
    def checkInputs(drive, tp):
        ''' Check validity of stimulation parameters.

            :param drive: voltage drive object
            :param tp: time protocol object
        '''
        if not isinstance(drive, Drive):
            raise TypeError(f'Invalid "drive" parameter (must be an "Drive" object)')
        if not isinstance(tp, TimeProtocol):
            raise TypeError('Invalid time protocol (must be "TimeProtocol" instance)')

    def derivatives(self, t, y, Vm=None):
        if Vm is None:
            Vm = self.pneuron.Vm0
        states_dict = dict(zip(self.pneuron.statesNames(), y))
        return self.pneuron.getDerStates(Vm, states_dict)

    @Model.addMeta
    @Model.logDesc
    @Model.checkSimParams
    def simulate(self, drive, tp):
        ''' Simulate a specific neuron model for a set of simulation parameters,
            and return output data in a dataframe.

            :param drive: voltage drive object
            :param tp: time protocol object
            :return: output dataframe
        '''
        # Set initial conditions
        y0 = {k: self.pneuron.steadyStates()[k](drive.Vhold) for k in self.pneuron.statesNames()}

        # Initialize solver and compute solution
        Vfunc = lambda x: (drive.Vstep - drive.Vhold) * x + drive.Vhold  # voltage setting function
        solver = EventDrivenSolver(
            lambda x: setattr(solver, 'V', Vfunc(x)),          # eventfunc
            y0.keys(),                                         # variables list
            lambda t, y: self.derivatives(t, y, Vm=solver.V),  # dfunc
            dt=DT_EFFECTIVE)                                   # time step
        data = solver(y0, tp.stimEvents(), tp.ttotal)

        # Compute clamped membrane potential vector
        Vm = np.zeros(len(data))
        Vm[data['stimstate'] == 0] = drive.Vhold
        Vm[data['stimstate'] == 1] = drive.Vstep

        # Add Qm and Vm timeries to solution
        data = addColumn(data, 'Qm', Vm * 1e-3 * self.pneuron.Cm0, preceding_key='stimstate')
        data = addColumn(data, 'Vm', Vm, preceding_key='Qm')

        # Return solution dataframe
        return data

    def desc(self, meta):
        return f'{self}: simulation @ {meta["drive"].desc}, {meta["tp"].desc}'
