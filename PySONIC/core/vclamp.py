# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-14 13:49:25
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-01 20:07:24

import numpy as np
import pandas as pd

from .batches import Batch
from .protocols import TimeProtocol
from .model import Model
from .pneuron import PointNeuron
from .simulators import OnOffSimulator
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

    @property
    @staticmethod
    def inputs():
        # Get pneuron input vars and replace stimulation current by held and step voltages
        inputvars = PointNeuron.inputs
        for key in ['Astim', 'PRF', 'DC']:
            del inputvars[key]
        inputvars.update({
            'Vhold': {
                'desc': 'held membrane potential',
                'label': 'V_{hold}',
                'unit': 'mV',
                'precision': 0
            },
            'Vstep': {
                'desc': 'step membrane potential',
                'label': 'V_{step}',
                'unit': 'mV',
                'precision': 0
            }
        })
        return inputvars

    def filecodes(self, Vhold, Vstep, tp):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'Vhold': f'{Vhold:.1f}mV',
            'Vstep': f'{Vstep:.1f}mV',
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
        queue = Batch.createQueue(holds, steps, durations, offsets)
        queue = [[item[0], item[1], TimeProtocol(item[2], item[3])] for item in queue]
        return queue

    @staticmethod
    def checkInputs(Vhold, Vstep, tp):
        ''' Check validity of stimulation parameters.

            :param Vhold: held membrane potential (mV)
            :param Vstep: step membrane potential (mV)
            :param tp: time protocol object
        '''
        for k, v in {'Vhold': Vhold, 'Vstep': Vstep}.items():
            if not isinstance(v, float):
                raise TypeError(f'Invalid {k} parameter (must be float typed)')
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
    def simulate(self, Vhold, Vstep, tp):
        ''' Simulate a specific neuron model for a set of simulation parameters,
            and return output data in a dataframe.

            :param Vhold: held membrane potential (mV)
            :param Vstep: step membrane potential (mV)
            :param tp: time protocol object
            :return: output dataframe
        '''
        # Set initial conditions
        y0 = self.pneuron.getSteadyStates(Vhold)

        # Initialize simulator and compute solution
        logger.debug('Computing solution')
        simulator = OnOffSimulator(
            lambda t, y: self.derivatives(t, y, Vm=Vstep),
            lambda t, y: self.derivatives(t, y, Vm=Vhold))
        t, y, stim = simulator(y0, DT_EFFECTIVE, tp)

        # Prepend initial conditions (prior to stimulation)
        t, y, stim = simulator.prependSolution(t, y, stim)

        # Compute clamped membrane potential vector
        Vm = np.zeros(stim.size)
        Vm[stim == 0] = Vhold
        Vm[stim == 1] = Vstep

        # Store output in dataframe and return
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Qm': Vm * 1e-3 * self.pneuron.Cm0,
            'Vm': Vm,
        })
        for i in range(len(self.pneuron.states)):
            data[self.pneuron.statesNames()[i]] = y[:, i]
        return data

    def meta(self, Vhold, Vstep, tp):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'Vhold': Vhold,
            'Vstep': Vstep,
            'tp': tp
        }

    def desc(self, meta):
        return '{}: simulation @ Vhold = {}V, Vstep = {}V, {}'.format(
            self, *si_format([meta['Vhold'] * 1e-3, meta['Vstep'] * 1e-3], 1), meta['tp'].desc)
