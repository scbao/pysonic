# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-08-14 13:49:25
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-14 15:12:57

import numpy as np
import pandas as pd

from .model import Model
from .pneuron import PointNeuron
from .simulators import PWSimulator
from ..constants import *
from ..utils import *


class VoltageClamp(Model):

    tscale = 'ms'  # relevant temporal scale of the model
    simkey = 'VCLAMP'  # keyword used to characterize simulations made with this model

    def __init__(self, pneuron):
        ''' Constructor of the class.

            :param pneuron: point-neuron model
        '''
        # Check validity of input parameters
        if not isinstance(pneuron, PointNeuron):
            raise ValueError('Invalid neuron type: "{}" (must inherit from PointNeuron class)'
                             .format(pneuron.name))
        self.pneuron = pneuron

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.pneuron)

    def params(self):
        return self.pneuron.params()

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        return self.pneuron.getPltVars(wrapleft, wrapright)

    def getPltScheme(self):
        return self.pneuron.getPltScheme()

    def filecode(self, *args):
        return Model.filecode(self, *args)

    @staticmethod
    def inputs():
        # Get pneuron input vars and replace stimulation current by held and step voltages
        inputvars = PointNeuron.inputs()
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

    def filecodes(self, Vhold, Vstep, tstim, toffset):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'Vhold': '{:.1f}mV'.format(Vhold),
            'Vstep': '{:.1f}mV'.format(Vstep),
            'tstim': '{:.0f}ms'.format(tstim * 1e3),
            'toffset': None
        }

    @staticmethod
    def checkInputs(Vhold, Vstep, tstim, toffset):
        ''' Check validity of stimulation parameters.

            :param Vhold: held membrane potential (mV)
            :param Vstep: step membrane potential (mV)
            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
        '''
        if not all(isinstance(param, float) for param in [Vhold, Vstep, tstim, toffset]):
            raise TypeError('Invalid stimulation parameters (must be float typed)')
        if tstim <= 0:
            raise ValueError('Invalid stimulus duration: {} ms (must be strictly positive)'
                             .format(tstim * 1e3))
        if toffset < 0:
            raise ValueError('Invalid stimulus offset: {} ms (must be positive or null)'
                             .format(toffset * 1e3))

    def derivatives(self, t, y, Vm=None):
        if Vm is None:
            Vm = self.pneuron.Vm0
        states_dict = dict(zip(self.pneuron.statesNames(), y))
        return self.pneuron.getDerStates(Vm, states_dict)

    @Model.addMeta
    def simulate(self, Vhold, Vstep, tstim, toffset):
        logger.info(
            '%s: simulation @ Vhold = %sV, Vstep = %sV, t = %ss (%ss offset)',
            self, *si_format([Vhold * 1e-3, Vstep * 1e-3, tstim, toffset], 1, space=' '))

        # Check validity of stimulation parameters
        self.checkInputs(Vhold, Vstep, tstim, toffset)

        # Set initial conditions
        y0 = self.pneuron.getSteadyStates(Vhold)

        # Initialize simulator and compute solution
        logger.debug('Computing solution')
        simulator = PWSimulator(
            lambda t, y: self.derivatives(t, y, Vm=Vstep),
            lambda t, y: self.derivatives(t, y, Vm=Vhold))
        t, y, stim = simulator(
            y0, DT_EFFECTIVE, tstim, toffset, None, 1.)
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

    def meta(self, Vhold, Vstep, tstim, toffset):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'Vhold': Vhold,
            'Vstep': Vstep,
            'tstim': tstim,
            'toffset': toffset,
        }

