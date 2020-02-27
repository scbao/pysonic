# -*- coding: utf-8 -*-
# @Author: Mariia Popova
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-02-27 21:24:05
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-27 22:53:08

import numpy as np
from ..core import PointNeuron
from ..utils import logger


class MRG(PointNeuron):
    ''' Mammalian myelinated fiber model.

        Reference:
        *McIntyre, C.C., Richardson, A.G., and Grill, W.M. (2002). Modeling the excitability
        of mammalian nerve fibers: influence of afterpotentials on the recovery cycle.
        J. Neurophysiol. 87, 995–1006.*
    '''

    # Neuron name
    name = 'MRG'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 2e-2  # Membrane capacitance (F/m2)
    Vm0 = -80.  # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 50.     # Sodium
    EK = -90.     # Potassium
    ELeak = -90.  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNafbar = 3e4   # Fast Sodium
    gNapbar = 100.  # Persistent Sodium
    gKsbar = 800.   # Slow Potassium
    gLeak = 70.     # Non-specific leakage

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNaf activation gate',
        'h': 'iNaf inactivation gate',
        'p': 'iNap activation gate',
        's': 'iKs activation gate',
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def alpham(cls, Vm):
        return 6.57 * cls.vtrap(-(Vm + 20.4), 10.3) * 1e3  # s-1

    @classmethod
    def betam(cls, Vm):
        return 0.304 * cls.vtrap(Vm + 25.7, 9.16) * 1e3  # s-1

    @classmethod
    def alphah(cls, Vm):
        return 0.34 * cls.vtrap(Vm + 114., 11.) * 1e3  # s-1

    @classmethod
    def betah(cls, Vm):
        return 12.6 / (1 + np.exp(-(Vm + 31.8) / 13.4)) * 1e3 # s-1

    @classmethod
    def alphap(cls, Vm):
        return 0.0353 * cls.vtrap(-(Vm + 27.), 10.2) * 1e3  # s-1

    @classmethod
    def betap(cls, Vm):
        return 0.000883 * cls.vtrap(Vm + 34., 10.) * 1e3  # s-1

    @classmethod
    def alphas(cls, Vm):
        return 0.3 / (1 + np.exp(-(Vm + 53.) / 5)) * 1e3  # s-1

    @classmethod
    def betas(cls, Vm):
        return 0.03 / (1 + np.exp(-(Vm + 90.))) * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'p': lambda Vm, x: cls.alphap(Vm) * (1 - x['p']) - cls.betap(Vm) * x['p'],
            's': lambda Vm, x: cls.alphas(Vm) * (1 - x['s']) - cls.betas(Vm) * x['s'],
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'p': lambda Vm: cls.alphap(Vm) / (cls.alphap(Vm) + cls.betap(Vm)),
            's': lambda Vm: cls.alphas(Vm) / (cls.alphas(Vm) + cls.betas(Vm)),
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNaf(cls, m, h, Vm):
        ''' fast Sodium current. '''
        return cls.gNafbar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iNap(cls, p, Vm):
        ''' persistent Sodium current. '''
        return cls.gNapbar * p**3 * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKs(cls, s, Vm):
        ''' slow Potassium current '''
        return cls.gKsbar * s * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNaf': lambda Vm, x: cls.iNaf(x['m'], x['h'], Vm),
            'iNap': lambda Vm, x: cls.iNap(x['p'], Vm),
            'iKs': lambda Vm, x: cls.iKs(x['s'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }

    def chooseTimeStep(self):
        ''' neuron-specific time step for fast dynamics. '''
        return super().chooseTimeStep() * 1e-2