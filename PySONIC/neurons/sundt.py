# -*- coding: utf-8 -*-
# @Author: Mariia Popova
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-10-03 15:58:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-10-10 15:28:25

import numpy as np

from ..core import PointNeuron


class Sundt(PointNeuron):
    ''' Sundt neuron only sodium and delayed-rectifier potassium currents '''

    # Neuron name
    name = 'Sundt'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-6   # Membrane capacitance (F/m2)
    Vm0 = -60  # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 55.0     # Sodium
    EK = -90.0     # Potassium
    ELeak = -110  # Non-specific leakage ???

    # Maximal channel conductances (S/m2)
    gNabar = 400.0  # Sodium
    gKdbar = 400.0   # Delayed-rectifier Potassium
    gLeak = 200.0   # Non-specific leakage ?????

    # Additional parameters
    #VT = -56.2  # Spike threshold adjustment parameter (mV) ????

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKdr gate',
        'l': 'iKdr Borg-Graham formalism gate'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def alpham(cls, Vm):
        return 0.55 * (7.1 - Vm) / np.exp((7.1-Vm) / 4) * 1e3 # s-1 ?? do i need 1e3

    @classmethod
    def betam(cls, Vm):
        return 0.48 * cls.vtrap((Vm - 46.1), 5) * 1e3  # s-1 ?? do i need 1e3

    @classmethod
    def alphah(cls, Vm):
        return 0.22 * np.exp((23 - Vm) / 18) * 1e3  # s-1 ?? do i need 1e3

    @classmethod
    def betah(cls, Vm):
        return 6.92 / (1 + np.exp((46 - Vm) / 5)) * 1e3 # s-1 ?? do i need 1e3

    @classmethod
    def alphan(cls, Vm):
        return np.exp((-5e-3 * (Vm + 32) * 9.648e4) / 2562.35) * 1e3  # s-1 ?? do i need 1e3

    @classmethod
    def betan(cls, Vm):
        return np.exp((-2e-3 * (Vm + 32) * 9.648e4) / 2562.35) * 1e3  # s-1 ?? do i need 1e3

    @classmethod
    def alphal(cls, Vm):
        return np.exp((2e-3 * (Vm + 61) * 9.648e4) / 2562.35) * 1e3  # s-1 ?? do i need 1e3

    @classmethod
    def betal(cls, Vm):
        return np.exp((-2e-3 * (Vm + 32) * 9.648e4) / 2562.35) * 1e3  # s-1 ?? do i need 1e3

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'], #?
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'], #?
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n'], #?
            'l': lambda Vm, x: cls.alphal(Vm) * (1 - x['l']) - cls.betal(Vm) * x['l']  #?
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'n': lambda Vm: 1 / (cls.alphan(Vm) + 1),
            'l': lambda Vm: 1 / (cls.alphal(Vm) + 1)
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current '''
        return cls.gNabar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKdr(cls, n, l, Vm):
        ''' delayed-rectifier Potassium current '''
        return cls.gKdbar * n**3 * l * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iKdr': lambda Vm, x: cls.iKdr(x['n'], x['l'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }
