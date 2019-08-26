# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-11 15:58:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-26 14:45:26

import numpy as np

from ..core import PointNeuron


class SweeneyNode(PointNeuron):
    ''' Mammalian myelinated motor fiber fiber node.

        Reference:
        *Sweeney, J.D., Mortimer, J.T., and Durand, D. (1987). Modeling of mammalian myelinated nerve
        for functional neuromuscular stimulation. IEEE 9th Annual Conference of the Engineering
        in Medicine and Biology Society 3, 1577–1578.*
    '''

    # Neuron name
    name = 'sweeney'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 2.5e-2  # Membrane capacitance (F/m2)
    Vm0 = -80.0   # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 35.64     # Sodium
    ELeak = -80.01  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 1445e1  # Sodium
    gLeak = 128e1   # Non-specific leakage

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def alpham(cls, Vm):
        return (126 + 0.363 * Vm) / (1 + np.exp(-(Vm + 49) / 53)) * 1e3  # s-1

    @classmethod
    def betam(cls, Vm):
        return cls.alpham(Vm) / (np.exp((Vm + 56.2) / 4.17))  # s-1

    @classmethod
    def betah(cls, Vm):
        return 15.6 / (1 + np.exp(-(Vm + 56) / 10)) * 1e3  # s-1

    @classmethod
    def alphah(cls, Vm):
        return cls.betah(Vm) / np.exp((Vm + 74.5) / 5)  # s-1

    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h']
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm))
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current '''
        return cls.gNabar * m**2 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }
