# -*- coding: utf-8 -*-
# @Author: Mariia Popova
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-10-03 15:58:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-01 14:53:29

import numpy as np

from ..core import PointNeuron


class Sundt(PointNeuron):
    ''' Sundt neuron only sodium and delayed-rectifier potassium currents

        Reference:
        *Sundt D., Gamper N., Jaffe D. B., Spike propagation through the dorsal
        root ganglia in an unmyelinated sensory neuron: a modeling study.
        Journal of Neurophysiology (2015)*
    '''

    # Neuron name
    name = 'sundt'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2   # Membrane capacitance (F/m2)
    Vm0 = -60    # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 55.0     # Sodium
    EK = -90.0     # Potassium

    # Maximal channel conductances (S/m2)
    gNa_bar = 400.0    # Sodium
    gKd_bar = 400.0   # Delayed-rectifier Potassium
    gKm_bar = 4.0      # KCNQ Potassium
    GCa_bar = 30       # Calcium ????
    gKCa_bar = 2.0     # Calcium dependent Potassium ????
    gLeak = 1.0        # Non-specific leakage

    # Additional parameters
    Cai0 = 7e-8     # Calcium concentration at rest (M) (Aradi 1999)
    R = 8.314468    # gas constant (J / K mol)
    F = 96485.332   # Faraday constant (C / mol)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKdr gate',
        'l': 'iKdr Borg-Graham formalism gate',
        'mkm': 'iKm gate',
        'mca': 'iCa gate',
        'q': 'iK calcium dependent gate',
        'Cai': 'calcium intracellular concentration'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def alpham(cls, Vm):
        return 0.55 * (7.1 - Vm) / np.exp((7.1-Vm) / 4) * 1e3 # s-1

    @classmethod
    def betam(cls, Vm):
        return 0.48 * cls.vtrap((Vm - 46.1), 5) * 1e3  # s-1

    @classmethod
    def alphah(cls, Vm):
        return 0.22 * np.exp((23 - Vm) / 18) * 1e3  # s-1

    @classmethod
    def betah(cls, Vm):
        return 6.92 / (1 + np.exp((46 - Vm) / 5)) * 1e3 # s-1

    @classmethod
    def alphan(cls, Vm):
        return np.exp((-5e-3 * (Vm + 32) * 9.648e4) / 2562.35) * 1e3  # s-1

    @classmethod
    def betan(cls, Vm):
        return np.exp((-2e-3 * (Vm + 32) * 9.648e4) / 2562.35) * 1e3  # s-1

    @classmethod
    def alphal(cls, Vm):
        return np.exp((2e-3 * (Vm + 61) * 9.648e4) / 2562.35) * 1e3  # s-1

    @classmethod
    def betal(cls, Vm):
        return np.exp((-2e-3 * (Vm + 32) * 9.648e4) / 2562.35) * 1e3  # s-1

    @staticmethod
    def mkminf(Vm):
        return 1.0 / (1 + np.exp(-(Vm + 35) / 10))

    @classmethod
    def taumkm(cls, Vm):
        return 1.0 / (3.3 * (np.exp((Vm + 35) / 20) + np.exp(-(Vm + 35) / 20)) /3.54)  # s

    @classmethod
    def alphamca(cls, Vm):
        return 15.69 * cls.vtrap((81.5 - Vm), 5) * 1e3  # s-1

    @classmethod
    def betamca(cls, Vm):
        return 0.29 * np.exp(-Vm /10.86)  # s-1

    @classmethod
    def alphaq(cls, Cai):
        return 0.00246 / np.exp((12 * np.log10(np.power(Cai,3)) + 28.48)/ -4)  # s-1

    @classmethod
    def betaq(cls, Cai):
        return 0.006 / np.exp((12 * np.log10(np.power(Cai,3)) + 60.4) / 35)  # s-1


    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n'],
            'l': lambda Vm, x: cls.alphal(Vm) * (1 - x['l']) - cls.betal(Vm) * x['l'],
            'mkm': lambda Vm, x: (cls.mkminf(Vm) - x['mkm']) / cls.taumkm(Vm),
            'mca': lambda Vm, x: cls.alphamca(Vm) * (1 - x['mca']) - cls.betamca(Vm) * x['mca'],
            'q': lambda Vm, x: cls.alphaq(x['Cai']) * (1 - x['q']) - cls.betaq(x['Cai']) * x['q'],
            'Cai': lambda Vm, x: -0.026 * cls.iCa(x['mca'], Vm) - (x['Cai'] - Cai0) / 20.
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'n': lambda Vm: cls.alphan(Vm) / (cls.alphan(Vm) + cls.betan(Vm)),
            'l': lambda Vm: cls.alphal(Vm) / (cls.alphal(Vm) + cls.betal(Vm)),
            'mkm': lambda Vm: cls.mkminf(Vm),
            'mca': lambda Vm: cls.alphamca(Vm) / (cls.alphamca(Vm) + cls.betamca(Vm)),
            'q': lambda Vm: cls.alphaq(Vm, x['Cai']) / (cls.alphaq(Vm, x['Cai']) + cls.betaq(Vm, x['Cai'])),
            'Cai': lambda Vm: Cai0
        }

    # ------------------------------ Membrane currents ------------------------------

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current '''
        return cls.gNa_bar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKd(cls, n, l, Vm):
        ''' delayed-rectifier Potassium current '''
        return cls.gKd_bar * n**3 * l * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iKm(cls, mkm, Vm):
        ''' slowly activating Potassium current '''
        return cls.gKm_bar * mkm * (Vm - cls.EK)  # mA/m2

    @classmethod
    def ECa(cls, Cai):
        ''' Calcium reversal potential '''
        return np.log (2 / Cai) * 3.08 * 1e5 * R / (2 * F)  # mV

    @classmethod
    def iCa(cls, mca, Vm):
        ''' Calcium current '''
        return cls.gCa_bar * mca**2 * (Vm - cls.ECa)  # mA/m2

    @classmethod
    def iKCa(cls, q, Vm):
        ''' Calcium-dependent Potassium current '''
        return cls.gKCa_bar * q**2 * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: cls.iKd(x['n'], x['l'], Vm),
            'iKm': lambda Vm, x: cls.iKm(x['mkm'], Vm),
            'iCa': lambda Vm, x: cls.iCa(x['mca'], Vm),
            'iKCa': lambda Vm, x: cls.iKCa(x['q'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }
