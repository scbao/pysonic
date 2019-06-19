# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-11 15:58:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-19 14:41:42

import numpy as np

from ..core import PointNeuron


class TemplateNeuron(PointNeuron):
    ''' Template neuron class '''

    # Neuron name
    name = 'template'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2   # Membrane capacitance (F/m2)
    Vm0 = -71.9  # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 50.0     # Sodium
    EK = -90.0     # Potassium
    ELeak = -70.3  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 560.0  # Sodium
    gKdbar = 60.0   # Delayed-rectifier Potassium
    gLeak = 0.205   # Non-specific leakage

    # Additional parameters
    VT = -56.2  # Spike threshold adjustment parameter (mV)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    def alpham(self, Vm):
        return 0.32 * self.vtrap(13 - (Vm - self.VT), 4) * 1e3  # s-1

    def betam(self, Vm):
        return 0.28 * self.vtrap((Vm - self.VT) - 40, 5) * 1e3  # s-1

    def alphah(self, Vm):
        return 0.128 * np.exp(-((Vm - self.VT) - 17) / 18) * 1e3  # s-1

    def betah(self, Vm):
        return 4 / (1 + np.exp(-((Vm - self.VT) - 40) / 5)) * 1e3  # s-1

    def alphan(self, Vm):
        return 0.032 * self.vtrap(15 - (Vm - self.VT), 5) * 1e3  # s-1

    def betan(self, Vm):
        return 0.5 * np.exp(-((Vm - self.VT) - 10) / 40) * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    def derStates(self):
        return {
            'm': lambda Vm, x: self.alpham(Vm) * (1 - x['m']) - self.betam(Vm) * x['m'],
            'h': lambda Vm, x: self.alphah(Vm) * (1 - x['h']) - self.betah(Vm) * x['h'],
            'n': lambda Vm, x: self.alphan(Vm) * (1 - x['n']) - self.betan(Vm) * x['n']
        }

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self):
        return {
            'm': lambda Vm: self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': lambda Vm: self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': lambda Vm: self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm))
        }

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm):
        ''' Sodium current '''
        return self.gNabar * m**3 * h * (Vm - self.ENa)  # mA/m2

    def iKd(self, n, Vm):
        ''' delayed-rectifier Potassium current '''
        return self.gKdbar * n**4 * (Vm - self.EK)  # mA/m2

    def iLeak(self, Vm):
        ''' non-specific leakage current '''
        return self.gLeak * (Vm - self.ELeak)  # mA/m2

    def currents(self):
        return {
            'iNa': lambda Vm, x: self.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: self.iKd(x['n'], Vm),
            'iLeak': lambda Vm, _: self.iLeak(Vm)
        }

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        return {
            'alpham': np.mean(self.alpham(Vm)),
            'betam': np.mean(self.betam(Vm)),
            'alphah': np.mean(self.alphah(Vm)),
            'betah': np.mean(self.betah(Vm)),
            'alphan': np.mean(self.alphan(Vm)),
            'betan': np.mean(self.betan(Vm))
        }
