# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-01-07 18:41:06
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-19 14:44:34

import numpy as np
from ..core import PointNeuron
from ..constants import CELSIUS_2_KELVIN, Z_Na, Z_K


class FrankenhaeuserHuxley(PointNeuron):
    ''' Xenopus myelinated fiber node

        Reference:
        *Frankenhaeuser, B., and Huxley, A.F. (1964). The action potential in the myelinated nerve
        fibre of Xenopus laevis as computed on the basis of voltage clamp data.
        J Physiol 171, 302–315.*
    '''

    # Neuron name
    name = 'FH'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 2e-2  # Membrane capacitance (F/m2)
    Vm0 = -70.  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -69.974  # Leakage resting potential (mV)

    # Maximal channel conductances (S/m2)
    gLeak = 300.3  # Leakage conductance (S/m2)

    # Channel permeability constant (m/s)
    pNabar = 8e-5   # Sodium
    pKbar = 1.2e-5  # Potassium
    pPbar = .54e-5  # Non-specific

    # Ionic concentrations (M)
    Nai = 13.74e-3  # Intracellular Sodium
    Nao = 114.5e-3  # Extracellular Sodium
    Ki = 120e-3     # Intracellular Potassium
    Ko = 2.5e-3     # Extracellular Potassium

    # Additional parameters
    celsius = 20.0  # Temperature (Celsius)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'p': 'iP gate'
    }

    def __init__(self):
        super().__init__()
        self.q10 = 3**((self.celsius - 20) / 10)
        self.T = self.celsius + CELSIUS_2_KELVIN

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        pltvars = super().getPltVars(wrapleft, wrapright)
        pltvars['Qm']['bounds'] = (-150, 100)
        return pltvars

    # ------------------------------ Gating states kinetics ------------------------------

    def alpham(self, Vm):
        return self.q10 * 0.36 * self.vtrap(22. - (Vm - self.Vm0), 3.) * 1e3  # s-1

    def betam(self, Vm):
        return self.q10 * 0.4 * self.vtrap(Vm - self.Vm0 - 13., 20.) * 1e3  # s-1

    def alphah(self, Vm):
        return self.q10 * 0.1 * self.vtrap(Vm - self.Vm0 + 10.0, 6.) * 1e3  # s-1

    def betah(self, Vm):
        return self.q10 * 4.5 / (np.exp((45. - (Vm - self.Vm0)) / 10.) + 1) * 1e3  # s-1

    def alphan(self, Vm):
        return self.q10 * 0.02 * self.vtrap(35. - (Vm - self.Vm0), 10.0) * 1e3  # s-1

    def betan(self, Vm):
        return self.q10 * 0.05 * self.vtrap(Vm - self.Vm0 - 10., 10.) * 1e3  # s-1

    def alphap(self, Vm):
        return self.q10 * 0.006 * self.vtrap(40. - (Vm - self.Vm0), 10.0) * 1e3  # s-1

    def betap(self, Vm):
        return self.q10 * 0.09 * self.vtrap(Vm - self.Vm0 + 25., 20.) * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    def derStates(self):
        return {
            'm': lambda Vm, x: self.alpham(Vm) * (1 - x['m']) - self.betam(Vm) * x['m'],
            'h': lambda Vm, x: self.alphah(Vm) * (1 - x['h']) - self.betah(Vm) * x['h'],
            'n': lambda Vm, x: self.alphan(Vm) * (1 - x['n']) - self.betan(Vm) * x['n'],
            'p': lambda Vm, x: self.alphap(Vm) * (1 - x['p']) - self.betap(Vm) * x['p']
        }

    # def derEffStates(self, Vm, states, rates):
    #     return {
    #         'm': ratex['alpham'] * (1 - x[['m']) - ratex['betam'] * x[['m'],
    #         'h': ratex['alphah'] * (1 - x[['h']) - ratex['betah'] * x[['h'],
    #         'n': ratex['alphan'] * (1 - x[['n']) - ratex['betan'] * x[['n'],
    #         'p': ratex['alphap'] * (1 - x[['p']) - ratex['betap'] * x[['p']
    #     }

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self):
        return {
            'm': lambda Vm: self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': lambda Vm: self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': lambda Vm: self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            'p': lambda Vm: self.alphap(Vm) / (self.alphap(Vm) + self.betap(Vm))
        }

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm):
        ''' Sodium current '''
        iNa_drive = self.ghkDrive(Vm, Z_Na, self.Nai, self.Nao, self.T)  # mC/m3
        return self.pNabar * m**2 * h * iNa_drive  # mA/m2

    def iKd(self, n, Vm):
        ''' delayed-rectifier Potassium current '''
        iKd_drive = self.ghkDrive(Vm, Z_K, self.Ki, self.Ko, self.T)  # mC/m3
        return self.pKbar * n**2 * iKd_drive  # mA/m2

    def iP(self, p, Vm):
        ''' non-specific delayed current '''
        iP_drive = self.ghkDrive(Vm, Z_Na, self.Nai, self.Nao, self.T)  # mC/m3
        return self.pPbar * p**2 * iP_drive  # mA/m2

    def iLeak(self, Vm):
        ''' non-specific leakage current '''
        return self.gLeak * (Vm - self.ELeak)  # mA/m2

    def currents(self):
        return {
            'iNa': lambda Vm, x: self.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: self.iKd(x['n'], Vm),
            'iP': lambda Vm, x: self.iP(x['p'], Vm),
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
            'betan': np.mean(self.betan(Vm)),
            'alphap': np.mean(self.alphap(Vm)),
            'betap': np.mean(self.betap(Vm))
        }
