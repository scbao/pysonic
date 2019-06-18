# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-11 15:58:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-18 18:12:08

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

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n')

    # ------------------------------ Gating states kinetics ------------------------------

    def alpham(self, Vm):
        ''' Voltage-dependent activation rate of m-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        alpha = 0.32 * self.vtrap(13 - Vdiff, 4)  # ms-1
        return alpha * 1e3  # s-1

    def betam(self, Vm):
        ''' Voltage-dependent inactivation rate of m-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        beta = 0.28 * self.vtrap(Vdiff - 40, 5)  # ms-1
        return beta * 1e3  # s-1

    def alphah(self, Vm):
        ''' Voltage-dependent activation rate of h-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        alpha = (0.128 * np.exp(-(Vdiff - 17) / 18))  # ms-1
        return alpha * 1e3  # s-1

    def betah(self, Vm):
        ''' Voltage-dependent inactivation rate of h-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        beta = (4 / (1 + np.exp(-(Vdiff - 40) / 5)))  # ms-1
        return beta * 1e3  # s-1

    def alphan(self, Vm):
        ''' Voltage-dependent activation rate of n-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        alpha = 0.032 * self.vtrap(15 - Vdiff, 5)  # ms-1
        return alpha * 1e3  # s-1

    def betan(self, Vm):
        ''' Voltage-dependent inactivation rate of n-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        beta = (0.5 * np.exp(-(Vdiff - 10) / 40))  # ms-1
        return beta * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    def derStates(self, Vm, states):
        return {
            'm': self.alpham(Vm) * (1 - states['m']) - self.betam(Vm) * states['m'],
            'h': self.alphah(Vm) * (1 - states['h']) - self.betah(Vm) * states['h'],
            'n': self.alphan(Vm) * (1 - states['n']) - self.betan(Vm) * states['n']
        }

    def derEffStates(self, Vm, states, rates):
        return {
            'm': rates['alpham'] * (1 - states['m']) - rates['betam'] * states['m'],
            'h': rates['alphah'] * (1 - states['h']) - rates['betah'] * states['h'],
            'n': rates['alphan'] * (1 - states['n']) - rates['betan'] * states['n']
        }

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
        return {
            'm': self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm))
        }

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm):
        ''' Sodium current

            :param m: open-probability of m-gate (-)
            :param h: open-probability of h-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gNabar * m**3 * h * (Vm - self.ENa)

    def iKd(self, n, Vm):
        ''' delayed-rectifier Potassium current

            :param n: open-probability of n-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gKdbar * n**4 * (Vm - self.EK)

    def iLeak(self, Vm):
        ''' non-specific leakage current

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gLeak * (Vm - self.ELeak)

    def currents(self, Vm, states):
        return {
            'iNa': self.iNa(states['m'], states['h'], Vm),
            'iKd': self.iKd(states['n'], Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2

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
