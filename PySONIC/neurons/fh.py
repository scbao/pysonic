# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-01-07 18:41:06
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-12 23:04:57

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

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n', 'p')

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
        ''' Voltage-dependent activation rate of m-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.Vm0
        alpha = 0.36 * self.vtrap(22. - Vdiff, 3.)  # ms-1
        return self.q10 * alpha * 1e3  # s-1

    def betam(self, Vm):
        ''' Voltage-dependent inactivation rate of m-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.Vm0
        beta = 0.4 * self.vtrap(Vdiff - 13., 20.)  # ms-1
        return self.q10 * beta * 1e3  # s-1

    def alphah(self, Vm):
        ''' Voltage-dependent activation rate of h-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.Vm0
        alpha = 0.1 * self.vtrap(Vdiff + 10.0, 6.)  # ms-1
        return self.q10 * alpha * 1e3  # s-1

    def betah(self, Vm):
        ''' Voltage-dependent inactivation rate of h-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.Vm0
        beta = 4.5 / (np.exp((45. - Vdiff) / 10.) + 1)  # ms-1
        return self.q10 * beta * 1e3  # s-1

    def alphan(self, Vm):
        ''' Voltage-dependent activation rate of n-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.Vm0
        alpha = 0.02 * self.vtrap(35. - Vdiff, 10.0)  # ms-1
        return self.q10 * alpha * 1e3  # s-1

    def betan(self, Vm):
        ''' Voltage-dependent inactivation rate of n-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.Vm0
        beta = 0.05 * self.vtrap(Vdiff - 10., 10.)  # ms-1
        return self.q10 * beta * 1e3  # s-1

    def alphap(self, Vm):
        ''' Voltage-dependent activation rate of p-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.Vm0
        alpha = 0.006 * self.vtrap(40. - Vdiff, 10.0)  # ms-1
        return self.q10 * alpha * 1e3  # s-1

    def betap(self, Vm):
        ''' Voltage-dependent inactivation rate of p-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.Vm0
        beta = 0.09 * self.vtrap(Vdiff + 25., 20.)  # ms-1
        return self.q10 * beta * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    def derM(self, Vm, m):
        ''' Evolution of m-gate open-probability

            :param Vm: membrane potential (mV)
            :param m: open-probability of m-gate (-)
            :return: time derivative of m-gate open-probability (s-1)
        '''
        return self.alpham(Vm) * (1 - m) - self.betam(Vm) * m

    def derH(self, Vm, h):
        ''' Evolution of h-gate open-probability

            :param Vm: membrane potential (mV)
            :param h: open-probability of h-gate (-)
            :return: time derivative of h-gate open-probability (s-1)
        '''
        return self.alphah(Vm) * (1 - h) - self.betah(Vm) * h

    def derN(self, Vm, n):
        ''' Evolution of n-gate open-probability

            :param Vm: membrane potential (mV)
            :param n: open-probability of n-gate (-)
            :return: time derivative of n-gate open-probability (s-1)
        '''
        return self.alphan(Vm) * (1 - n) - self.betan(Vm) * n

    def derP(self, Vm, p):
        ''' Evolution of p-gate open-probability

            :param Vm: membrane potential (mV)
            :param p: open-probability of p-gate (-)
            :return: time derivative of p-gate open-probability (s-1)
        '''
        return self.alphap(Vm) * (1 - p) - self.betap(Vm) * p

    def derStates(self, Vm, states):
        m, h, n, p = states
        return {
            'm': self.derM(Vm, m),
            'h': self.derH(Vm, h),
            'n': self.derN(Vm, n),
            'p': self.derP(Vm, p)
        }

    def derEffStates(self, Qm, states, lkp):
        rates = self.interpEffRates(Qm, lkp)
        m, h, n, p = states
        return {
            'm': rates['alpham'] * (1 - m) - rates['betam'] * m,
            'h': rates['alphah'] * (1 - h) - rates['betah'] * h,
            'n': rates['alphan'] * (1 - n) - rates['betan'] * n,
            'p': rates['alphap'] * (1 - p) - rates['betap'] * p
        }

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
        # Solve the equation dx/dt = 0 at Vm for each x-state
        return {
            'm': self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            'p': self.alphap(Vm) / (self.alphap(Vm) + self.betap(Vm))
        }

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm):
        ''' Sodium current

            :param m: open-probability of m-gate
            :param h: open-probability of h-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        iNa_drive = self.ghkDrive(Vm, Z_Na, self.Nai, self.Nao, self.T)  # mC/m3
        return self.pNabar * m**2 * h * iNa_drive

    def iKd(self, n, Vm):
        ''' delayed-rectifier Potassium current

            :param n: open-probability of n-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        iKd_drive = self.ghkDrive(Vm, Z_K, self.Ki, self.Ko, self.T)  # mC/m3
        return self.pKbar * n**2 * iKd_drive

    def iP(self, p, Vm):
        ''' non-specific delayed current

            :param p: open-probability of p-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        iP_drive = self.ghkDrive(Vm, Z_Na, self.Nai, self.Nao, self.T)  # mC/m3
        return self.pPbar * p**2 * iP_drive

    def iLeak(self, Vm):
        ''' non-specific leakage current

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gLeak * (Vm - self.ELeak)

    def currents(self, Vm, states):
        m, h, n, p = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iKd': self.iKd(n, Vm),
            'iP': self.iP(p, Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        # Compute average cycle value for rate constants
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
