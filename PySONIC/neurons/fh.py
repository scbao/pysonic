# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2019-01-07 18:41:06
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-16 15:24:02

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

    # Name of channel mechanism
    name = 'FH'

    # Cell biophysical parameters
    Cm0 = 2e-2  # Cell membrane resting capacitance (F/m2)
    Vm0 = -70.  # Membrane resting potential (mV)
    celsius = 20.0  # Temperature (Celsius)
    gLeak = 300.3  # Leakage conductance (S/m2)
    ELeak = -69.974  # Leakage resting potential (mV)
    pNabar = 8e-5  # Sodium permeability constant (m/s)
    pPbar = .54e-5  # Non-specific permeability constant (m/s)
    pKbar = 1.2e-5  # Potassium permeability constant (m/s)
    Nai = 13.74e-3  # Sodium intracellular concentration (M)
    Nao = 114.5e-3  # Sodium extracellular concentration (M)
    Ki = 120e-3  # Potassium intracellular concentration (M)
    Ko = 2.5e-3  # Potassium extracellular concentration (M)


    def __init__(self):
        self.states = ['m', 'h', 'n', 'p']
        self.rates = self.getRatesNames(self.states)
        self.q10 = 3**((self.celsius - 20) / 10)
        self.T = self.celsius + CELSIUS_2_KELVIN

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        pltvars = super().getPltVars(wrapleft, wrapright)
        pltvars['Qm']['bounds'] = (-150, 50)
        return pltvars


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
        ''' Overriding of abstract parent method. '''
        m, h, n, p = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iKd': self.iKd(n, Vm),
            'iP': self.iP(p, Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2


    def steadyStates(self, Vm):
        ''' Overriding of abstract parent method. '''
        # Solve the equation dx/dt = 0 at Vm for each x-state
        return {
            'm': self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            'p': self.alphap(Vm) / (self.alphap(Vm) + self.betap(Vm))
        }


    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, p = states
        return {
            'm': self.derM(Vm, m),
            'h': self.derH(Vm, h),
            'n': self.derN(Vm, n),
            'p': self.derP(Vm, p)
        }


    def computeEffRates(self, Vm):
        ''' Overriding of abstract parent method. '''

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


    def derEffStates(self, Qm, states, lkp):
        ''' Overriding of abstract parent method. '''

        rates = self.interpEffRates(Qm, lkp)
        m, h, n, p = states

        return {
            'm': rates['alpham'] * (1 - m) - rates['betam'] * m,
            'h': rates['alphah'] * (1 - h) - rates['betah'] * h,
            'n': rates['alphan'] * (1 - n) - rates['betan'] * n,
            'p': rates['alphap'] * (1 - p) - rates['betap'] * p
        }


    def quasiSteadyStates(self, lkp):
        ''' Overriding of abstract parent method. '''
        return self.qsStates(lkp, ['m', 'h', 'n', 'p'])
