# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2019-01-07 18:41:06
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-13 18:18:59

import numpy as np
from ..core import PointNeuron
from ..utils import vtrap, ghkDrive
from ..constants import CELSIUS_2_KELVIN, Z_Na, Z_K


class FrankenhaeuserHuxley(PointNeuron):
    ''' Class defining the membrane channel dynamics of a Xenopus myelinated neuron
        with 4 different current types:
            - Inward Sodium current
            - Outward, delayed-rectifier Potassium current
            - Non-specific delayed current
            - Non-specific leakage current

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

    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h'],
        'i_{Kd}\ kin.': ['n'],
        'i_P\ kin.': ['p']
    }


    def __init__(self):
        self.states_names = ['m', 'h', 'n', 'p']
        self.coeff_names = ['alpham', 'betam', 'alphah', 'betah', 'alphan', 'betan',
                            'alphap', 'betap']
        self.q10 = 3**((self.celsius - 20) / 10)
        self.T = self.celsius + CELSIUS_2_KELVIN
        self.states0 = self.steadyStates(self.Vm0)


    def alpham(self, Vm):
        ''' Compute the alpha rate for the open-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        Vdiff = Vm - self.Vm0
        alpha = 0.36 * vtrap(22. - Vdiff, 3.)  # ms-1
        return self.q10 * alpha * 1e3  # s-1


    def betam(self, Vm):
        ''' Compute the beta rate for the open-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        Vdiff = Vm - self.Vm0
        beta = 0.4 * vtrap(Vdiff - 13., 20.)  # ms-1
        return self.q10 * beta * 1e3  # s-1


    def alphah(self, Vm):
        ''' Compute the alpha rate for the inactivation-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        Vdiff = Vm - self.Vm0
        alpha = 0.1 * vtrap(Vdiff + 10.0, 6.)  # ms-1
        return self.q10 * alpha * 1e3  # s-1


    def betah(self, Vm):
        ''' Compute the beta rate for the inactivation-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        Vdiff = Vm - self.Vm0
        beta = 4.5 / (np.exp((45. - Vdiff) / 10.) + 1)  # ms-1
        return self.q10 * beta * 1e3  # s-1


    def alphan(self, Vm):
        ''' Compute the alpha rate for the open-probability of delayed-rectifier Potassium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        Vdiff = Vm - self.Vm0
        alpha = 0.02 * vtrap(35. - Vdiff, 10.0)  # ms-1
        return self.q10 * alpha * 1e3  # s-1


    def betan(self, Vm):
        ''' Compute the beta rate for the open-probability of delayed-rectifier Potassium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        Vdiff = Vm - self.Vm0
        beta = 0.05 * vtrap(Vdiff - 10., 10.)  # ms-1
        return self.q10 * beta * 1e3  # s-1


    def alphap(self, Vm):
        ''' Compute the alpha rate for the open-probability of non-specific delayed current.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        Vdiff = Vm - self.Vm0
        alpha = 0.006 * vtrap(40. - Vdiff, 10.0)  # ms-1
        return self.q10 * alpha * 1e3  # s-1


    def betap(self, Vm):
        ''' Compute the beta rate for the open-probability of non-specific delayed current.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        Vdiff = Vm - self.Vm0
        beta = 0.09 * vtrap(Vdiff + 25., 20.)  # ms-1
        return self.q10 * beta * 1e3  # s-1


    def derM(self, Vm, m):
        ''' Compute the evolution of the open-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :param m: open-probability of Sodium channels (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''
        return self.alpham(Vm) * (1 - m) - self.betam(Vm) * m


    def derH(self, Vm, h):
        ''' Compute the evolution of the inactivation-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :param h: inactivation-probability of Sodium channels (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''
        return self.alphah(Vm) * (1 - h) - self.betah(Vm) * h


    def derN(self, Vm, n):
        ''' Compute the evolution of the open-probability of delayed-rectifier Potassium channels.

            :param Vm: membrane potential (mV)
            :param n: open-probability of delayed-rectifier Potassium channels (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''
        return self.alphan(Vm) * (1 - n) - self.betan(Vm) * n


    def derP(self, Vm, p):
        ''' Compute the evolution of the open-probability of non-specific delayed current.

            :param Vm: membrane potential (mV)
            :param p: open-probability (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''
        return self.alphap(Vm) * (1 - p) - self.betap(Vm) * p


    def iNa(self, m, h, Vm):
        ''' Sodium current

            :param m: open-probability of m-gate
            :param h: open-probability of h-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        iNa_drive = ghkDrive(Vm, Z_Na, self.Nai, self.Nao, self.T)  # mC/m3
        return self.pNabar * m**2 * h * iNa_drive  # mA/m2


    def iKd(self, n, Vm):
        ''' Delayed-rectifier Potassium current

            :param n: open-probability of n-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        iKd_drive = ghkDrive(Vm, Z_K, self.Ki, self.Ko, self.T)  # mC/m3
        return self.pKbar * n**2 * iKd_drive  # mA/m2


    def iP(self, p, Vm):
        ''' Non-specific delayed current

            :param p: open-probability of p-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        iP_drive = ghkDrive(Vm, Z_Na, self.Nai, self.Nao, self.T)  # mC/m3
        return self.pPbar * p**2 * iP_drive  # mA/m2


    def iLeak(self, Vm):
        ''' Non-specific leakage current

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
        meq = self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm))
        heq = self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm))
        neq = self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm))
        peq = self.alphap(Vm) / (self.alphap(Vm) + self.betap(Vm))
        return np.array([meq, heq, neq, peq])


    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, p = states
        dmdt = self.derM(Vm, m)
        dhdt = self.derH(Vm, h)
        dndt = self.derN(Vm, n)
        dpdt = self.derP(Vm, p)
        return [dmdt, dhdt, dndt, dpdt]


    def getEffRates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Compute average cycle value for rate constants
        am_avg = np.mean(self.alpham(Vm))
        bm_avg = np.mean(self.betam(Vm))
        ah_avg = np.mean(self.alphah(Vm))
        bh_avg = np.mean(self.betah(Vm))
        an_avg = np.mean(self.alphan(Vm))
        bn_avg = np.mean(self.betan(Vm))
        ap_avg = np.mean(self.alphap(Vm))
        bp_avg = np.mean(self.betap(Vm))

        # Return array of coefficients
        return np.array([am_avg, bm_avg, ah_avg, bh_avg, an_avg, bn_avg, ap_avg, bp_avg])


    def derStatesEff(self, Qm, states, interp_data):
        ''' Overriding of abstract parent method. '''
        rates = np.array([np.interp(Qm, interp_data['Q'], interp_data[rn])
                          for rn in self.coeff_names])
        m, h, n, p = states
        dmdt = rates[0] * (1 - m) - rates[1] * m
        dhdt = rates[2] * (1 - h) - rates[3] * h
        dndt = rates[4] * (1 - n) - rates[5] * n
        dpdt = rates[6] * (1 - p) - rates[7] * p

        return [dmdt, dhdt, dndt, dpdt]
