# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-11-29 16:56:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-20 10:07:34

import numpy as np
from scipy.optimize import brentq
from ..core import PointNeuron
from ..constants import FARADAY, Z_Ca


class OtsukaSTN(PointNeuron):
    ''' Sub-thalamic nucleus neuron

        References:
        *Otsuka, T., Abe, T., Tsukagawa, T., and Song, W.-J. (2004). Conductance-Based Model
        of the Voltage-Dependent Generation of a Plateau Potential in Subthalamic Neurons.
        Journal of Neurophysiology 92, 255–264.*

        *Tarnaud, T., Joseph, W., Martens, L., and Tanghe, E. (2018). Computational Modeling
        of Ultrasonic Subthalamic Nucleus Stimulation. IEEE Trans Biomed Eng.*
    '''

    # Neuron name
    name = 'STN'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2   # Membrane capacitance (F/m2)
    Vm0 = -58.0  # Membrane potential (mV)
    Cai0 = 5e-9  # Intracellular Calcium concentration (M)

    # Reversal potentials (mV)
    ENa = 60.0     # Sodium
    EK = -90.0     # Potassium
    ELeak = -60.0  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 490.0   # Sodium
    gLeak = 3.5      # Non-specific leakage
    gKdbar = 570.0   # Delayed-rectifier Potassium
    gCaTbar = 50.0   # Low-threshold Calcium
    gCaLbar = 150.0  # High-threshold Calcium
    gAbar = 50.0     # A-type Potassium
    gKCabar = 10.0   # Calcium-dependent Potassium

    # Physical constants
    T = 306.15  # K (33°C)

    # Calcium dynamics
    Cao = 2e-3         # extracellular Calcium concentration (M)
    taur_Cai = 0.5e-3  # decay time constant for intracellular Ca2+ dissolution (s)

    # Fast Na current m-gate
    thetax_m = -40   # mV
    kx_m = -8        # mV
    tau0_m = 0.2e-3  # s
    tau1_m = 3e-3    # s
    thetaT_m = -53   # mV
    sigmaT_m = -0.7  # mV

    # Fast Na current h-gate
    thetax_h = -45.5  # mV
    kx_h = 6.4        # mV
    tau0_h = 0e-3     # s
    tau1_h = 24.5e-3  # s
    thetaT1_h = -50   # mV
    thetaT2_h = -50   # mV
    sigmaT1_h = -15   # mV
    sigmaT2_h = 16    # mV

    # Delayed rectifier K+ current n-gate
    thetax_n = -41   # mV
    kx_n = -14       # mV
    tau0_n = 0e-3    # s
    tau1_n = 11e-3   # s
    thetaT1_n = -40  # mV
    thetaT2_n = -40  # mV
    sigmaT1_n = -40  # mV
    sigmaT2_n = 50   # mV

    # T-type Ca2+ current p-gate
    thetax_p = -56    # mV
    kx_p = -6.7       # mV
    tau0_p = 5e-3     # s
    tau1_p = 0.33e-3  # s
    thetaT1_p = -27   # mV
    thetaT2_p = -102  # mV
    sigmaT1_p = -10   # mV
    sigmaT2_p = 15    # mV

    # T-type Ca2+ current q-gate
    thetax_q = -85   # mV
    kx_q = 5.8       # mV
    tau0_q = 0e-3    # s
    tau1_q = 400e-3  # s
    thetaT1_q = -50  # mV
    thetaT2_q = -50  # mV
    sigmaT1_q = -15  # mV
    sigmaT2_q = 16   # mV

    # L-type Ca2+ current c-gate
    thetax_c = -30.6  # mV
    kx_c = -5         # mV
    tau0_c = 45e-3    # s
    tau1_c = 10e-3    # s
    thetaT1_c = -27   # mV
    thetaT2_c = -50   # mV
    sigmaT1_c = -20   # mV
    sigmaT2_c = 15    # mV

    # L-type Ca2+ current d1-gate
    thetax_d1 = -60   # mV
    kx_d1 = 7.5       # mV
    tau0_d1 = 400e-3  # s
    tau1_d1 = 500e-3  # s
    thetaT1_d1 = -40  # mV
    thetaT2_d1 = -20  # mV
    sigmaT1_d1 = -15  # mV
    sigmaT2_d1 = 20   # mV

    # L-type Ca2+ current d2-gate
    thetax_d2 = 0.1e-6  # M
    kx_d2 = 0.02e-6     # M
    tau_d2 = 130e-3     # s

    # A-type K+ current a-gate
    thetax_a = -45   # mV
    kx_a = -14.7     # mV
    tau0_a = 1e-3    # s
    tau1_a = 1e-3    # s
    thetaT_a = -40   # mV
    sigmaT_a = -0.5  # mV

    # A-type K+ current b-gate
    thetax_b = -90   # mV
    kx_b = 7.5       # mV
    tau0_b = 0e-3    # s
    tau1_b = 200e-3  # s
    thetaT1_b = -60  # mV
    thetaT2_b = -40  # mV
    sigmaT1_b = -30  # mV
    sigmaT2_b = 10   # mV

    # Ca2+-activated K+ current r-gate
    thetax_r = 0.17e-6  # M
    kx_r = -0.08e-6     # M
    tau_r = 2e-3        # s

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'a': 'iA activation gate',
        'b': 'iA inactivation gate',
        'p': 'iCaT activation gate',
        'q': 'iCaT inactivation gate',
        'c': 'iCaL activation gate',
        'd1': 'iCaL inactivation gate 1',
        'd2': 'iCaL inactivation gate 2',
        'r': 'iCaK gate',
        'Cai': 'submembrane Calcium concentration (M)'
    }

    def __init__(self):
        super().__init__()
        self.rates = self.getRatesNames(['a', 'b', 'c', 'd1', 'm', 'h', 'n', 'p', 'q'])
        self.deff = self.getEffectiveDepth(self.Cai0, self.Vm0)  # m
        self.iCa_to_Cai_rate = self.currentToConcentrationRate(Z_Ca, self.deff)  # Mmol.m-1.C-1

    def getPltScheme(self):
        pltscheme = super().getPltScheme()
        pltscheme['[Ca^{2+}]_i'] = ['Cai']
        return pltscheme

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        pltvars = super().getPltVars(wrapleft, wrapright)
        pltvars['Cai'] = {
            'desc': 'submembrane Ca2+ concentration',
            'label': '[Ca^{2+}]_i',
            'unit': 'uM',
            'factor': 1e6
        }
        return pltvars

    def titrationFunc(self, *args, **kwargs):
        return self.isSilenced(*args, **kwargs)

    def getEffectiveDepth(self, Cai, Vm):
        ''' Compute effective depth that matches a given membrane potential
            and intracellular Calcium concentration.

            :return: effective depth (m)
        '''
        iCaT = self.iCaT(self.pinf(Vm), self.qinf(Vm), Vm, Cai)  # mA/m2
        iCaL = self.iCaL(self.cinf(Vm), self.d1inf(Vm), self.d2inf(Cai), Vm, Cai)  # mA/m2
        return -(iCaT + iCaL) / (Z_Ca * FARADAY * Cai / self.taur_Cai) * 1e-6  # m

    # ------------------------------ Gating states kinetics ------------------------------

    def _xinf(self, var, theta, k):
        ''' Generic function computing the steady-state opening of a
            particular channel gate at a given voltage or ion concentration.

            :param var: membrane potential (mV) or ion concentration (mM)
            :param theta: half-(in)activation voltage or concentration (mV or mM)
            :param k: slope parameter of (in)activation function (mV or mM)
            :return: steady-state opening (-)
        '''
        return 1 / (1 + np.exp((var - theta) / k))

    def ainf(self, Vm):
        return self._xinf(Vm, self.thetax_a, self.kx_a)

    def binf(self, Vm):
        return self._xinf(Vm, self.thetax_b, self.kx_b)

    def cinf(self, Vm):
        return self._xinf(Vm, self.thetax_c, self.kx_c)

    def d1inf(self, Vm):
        return self._xinf(Vm, self.thetax_d1, self.kx_d1)

    def d2inf(self, Cai):
        return self._xinf(Cai, self.thetax_d2, self.kx_d2)

    def minf(self, Vm):
        return self._xinf(Vm, self.thetax_m, self.kx_m)

    def hinf(self, Vm):
        return self._xinf(Vm, self.thetax_h, self.kx_h)

    def ninf(self, Vm):
        return self._xinf(Vm, self.thetax_n, self.kx_n)

    def pinf(self, Vm):
        return self._xinf(Vm, self.thetax_p, self.kx_p)

    def qinf(self, Vm):
        return self._xinf(Vm, self.thetax_q, self.kx_q)

    def rinf(self, Cai):
        return self._xinf(Cai, self.thetax_r, self.kx_r)

    def _taux1(self, Vm, theta, sigma, tau0, tau1):
        ''' Generic function computing the voltage-dependent, activation/inactivation time constant
            of a particular ion channel at a given voltage (first variant).

            :param Vm: membrane potential (mV)
            :param theta: voltage at which (in)activation time constant is half-maximal (mV)
            :param sigma: slope parameter of (in)activation time constant function (mV)
            :param tau0: minimal time constant (s)
            :param tau1: modulated time constant (s)
            :return: (in)activation time constant (s)
        '''
        return tau0 + tau1 / (1 + np.exp(-(Vm - theta) / sigma))

    def taua(self, Vm):
        return self._taux1(Vm, self.thetaT_a, self.sigmaT_a, self.tau0_a, self.tau1_a)

    def taum(self, Vm):
        return self._taux1(Vm, self.thetaT_m, self.sigmaT_m, self.tau0_m, self.tau1_m)

    def _taux2(self, Vm, theta1, theta2, sigma1, sigma2, tau0, tau1):
        ''' Generic function computing the voltage-dependent, activation/inactivation time constant
            of a particular ion channel at a given voltage (second variant).

            :param Vm: membrane potential (mV)
            :param theta: voltage at which (in)activation time constant is half-maximal (mV)
            :param sigma: slope parameter of (in)activation time constant function (mV)
            :param tau0: minimal time constant (s)
            :param tau1: modulated time constant (s)
            :return: (in)activation time constant (s)
        '''
        return tau0 + tau1 / (np.exp(-(Vm - theta1) / sigma1) + np.exp(-(Vm - theta2) / sigma2))

    def taub(self, Vm):
        return self._taux2(Vm, self.thetaT1_b, self.thetaT2_b, self.sigmaT1_b, self.sigmaT2_b,
                           self.tau0_b, self.tau1_b)

    def tauc(self, Vm):
        return self._taux2(Vm, self.thetaT1_c, self.thetaT2_c, self.sigmaT1_c, self.sigmaT2_c,
                           self.tau0_c, self.tau1_c)

    def taud1(self, Vm):
        return self._taux2(Vm, self.thetaT1_d1, self.thetaT2_d1, self.sigmaT1_d1, self.sigmaT2_d1,
                           self.tau0_d1, self.tau1_d1)

    def tauh(self, Vm):
        return self._taux2(Vm, self.thetaT1_h, self.thetaT2_h, self.sigmaT1_h, self.sigmaT2_h,
                           self.tau0_h, self.tau1_h)

    def taun(self, Vm):
        return self._taux2(Vm, self.thetaT1_n, self.thetaT2_n, self.sigmaT1_n, self.sigmaT2_n,
                           self.tau0_n, self.tau1_n)

    def taup(self, Vm):
        return self._taux2(Vm, self.thetaT1_p, self.thetaT2_p, self.sigmaT1_p, self.sigmaT2_p,
                           self.tau0_p, self.tau1_p)

    def tauq(self, Vm):
        return self._taux2(Vm, self.thetaT1_q, self.thetaT2_q, self.sigmaT1_q, self.sigmaT2_q,
                           self.tau0_q, self.tau1_q)

    # ------------------------------ States derivatives ------------------------------

    def derCai(self, p, q, c, d1, d2, Cai, Vm):
        iCaT = self.iCaT(p, q, Vm, Cai)
        iCaL = self.iCaL(c, d1, d2, Vm, Cai)
        return - self.iCa_to_Cai_rate * (iCaT + iCaL) - Cai / self.taur_Cai  # M/s

    def derStates(self):
        return {
            'a': lambda Vm, x: (self.ainf(Vm) - x['a']) / self.taua(Vm),
            'b': lambda Vm, x: (self.binf(Vm) - x['b']) / self.taub(Vm),
            'c': lambda Vm, x: (self.cinf(Vm) - x['c']) / self.tauc(Vm),
            'd1': lambda Vm, x: (self.d1inf(Vm) - x['d1']) / self.taud1(Vm),
            'd2': lambda Vm, x: (self.d2inf(x['Cai']) - x['d2']) / self.tau_d2,
            'm': lambda Vm, x: (self.minf(Vm) - x['m']) / self.taum(Vm),
            'h': lambda Vm, x: (self.hinf(Vm) - x['h']) / self.tauh(Vm),
            'n': lambda Vm, x: (self.ninf(Vm) - x['n']) / self.taun(Vm),
            'p': lambda Vm, x: (self.pinf(Vm) - x['p']) / self.taup(Vm),
            'q': lambda Vm, x: (self.qinf(Vm) - x['q']) / self.tauq(Vm),
            'r': lambda Vm, x: (self.rinf(x['Cai']) - x['r']) / self.tau_r,
            'Cai': lambda Vm, x: self.derCai(x['p'], x['q'], x['c'], x['d1'], x['d2'], x['Cai'], Vm)
        }

    # def derEffStates(self):
    #     return {
    #         'a': lambda Vm, x, rates: rates['alphaa'] * (1 - x['a']) - rates['betaa'] * x['a'],
    #         'b': lambda Vm, x, rates: rates['alphab'] * (1 - x['b']) - rates['betab'] * x['b'],
    #         'c': lambda Vm, x, rates: rates['alphac'] * (1 - x['c']) - rates['betac'] * x['c'],
    #         'd1': lambda Vm, x, rates: rates['alphad1'] * (1 - x['d1']) - rates['betad1'] * x['d1'],
    #         'd2': lambda Vm, x, rates: (self.d2inf(x['Cai']) - x['d2']) / self.tau_d2,
    #         'm': lambda Vm, x, rates: rates['alpham'] * (1 - x['m']) - rates['betam'] * x['m'],
    #         'h': lambda Vm, x, rates: rates['alphah'] * (1 - x['h']) - rates['betah'] * x['h'],
    #         'n': lambda Vm, x, rates: rates['alphan'] * (1 - x['n']) - rates['betan'] * x['n'],
    #         'p': lambda Vm, x, rates: rates['alphap'] * (1 - x['p']) - rates['betap'] * x['p'],
    #         'q': lambda Vm, x, rates: rates['alphaq'] * (1 - x['q']) - rates['betaq'] * x['q'],
    #         'r': lambda Vm, x, rates: (self.rinf(x['Cai']) - x['r']) / self.tau_r,
    #         'Cai': lambda Vm, x, rates: self.derCai(x['p'], x['q'], x['c'], x['d1'], x['d2'], x['Cai'], Vm)
    #     }

    # def derEffStates(self, Vm, states, rates):
    #     return {
    #         'a': rates['alphaa'] * (1 - states['a']) - rates['betaa'] * states['a'],
    #         'b': rates['alphab'] * (1 - states['b']) - rates['betab'] * states['b'],
    #         'c': rates['alphac'] * (1 - states['c']) - rates['betac'] * states['c'],
    #         'd1': rates['alphad1'] * (1 - states['d1']) - rates['betad1'] * states['d1'],
    #         'd2': (self.d2inf(states['Cai']) - states['d2']) / self.tau_d2,
    #         'm': rates['alpham'] * (1 - states['m']) - rates['betam'] * states['m'],
    #         'h': rates['alphah'] * (1 - states['h']) - rates['betah'] * states['h'],
    #         'n': rates['alphan'] * (1 - states['n']) - rates['betan'] * states['n'],
    #         'p': rates['alphap'] * (1 - states['p']) - rates['betap'] * states['p'],
    #         'q': rates['alphaq'] * (1 - states['q']) - rates['betaq'] * states['q'],
    #         'r': (self.rinf(states['Cai']) - states['r']) / self.tau_r,
    #         'Cai': self.derCai(*[states[x] for x in ['p', 'q', 'c', 'd1', 'd2', 'Cai']], Vm)
    #     }

    # ------------------------------ Steady states ------------------------------

    def Caiinf(self, Vm):
        ''' Steady-state intracellular Calcium concentration '''
        if isinstance(Vm, np.ndarray):
            return np.array([self.Caiinf(v) for v in Vm])
        else:
            return brentq(
                lambda x: self.derCai(self.pinf(Vm), self.qinf(Vm), self.cinf(Vm),
                                      self.d1inf(Vm), self.d2inf(x), x, Vm),
                self.Cai0 * 1e-4, self.Cai0 * 1e3,
                xtol=1e-16
            )

    def steadyStates(self):
        return {
            'a': lambda Vm: self.ainf(Vm),
            'b': lambda Vm: self.binf(Vm),
            'c': lambda Vm: self.cinf(Vm),
            'd1': lambda Vm: self.d1inf(Vm),
            'd2': lambda Vm: self.d2inf(self.Caiinf(Vm)),
            'm': lambda Vm: self.minf(Vm),
            'h': lambda Vm: self.hinf(Vm),
            'n': lambda Vm: self.ninf(Vm),
            'p': lambda Vm: self.pinf(Vm),
            'q': lambda Vm: self.qinf(Vm),
            'r': lambda Vm: self.rinf(self.Caiinf(Vm)),
            'Cai': lambda Vm: self.Caiinf(Vm)
        }

    # def quasiSteadyStates(self, lkp):
    #     qsstates = self.qsStates(lkp, ['a', 'b', 'c', 'd1', 'm', 'h', 'n', 'p', 'q'])
    #     qsstates['Cai'] = self.Caiinf(lkp['V'], qsstates['p'], qsstates['q'], qsstates['c'],
    #                                   qsstates['d1'])
    #     qsstates['d2'] = self.d2inf(qsstates['Cai'])
    #     qsstates['r'] = self.rinf(qsstates['Cai'])
    #     return qsstates

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm):
        ''' Sodium current '''
        return self.gNabar * m**3 * h * (Vm - self.ENa)  # mA/m2

    def iKd(self, n, Vm):
        ''' delayed-rectifier Potassium current '''
        return self.gKdbar * n**4 * (Vm - self.EK)  # mA/m2

    def iA(self, a, b, Vm):
        ''' A-type Potassium current '''
        return self.gAbar * a**2 * b * (Vm - self.EK)  # mA/m2

    def iCaT(self, p, q, Vm, Cai):
        ''' low-threshold (T-type) Calcium current '''
        return self.gCaTbar * p**2 * q * (Vm - self.nernst(Z_Ca, Cai, self.Cao, self.T))  # mA/m2

    def iCaL(self, c, d1, d2, Vm, Cai):
        ''' high-threshold (L-type) Calcium current '''
        return self.gCaLbar * c**2 * d1 * d2 * (
            Vm - self.nernst(Z_Ca, Cai, self.Cao, self.T))  # mA/m2

    def iKCa(self, r, Vm):
        ''' Calcium-activated Potassium current '''
        return self.gKCabar * r**2 * (Vm - self.EK)  # mA/m2

    def iLeak(self, Vm):
        ''' non-specific leakage current '''
        return self.gLeak * (Vm - self.ELeak)  # mA/m2

    def currents(self):
        return {
            'iNa': lambda Vm, x: self.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: self.iKd(x['n'], Vm),
            'iA': lambda Vm, x: self.iA(x['a'], x['b'], Vm),
            'iCaT': lambda Vm, x: self.iCaT(x['p'], x['q'], Vm, x['Cai']),
            'iCaL': lambda Vm, x: self.iCaL(
                x['c'], x['d1'], x['d2'], Vm, x['Cai']),
            'iKCa': lambda Vm, x: self.iKCa(x['r'], Vm),
            'iLeak': lambda Vm, _: self.iLeak(Vm)
        }

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        return {
            'alphaa': np.mean(self.ainf(Vm) / self.taua(Vm)),
            'betaa': np.mean((1 - self.ainf(Vm)) / self.taua(Vm)),
            'alphab': np.mean(self.binf(Vm) / self.taub(Vm)),
            'betab': np.mean((1 - self.binf(Vm)) / self.taub(Vm)),
            'alphac': np.mean(self.cinf(Vm) / self.tauc(Vm)),
            'betac': np.mean((1 - self.cinf(Vm)) / self.tauc(Vm)),
            'alphad1': np.mean(self.d1inf(Vm) / self.taud1(Vm)),
            'betad1': np.mean((1 - self.d1inf(Vm)) / self.taud1(Vm)),
            'alpham': np.mean(self.minf(Vm) / self.taum(Vm)),
            'betam': np.mean((1 - self.minf(Vm)) / self.taum(Vm)),
            'alphah': np.mean(self.hinf(Vm) / self.tauh(Vm)),
            'betah': np.mean((1 - self.hinf(Vm)) / self.tauh(Vm)),
            'alphan': np.mean(self.ninf(Vm) / self.taun(Vm)),
            'betan': np.mean((1 - self.ninf(Vm)) / self.taun(Vm)),
            'alphap': np.mean(self.pinf(Vm) / self.taup(Vm)),
            'betap': np.mean((1 - self.pinf(Vm)) / self.taup(Vm)),
            'alphaq': np.mean(self.qinf(Vm) / self.tauq(Vm)),
            'betaq': np.mean((1 - self.qinf(Vm)) / self.tauq(Vm))
        }

    def getLowIntensities(self):
        ''' Return an array of acoustic intensities (W/m2) used to study the STN neuron in
            Tarnaud, T., Joseph, W., Martens, L., and Tanghe, E. (2018). Computational Modeling
            of Ultrasonic Subthalamic Nucleus Stimulation. IEEE Trans Biomed Eng.
        '''
        return np.hstack((
            np.arange(10, 101, 10),
            np.arange(101, 131, 1),
            np.array([140])
        ))  # W/m2
