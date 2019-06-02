# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-11-29 16:56:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-02 12:33:58


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

    name = 'STN'

    # Resting parameters
    Cm0 = 1e-2  # Cell membrane resting capacitance (F/m2)
    Vm0 = -58.0  # Resting membrane potential (mV)
    Cai0 = 5e-9  # M (5 nM)

    # Reversal potentials
    ENa = 60.0  # Sodium Nernst potential (mV)
    EK = -90.0  # Potassium Nernst potential (mV)

    # Physical constants
    T = 306.15  # K (33°C)

    # Calcium dynamics
    Cao = 2e-3  # M (2 mM)
    taur_Cai = 0.5e-3  # decay time constant for intracellular Ca2+ dissolution (s)

    # Leakage current
    gLeak = 3.5  # Conductance of non-specific leakage current (S/m^2)
    ELeak = -60.0  # Leakage reversal potential (mV)

    # Fast Na current
    gNabar = 490.0  # Max. conductance of Sodium current (S/m^2)
    thetax_m = -40  # mV
    thetax_h = -45.5  # mV
    kx_m = -8  # mV
    kx_h = 6.4  # mV
    tau0_m = 0.2 * 1e-3  # s
    tau1_m = 3 * 1e-3  # s
    tau0_h = 0 * 1e-3  # s
    tau1_h = 24.5 * 1e-3  # s
    thetaT_m = -53  # mV
    thetaT1_h = -50  # mV
    thetaT2_h = -50  # mV
    sigmaT_m = -0.7  # mV
    sigmaT1_h = -15  # mV
    sigmaT2_h = 16  # mV

    # Delayed rectifier K+ current
    gKdbar = 570.0  # Max. conductance of delayed-rectifier Potassium current (S/m^2)
    thetax_n = -41  # mV
    kx_n = -14  # mV
    tau0_n = 0 * 1e-3  # s
    tau1_n = 11 * 1e-3  # s
    thetaT1_n = -40  # mV
    thetaT2_n = -40  # mV
    sigmaT1_n = -40  # mV
    sigmaT2_n = 50  # mV

    # T-type Ca2+ current
    gCaTbar = 50.0  # Max. conductance of low-threshold Calcium current (S/m^2)
    thetax_p = -56  # mV
    thetax_q = -85  # mV
    kx_p = -6.7  # mV
    kx_q = 5.8  # mV
    tau0_p = 5 * 1e-3  # s
    tau1_p = 0.33 * 1e-3  # s
    tau0_q = 0 * 1e-3  # s
    tau1_q = 400 * 1e-3  # s
    thetaT1_p = -27  # mV
    thetaT2_p = -102  # mV
    thetaT1_q = -50  # mV
    thetaT2_q = -50  # mV
    sigmaT1_p = -10  # mV
    sigmaT2_p = 15  # mV
    sigmaT1_q = -15  # mV
    sigmaT2_q = 16  # mV

    # L-type Ca2+ current
    gCaLbar = 150.0  # Max. conductance of high-threshold Calcium current (S/m^2)
    thetax_c = -30.6  # mV
    thetax_d1 = -60  # mV
    thetax_d2 = 0.1 * 1e-6  # M
    kx_c = -5  # mV
    kx_d1 = 7.5  # mV
    kx_d2 = 0.02 * 1e-6  # M
    tau0_c = 45 * 1e-3  # s
    tau1_c = 10 * 1e-3  # s
    tau0_d1 = 400 * 1e-3  # s
    tau1_d1 = 500 * 1e-3  # s
    tau_d2 = 130 * 1e-3  # s
    thetaT1_c = -27  # mV
    thetaT2_c = -50  # mV
    thetaT1_d1 = -40  # mV
    thetaT2_d1 = -20  # mV
    sigmaT1_c = -20  # mV
    sigmaT2_c = 15  # mV
    sigmaT1_d1 = -15  # mV
    sigmaT2_d1 = 20  # mV

    # A-type K+ current
    gAbar = 50.0  # Max. conductance of A-type Potassium current (S/m^2)
    thetax_a = -45  # mV
    thetax_b = -90  # mV
    kx_a = -14.7  # mV
    kx_b = 7.5  # mV
    tau0_a = 1 * 1e-3  # s
    tau1_a = 1 * 1e-3  # s
    tau0_b = 0 * 1e-3  # s
    tau1_b = 200 * 1e-3  # s
    thetaT_a = -40  # mV
    thetaT1_b = -60  # mV
    thetaT2_b = -40  # mV
    sigmaT_a = -0.5  # mV
    sigmaT1_b = -30  # mV
    sigmaT2_b = 10  # mV

    # Ca2+-activated K+ current
    gKCabar = 10.0  # Max. conductance of Calcium-dependent Potassium current (S/m^2)
    thetax_r = 0.17 * 1e-6  # M
    kx_r = -0.08 * 1e-6  # M
    tau_r = 2 * 1e-3  # s

    def __init__(self):
        super().__init__()
        self.states = ['a', 'b', 'c', 'd1', 'd2', 'm', 'h', 'n', 'p', 'q', 'r', 'Cai']
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
        ''' Overriding default titration function. '''
        return self.isSilenced(*args, **kwargs)

    def getEffectiveDepth(self, Cai, Vm):
        ''' Compute effective depth that matches a given membrane potential
            and intracellular Calcium concentration.

            :return: effective depth (m)
        '''
        iCaT = self.iCaT(self.pinf(Vm), self.qinf(Vm), Vm, Cai)  # mA/m2
        iCaL = self.iCaL(self.cinf(Vm), self.d1inf(Vm), self.d2inf(Cai), Vm, Cai)  # mA/m2
        return -(iCaT + iCaL) / (Z_Ca * FARADAY * Cai / self.taur_Cai) * 1e-6  # m

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


    def derA(self, Vm, a):
        ''' Evolution of a-gate open-probability

            :param Vm: membrane potential (mV)
            :param a: open-probability of a-gate (-)
            :return: time derivative of a-gate open-probability (s-1)
        '''
        return (self.ainf(Vm) - a) / self.taua(Vm)


    def derB(self, Vm, b):
        ''' Evolution of b-gate open-probability

            :param Vm: membrane potential (mV)
            :param b: open-probability of b-gate (-)
            :return: time derivative of b-gate open-probability (s-1)
        '''
        return (self.binf(Vm) - b) / self.taub(Vm)


    def derC(self, Vm, c):
        ''' Evolution of c-gate open-probability

            :param Vm: membrane potential (mV)
            :param c: open-probability of c-gate (-)
            :return: time derivative of c-gate open-probability (s-1)
        '''
        return (self.cinf(Vm) - c) / self.tauc(Vm)


    def derD1(self, Vm, d1):
        ''' Evolution of d1-gate open-probability

            :param Vm: membrane potential (mV)
            :param d1: open-probability of d1-gate (-)
            :return: time derivative of d1-gate open-probability (s-1)
        '''
        return (self.d1inf(Vm) - d1) / self.taud1(Vm)


    def derD2(self, Cai, d2):
        ''' Evolution of Calcium-dependent d2-gate open-probability

            :param Vm: membrane potential (mV)
            :param d2: open-probability of d2-gate (-)
            :return: time derivative of d2-gate open-probability (s-1)
        '''
        return (self.d2inf(Cai) - d2) / self.tau_d2


    def derM(self, Vm, m):
        ''' Evolution of m-gate open-probability

            :param Vm: membrane potential (mV)
            :param m: open-probability of m-gate (-)
            :return: time derivative of m-gate open-probability (s-1)
        '''
        return (self.minf(Vm) - m) / self.taum(Vm)


    def derH(self, Vm, h):
        ''' Evolution of h-gate open-probability

            :param Vm: membrane potential (mV)
            :param h: open-probability of h-gate (-)
            :return: time derivative of h-gate open-probability (s-1)
        '''
        return (self.hinf(Vm) - h) / self.tauh(Vm)


    def derN(self, Vm, n):
        ''' Evolution of n-gate open-probability

            :param Vm: membrane potential (mV)
            :param n: open-probability of n-gate (-)
            :return: time derivative of n-gate open-probability (s-1)
        '''
        return (self.ninf(Vm) - n) / self.taun(Vm)


    def derP(self, Vm, p):
        ''' Evolution of p-gate open-probability

            :param Vm: membrane potential (mV)
            :param p: open-probability of p-gate (-)
            :return: time derivative of p-gate open-probability (s-1)
        '''
        return (self.pinf(Vm) - p) / self.taup(Vm)


    def derQ(self, Vm, q):
        ''' Evolution of q-gate open-probability

            :param Vm: membrane potential (mV)
            :param q: open-probability of q-gate (-)
            :return: time derivative of q-gate open-probability (s-1)
        '''
        return (self.qinf(Vm) - q) / self.tauq(Vm)

    def derR(self, Cai, r):
        ''' Evolution of Calcium-dependent r-gate open-probability

            :param Vm: membrane potential (mV)
            :param s: open-probability of r-gate (-)
            :return: time derivative of r-gate open-probability (s-1)
        '''
        return (self.rinf(Cai) - r) / self.tau_r


    def derCai(self, p, q, c, d1, d2, Cai, Vm):
        ''' Evolution of Calcium concentration in submembrane space.

            :param p: open-probability of p-gate
            :param q: open-probability of q-gate
            :param c: open-probability of c-gate
            :param d1: open-probability of d1-gate
            :param d2: open-probability of d2-gate
            :param Cai: Calcium concentration in submembranal space (M)
            :param Vm: membrane potential (mV)
            :return: time derivative of Calcium concentration in submembrane space (M/s)
        '''
        iCaT = self.iCaT(p, q, Vm, Cai)
        iCaL = self.iCaL(c, d1, d2, Vm, Cai)
        return - self.iCa_to_Cai_rate * (iCaT + iCaL) - Cai / self.taur_Cai


    def Caiinf(self, Vm, p, q, c, d1):
        ''' Find the steady-state intracellular Calcium concentration for a
            specific membrane potential and voltage-gated channel states.

            :param Vm: membrane potential (mV)
            :param p: open-probability of p-gate
            :param q: open-probability of q-gate
            :param c: open-probability of c-gate
            :param d1: open-probability of d1-gate
            :return: steady-state Calcium concentration in submembrane space (M)
        '''
        if isinstance(Vm, np.ndarray):
            return np.array([self.Caiinf(Vm[i], p[i], q[i], c[i], d1[i]) for i in range(Vm.size)])
        else:
            return brentq(
                lambda x: self.derCai(p, q, c, d1, self.d2inf(x), x, Vm),
                self.Cai0 * 1e-4, self.Cai0 * 1e3,
                xtol=1e-16
            )


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


    def iA(self, a, b, Vm):
        ''' A-type Potassium current

            :param a: open-probability of a-gate (-)
            :param b: open-probability of b-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gAbar * a**2 * b * (Vm - self.EK)


    def iCaT(self, p, q, Vm, Cai):
        ''' low-threshold (T-type) Calcium current

            :param p: open-probability of p-gate (-)
            :param q: open-probability of q-gate (-)
            :param Vm: membrane potential (mV)
            :param Cai: submembrane Calcium concentration (M)
            :return: current per unit area (mA/m2)
        '''
        return self.gCaTbar * p**2 * q * (Vm - self.nernst(Z_Ca, Cai, self.Cao, self.T))


    def iCaL(self, c, d1, d2, Vm, Cai):
        ''' high-threshold (L-type) Calcium current

            :param c: open-probability of c-gate (-)
            :param d1: open-probability of d1-gate (-)
            :param d2: open-probability of d2-gate (-)
            :param Vm: membrane potential (mV)
            :param Cai: submembrane Calcium concentration (M)
            :return: current per unit area (mA/m2)
        '''
        return self.gCaLbar * c**2 * d1 * d2 * (Vm - self.nernst(Z_Ca, Cai, self.Cao, self.T))


    def iKCa(self, r, Vm):
        ''' Calcium-activated Potassium current

            :param r: open-probability of r-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gKCabar * r**2 * (Vm - self.EK)


    def iLeak(self, Vm):
        ''' non-specific leakage current

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gLeak * (Vm - self.ELeak)


    def currents(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        a, b, c, d1, d2, m, h, n, p, q, r, Cai = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iKd': self.iKd(n, Vm),
            'iA': self.iA(a, b, Vm),
            'iCaT': self.iCaT(p, q, Vm, Cai),
            'iCaL': self.iCaL(c, d1, d2, Vm, Cai),
            'iKCa': self.iKCa(r, Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2


    def steadyStates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # voltage-gated steady states
        sstates = {
            'a': self.ainf(Vm),
            'b': self.binf(Vm),
            'c': self.cinf(Vm),
            'd1': self.d1inf(Vm),
            'm': self.minf(Vm),
            'h': self.hinf(Vm),
            'n': self.ninf(Vm),
            'p': self.pinf(Vm),
            'q': self.qinf(Vm)
        }
        sstates['Cai'] = self.Caiinf(Vm, sstates['p'], sstates['q'], sstates['c'], sstates['d1'])
        sstates['d2'] = self.d2inf(sstates['Cai'])
        sstates['r'] = self.rinf(sstates['Cai'])

        return sstates


    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        a, b, c, d1, d2, m, h, n, p, q, r, Cai = states

        return {
            'a': self.derA(Vm, a),
            'b': self.derB(Vm, b),
            'c': self.derC(Vm, c),
            'd1': self.derD1(Vm, d1),
            'd2': self.derD2(Cai, d2),
            'm': self.derM(Vm, m),
            'h': self.derH(Vm, h),
            'n': self.derN(Vm, n),
            'p': self.derP(Vm, p),
            'q': self.derQ(Vm, q),
            'r': self.derR(Cai, r),
            'Cai': self.derCai(p, q, c, d1, d2, Cai, Vm),
        }



    def computeEffRates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Compute average cycle value for rate constants
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


    def derEffStates(self, Qm, states, lkp):
        ''' Overriding of abstract parent method. '''

        rates = self.interpEffRates(Qm, lkp)
        Vmeff = self.interpVmeff(Qm, lkp)
        a, b, c, d1, d2, m, h, n, p, q, r, Cai = states

        return {
            'a': rates['alphaa'] * (1 - a) - rates['betaa'] * a,
            'b': rates['alphab'] * (1 - b) - rates['betab'] * b,
            'c': rates['alphac'] * (1 - c) - rates['betac'] * c,
            'd1': rates['alphad1'] * (1 - d1) - rates['betad1'] * d1,
            'd2': self.derD2(Cai, d2),
            'm': rates['alpham'] * (1 - m) - rates['betam'] * m,
            'h': rates['alphah'] * (1 - h) - rates['betah'] * h,
            'n': rates['alphan'] * (1 - n) - rates['betan'] * n,
            'p': rates['alphap'] * (1 - p) - rates['betap'] * p,
            'q': rates['alphaq'] * (1 - q) - rates['betaq'] * q,
            'r': self.derR(Cai, r),
            'Cai': self.derCai(p, q, c, d1, d2, Cai, Vmeff)
        }


    def quasiSteadyStates(self, lkp):
        ''' Overriding of abstract parent method. '''
        qsstates = self.qsStates(lkp, ['a', 'b', 'c', 'd1', 'm', 'h', 'n', 'p', 'q'])
        qsstates['Cai'] = self.Caiinf(lkp['V'], qsstates['p'], qsstates['q'], qsstates['c'],
                                      qsstates['d1'])
        qsstates['d2'] = self.d2inf(qsstates['Cai'])
        qsstates['r'] = self.rinf(qsstates['Cai'])
        return qsstates
