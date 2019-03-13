# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-11-29 16:56:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-13 18:19:55


import numpy as np
from scipy.optimize import brentq
from ..core import PointNeuron
from ..constants import FARADAY, Z_Ca
from ..utils import nernst


class OtsukaSTN(PointNeuron):
    ''' Class defining the Otsuka model of sub-thalamic nucleus neuron
        with 5 different current types:
            - Inward Sodium current (iNa)
            - Outward, delayed-rectifer Potassium current (iKd)
            - Inward, A-type Potassium current (iA)
            - Inward, low-threshold Calcium current (iCaT)
            - Inward, high-threshold Calcium current (iCaL)
            - Outward, Calcium-dependent Potassium current (iKCa)
            - Non-specific leakage current (iLeak)

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
    taur_Cai = 1 / 2e3  # decay time constant for intracellular Ca2+ dissolution (s)

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
    gKbar = 570.0  # Max. conductance of delayed-rectifier Potassium current (S/m^2)
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
    gKCaMax = 10.0  # Max. conductance of Calcium-dependent Potassium current (S/m^2)
    thetax_r = 0.17 * 1e-6  # M
    kx_r = -0.08 * 1e-6  # M
    tau_r = 2 * 1e-3  # s

    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h'],
        'i_{Kd}\ kin.': ['n'],
        'i_A\ kin.': ['a', 'b'],
        'i_{CaT}\ kin.': ['p', 'q'],
        'i_{CaL}\ kin.': ['c', 'd1', 'd2'],
        'Ca^{2+}_i': ['Cai'],
        'i_{KCa}\ kin.': ['r'],
    }


    def __init__(self):
        self.states_names = ['a', 'b', 'c', 'd1', 'd2', 'm', 'h', 'n', 'p', 'q', 'r', 'Cai']
        self.coeff_names = [
            'alphaa', 'betaa',
            'alphab', 'betab',
            'alphac', 'betac',
            'alphad1', 'betad1',
            'alpham', 'betam',
            'alphah', 'betah',
            'alphan', 'betan',
            'alphap', 'betap',
            'alphaq', 'betaq',
        ]
        self.deff = self.getEffectiveDepth(self.Cai0, self.Vm0)  # m
        self.iCa_to_Cai_rate = self.currentToConcentrationRate(Z_Ca, self.deff)
        self.states0 = self.steadyStates(self.Vm0)


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
        return (self.ainf(Vm) - a) / self.taua(Vm)

    def derB(self, Vm, b):
        return (self.binf(Vm) - b) / self.taub(Vm)

    def derC(self, Vm, c):
        return (self.cinf(Vm) - c) / self.tauc(Vm)

    def derD1(self, Vm, d1):
        return (self.d1inf(Vm) - d1) / self.taud1(Vm)

    def derD2(self, Cai, d2):
        return (self.d2inf(Cai) - d2) / self.tau_d2

    def derM(self, Vm, m):
        return (self.minf(Vm) - m) / self.taum(Vm)

    def derH(self, Vm, h):
        return (self.hinf(Vm) - h) / self.tauh(Vm)

    def derN(self, Vm, n):
        return (self.ninf(Vm) - n) / self.taun(Vm)

    def derP(self, Vm, p):
        return (self.pinf(Vm) - p) / self.taup(Vm)

    def derQ(self, Vm, q):
        return (self.qinf(Vm) - q) / self.tauq(Vm)

    def derR(self, Cai, r):
        return (self.rinf(Cai) - r) / self.tau_r


    def derCai(self, p, q, c, d1, d2, Cai, Vm):
        ''' Compute the time derivative of the Calcium concentration in submembranal space.

            :param Vm: membrane potential (mV)
            :param Cai: Calcium concentration in submembranal space (M)
            :param p: open-probability of p-gate
            :param q: open-probability of q-gate
            :param c: open-probability of c-gate
            :param d1: open-probability of d1-gate
            :param d2: open-probability of d2-gate
            :return: time derivative of Calcium concentration in submembranal space (M/s)
        '''
        iCaT = self.iCaT(p, q, Vm, Cai)
        iCaL = self.iCaL(c, d1, d2, Vm, Cai)
        return - self.iCa_to_Cai_rate * (iCaT + iCaL) - Cai / self.taur_Cai


    def derCaiSteadyState(self, Cai, Vm):
        ''' Compute the time derivative of the Calcium concentration in submembranal space,
            assuming quasi-steady gating states.

            :param Vm: membrane potential (mV)
            :param Cai: Calcium concentration in submembranal space (M)
            :return: time derivative of Calcium concentration in submembranal space (M/s)
        '''
        return self.derCai(self.pinf(Vm), self.qinf(Vm), self.cinf(Vm), self.d1inf(Vm),
                           self.d2inf(Cai), Cai, Vm)


    def findCaiSteadyState(self, Vm):
        ''' Find the steady-state intracellular Calcium concentration for a
            specific membrane potential.

            :param Vm: membrane potential (mV)
            :return: steady-state Calcium concentration in submembranal space (M)
        '''
        return brentq(
            lambda x: self.derCaiSteadyState(x, Vm),
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
        ''' Delayed-rectifier Potassium current

            :param n: open-probability of n-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gKbar * n**4 * (Vm - self.EK)


    def iA(self, a, b, Vm):
        ''' A-type Potassium current

            :param a: open-probability of a-gate (-)
            :param b: open-probability of b-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gAbar * a**2 * b * (Vm - self.EK)


    def iCaT(self, p, q, Vm, Cai):
        ''' Low-threshold (T-type) Calcium current

            :param p: open-probability of p-gate (-)
            :param q: open-probability of q-gate (-)
            :param Vm: membrane potential (mV)
            :param Cai: submembrane Calcium concentration (M)
            :return: current per unit area (mA/m2)
        '''
        return self.gCaTbar * p**2 * q * (Vm - nernst(Z_Ca, Cai, self.Cao, self.T))


    def iCaL(self, c, d1, d2, Vm, Cai):
        ''' High-threshold (L-type) Calcium current

            :param c: open-probability of c-gate (-)
            :param d1: open-probability of d1-gate (-)
            :param d2: open-probability of d2-gate (-)
            :param Vm: membrane potential (mV)
            :param Cai: submembrane Calcium concentration (M)
            :return: current per unit area (mA/m2)
        '''
        return self.gCaLbar * c**2 * d1 * d2 * (Vm - nernst(Z_Ca, Cai, self.Cao, self.T))


    def iKCa(self, r, Vm):
        ''' Calcium-activated Potassium current

            :param r: open-probability of r-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gKCaMax * r**2 * (Vm - self.EK)


    def iLeak(self, Vm):
        ''' Non-specific leakage current

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
        aeq = self.ainf(Vm)
        beq = self.binf(Vm)
        ceq = self.cinf(Vm)
        d1eq = self.d1inf(Vm)
        meq = self.minf(Vm)
        heq = self.hinf(Vm)
        neq = self.ninf(Vm)
        peq = self.pinf(Vm)
        qeq = self.qinf(Vm)
        # Cai_eq = self.Cai0
        Cai_eq = self.findCaiSteadyState(Vm)
        d2eq = self.d2inf(Cai_eq)
        req = self.rinf(Cai_eq)

        return np.array([aeq, beq, ceq, d1eq, d2eq, meq, heq, neq, peq, qeq, req, Cai_eq])


    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        a, b, c, d1, d2, m, h, n, p, q, r, Cai = states

        dadt = self.derA(Vm, a)
        dbdt = self.derB(Vm, b)
        dcdt = self.derC(Vm, c)
        dd1dt = self.derD1(Vm, d1)
        dd2dt = self.derD2(Cai, d2)
        dmdt = self.derM(Vm, m)
        dhdt = self.derH(Vm, h)
        dndt = self.derN(Vm, n)
        dpdt = self.derP(Vm, p)
        dqdt = self.derQ(Vm, q)
        drdt = self.derR(Cai, r)
        dCaidt = self.derCai(p, q, c, d1, d2, Cai, Vm)

        return [dadt, dbdt, dcdt, dd1dt, dd2dt, dmdt, dhdt, dndt, dpdt, dqdt, drdt, dCaidt]


    def getEffRates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Compute average cycle value for rate constants
        Ta = self.taua(Vm)
        alphaa_avg = np.mean(self.ainf(Vm) / Ta)
        betaa_avg = np.mean(1 / Ta) - alphaa_avg

        Tb = self.taub(Vm)
        alphab_avg = np.mean(self.binf(Vm) / Tb)
        betab_avg = np.mean(1 / Tb) - alphab_avg

        Tc = self.tauc(Vm)
        alphac_avg = np.mean(self.cinf(Vm) / Tc)
        betac_avg = np.mean(1 / Tc) - alphac_avg

        Td1 = self.taud1(Vm)
        alphad1_avg = np.mean(self.d1inf(Vm) / Td1)
        betad1_avg = np.mean(1 / Td1) - alphad1_avg

        Tm = self.taum(Vm)
        alpham_avg = np.mean(self.minf(Vm) / Tm)
        betam_avg = np.mean(1 / Tm) - alpham_avg

        Th = self.tauh(Vm)
        alphah_avg = np.mean(self.hinf(Vm) / Th)
        betah_avg = np.mean(1 / Th) - alphah_avg

        Tn = self.taun(Vm)
        alphan_avg = np.mean(self.ninf(Vm) / Tn)
        betan_avg = np.mean(1 / Tn) - alphan_avg

        Tp = self.taup(Vm)
        alphap_avg = np.mean(self.pinf(Vm) / Tp)
        betap_avg = np.mean(1 / Tp) - alphap_avg

        Tq = self.tauq(Vm)
        alphaq_avg = np.mean(self.qinf(Vm) / Tq)
        betaq_avg = np.mean(1 / Tq) - alphaq_avg

        # Return array of coefficients
        return np.array([
            alphaa_avg, betaa_avg,
            alphab_avg, betab_avg,
            alphac_avg, betac_avg,
            alphad1_avg, betad1_avg,
            alpham_avg, betam_avg,
            alphah_avg, betah_avg,
            alphan_avg, betan_avg,
            alphap_avg, betap_avg,
            alphaq_avg, betaq_avg
        ])


    def derStatesEff(self, Qm, states, interp_data):
        ''' Overriding of abstract parent method. '''

        rates = np.array([np.interp(Qm, interp_data['Q'], interp_data[rn])
                          for rn in self.coeff_names])
        Vmeff = np.interp(Qm, interp_data['Q'], interp_data['V'])

        a, b, c, d1, d2, m, h, n, p, q, r, Cai = states

        dadt = rates[0] * (1 - a) - rates[1] * a
        dbdt = rates[2] * (1 - b) - rates[3] * b
        dcdt = rates[4] * (1 - c) - rates[5] * c
        dd1dt = rates[6] * (1 - d1) - rates[7] * d1
        dd2dt = self.derD2(Cai, d2)
        dmdt = rates[8] * (1 - m) - rates[9] * m
        dhdt = rates[10] * (1 - h) - rates[11] * h
        dndt = rates[12] * (1 - n) - rates[13] * n
        dpdt = rates[14] * (1 - p) - rates[15] * p
        dqdt = rates[16] * (1 - q) - rates[17] * q
        drdt = self.derR(Cai, r)
        dCaidt = self.derCai(p, q, c, d1, d2, Cai, Vmeff)

        return [dadt, dbdt, dcdt, dd1dt, dd2dt, dmdt, dhdt, dndt, dpdt, dqdt, drdt, dCaidt]
