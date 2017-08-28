#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-31 15:20:54
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-28 17:57:49

''' Channels mechanisms for leech ganglion neurons. '''

import logging
from functools import partialmethod
import numpy as np
from .base import BaseMech

# Get package logger
logger = logging.getLogger('PointNICE')


class LeechTouch(BaseMech):
    ''' Class defining the membrane channel dynamics of a leech touch sensory neuron.
        with 4 different current types:
            - Inward Sodium current
            - Outward Potassium current
            - Inward Calcium current
            - Non-specific leakage current
            - Calcium-dependent, outward Potassium current
            - Outward, Sodium pumping current

        Reference:
        *Cataldo, E., Brunelli, M., Byrne, J.H., Av-Ron, E., Cai, Y., and Baxter, D.A. (2005).
        Computational model of touch sensory cells (T Cells) of the leech: role of the
        afterhyperpolarization (AHP) in activity-dependent conduction failure.
        J Comput Neurosci 18, 5–24.*
    '''

    # Name of channel mechanism
    name = 'LeechT'

    # Cell-specific biophysical parameters
    Cm0 = 1e-2  # Cell membrane resting capacitance (F/m2)
    Vm0 = -53.0  # Cell membrane resting potential (mV)
    VNa = 45.0  # Sodium Nernst potential (mV)
    VK = -62.0  # Potassium Nernst potential (mV)
    VCa = 60.0  # Calcium Nernst potential (mV)
    VKCa = -62.0  # Calcium-dependent, Potassium current Nernst potential (mV)
    VL = -48.0  # Non-specific leakage Nernst potential (mV)
    VPumpNa = -300.0  # Sodium pump current reversal potential (mV)
    GNaMax = 3500.0  # Max. conductance of Sodium current (S/m^2)
    GKMax = 900.0  # Max. conductance of Potassium current (S/m^2)
    GCaMax = 20.0  # Max. conductance of Calcium current (S/m^2)
    GKCaMax = 236.0  # Max. conductance of Calcium-dependent Potassium current (S/m^2)
    GL = 1.0  # Conductance of non-specific leakage current (S/m^2)
    GPumpNa = 20.0  # Max. conductance of Sodium pump current (S/m^2)

    taum = 0.1e-3  # Sodium activation time constant (s)
    taus = 0.6e-3  # Calcium activation time constant (s)


    surface = 6434.0  # surface of cell assumed as a single soma (um2)
    curr_factor = 1e6  # 1/nA to 1/mA

    tau_Na_removal = 16.0  # Time constant for the removal of Sodium ions from the pool (s)
    tau_Ca_removal = 1.25  # Time constant for the removal of Calcium ions from the pool (s)

    tau_PumpNa_act = 0.1  # Time constant for the PumpNa current activation from Sodium ions (s)
    tau_KCa_act = 0.01  # Time constant for the KCa current activation from Calcium ions (s)

    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h'],
        'i_K\ kin.': ['n'],
        'i_{Ca}\ kin.': ['s'],
        'pools': ['C_Na_arb', 'C_Na_arb_activation', 'C_Ca_arb', 'C_Ca_arb_activation'],
        'I': ['iNa', 'iK', 'iCa', 'iKCa', 'iPumpNa', 'iL', 'iNet']
    }


    def __init__(self):
        ''' Constructor of the class. '''

        # Names and initial states of the channels state probabilities
        self.states_names = ['m', 'h', 'n', 's', 'C_Na', 'A_Na', 'C_Ca', 'A_Ca']
        self.states0 = np.array([])

        # Names of the channels effective coefficients
        self.coeff_names = ['alpham', 'betam', 'alphah', 'betah', 'alphan', 'betan',
                            'alphas', 'betas']

        self.K_Na = 0.016 * self.surface / self.curr_factor
        self.K_Ca = 0.1 * self.surface / self.curr_factor

        # Define initial channel probabilities (solving dx/dt = 0 at resting potential)
        self.states0 = self.steadyStates(self.Vm0)


    # ----------------- Generic -----------------

    def _xinf(self, Vm, halfmax, slope, power):
        ''' Generic function computing the steady-state activation/inactivation of a
            particular ion channel at a given voltage.

            :param Vm: membrane potential (mV)
            :param halfmax: half-(in)activation voltage (mV)
            :param slope: slope parameter of (in)activation function (mV)
            :param power: power exponent multiplying the exponential expression (integer)
            :return: steady-state (in)activation (-)
        '''

        return 1 / (1 + np.exp((Vm - halfmax) / slope))**power


    def _taux(self, Vm, halfmax, slope, tauMax, tauMin):
        ''' Generic function computing the voltage-dependent, activation/inactivation time constant
            of a particular ion channel at a given voltage.

            :param Vm: membrane potential (mV)
            :param halfmax: voltage at which (in)activation time constant is half-maximal (mV)
            :param slope: slope parameter of (in)activation time constant function (mV)
            :return: steady-state (in)activation (-)
        '''

        return (tauMax - tauMin) / (1 + np.exp((Vm - halfmax) / slope)) + tauMin


    def _derC_ion(self, Cion, Iion, Kion, tau):
        ''' Generic function computing the time derivative of the concentration
            of a specific ion in its intracellular pool.

            :param Cion: ion concentration in the pool (arbitrary unit)
            :param Iion: ionic current (mA/m2)
            :param Kion: scaling factor for current contribution to pool (arb. unit / nA???)
            :param tau: time constant for removal of ions from the pool (s)
            :return: variation of ionic concentration in the pool (arbitrary unit /s)
        '''

        return (Kion * (-Iion) - Cion) / tau



    def _derA_ion(self, Aion, Cion, tau):
        ''' Generic function computing the time derivative of the concentration and time
            dependent activation function, for a specific pool-dependent ionic current.

            :param Aion: concentration and time dependent activation function (arbitrary unit)
            :param Cion: ion concentration in the pool (arbitrary unit)
            :param tau: time constant for activation function variation (s)
            :return: variation of activation function (arbitrary unit / s)
        '''
        return (Cion - Aion) / tau


    # ------------------ Na -------------------


    minf = partialmethod(_xinf, halfmax=-35.0, slope=-5.0, power=1)
    hinf = partialmethod(_xinf, halfmax=-50.0, slope=9.0, power=2)
    tauh = partialmethod(_taux, halfmax=-36.0, slope=3.5, tauMax=14.0e-3, tauMin=0.2e-3)


    def derM(self, Vm, m):
        ''' Instantaneous derivative of Sodium activation. '''
        return (self.minf(Vm) - m) / self.taum  # s-1


    def derH(self, Vm, h):
        ''' Instantaneous derivative of Sodium inactivation. '''
        return (self.hinf(Vm) - h) / self.tauh(Vm)  # s-1


    # ------------------ K -------------------

    ninf = partialmethod(_xinf, halfmax=-22.0, slope=-9.0, power=1)
    taun = partialmethod(_taux, halfmax=-10.0, slope=10.0, tauMax=6.0e-3, tauMin=1.0e-3)


    def derN(self, Vm, n):
        ''' Instantaneous derivative of Potassium activation. '''
        return (self.ninf(Vm) - n) / self.taun(Vm)  # s-1


    # ------------------ Ca -------------------

    sinf = partialmethod(_xinf, halfmax=-10.0, slope=-2.8, power=1)


    def derS(self, Vm, s):
        ''' Instantaneous derivative of Calcium activation. '''
        return (self.sinf(Vm) - s) / self.taus  # s-1


    # ------------------ Pools -------------------


    def derC_Na(self, C_Na, I_Na):
        ''' Derivative of Sodium concentration in intracellular pool. '''
        return self._derC_ion(C_Na, I_Na, self.K_Na, self.tau_Na_removal)


    def derA_Na(self, A_Na, C_Na):
        ''' Derivative of Sodium pool-dependent activation function for iPumpNa. '''
        return self._derA_ion(A_Na, C_Na, self.tau_PumpNa_act)


    def derC_Ca(self, C_Ca, I_Ca):
        ''' Derivative of Calcium concentration in intracellular pool. '''
        return self._derC_ion(C_Ca, I_Ca, self.K_Ca, self.tau_Ca_removal)


    def derA_Ca(self, A_Ca, C_Ca):
        ''' Derivative of Calcium pool-dependent activation function for iKCa. '''
        return self._derA_ion(A_Ca, C_Ca, self.tau_KCa_act)


    # ------------------ Currents -------------------

    def currNa(self, m, h, Vm):
        ''' Sodium inward current. '''
        return self.GNaMax * m**3 * h * (Vm - self.VNa)


    def currK(self, n, Vm):
        ''' Potassium outward current. '''
        return self.GKMax * n**2 * (Vm - self.VK)


    def currCa(self, s, Vm):
        ''' Calcium inward current. '''
        return self.GCaMax * s * (Vm - self.VCa)


    def currKCa(self, A_Ca, Vm):
        ''' Calcium-activated Potassium outward current. '''
        return self.GKCaMax * A_Ca * (Vm - self.VKCa)


    def currPumpNa(self, A_Na, Vm):
        ''' Outward current mimicking the activity of the NaK-ATPase pump. '''
        return self.GPumpNa * A_Na * (Vm - self.VPumpNa)


    def currL(self, Vm):
        ''' Leakage current. '''
        return self.GL * (Vm - self.VL)


    def currNet(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        m, h, n, s, _, A_Na, _, A_Ca = states
        return (self.currNa(m, h, Vm) + self.currK(n, Vm) + self.currCa(s, Vm)
                + self.currL(Vm) + self.currPumpNa(A_Na, Vm) + self.currKCa(A_Ca, Vm))  # mA/m2


    def steadyStates(self, Vm):
        ''' Concrete implementation of the abstract API method. '''

        # Standard gating dynamics: Solve the equation dx/dt = 0 at Vm for each x-state
        meq = self.minf(Vm)
        heq = self.hinf(Vm)
        neq = self.ninf(Vm)
        seq = self.sinf(Vm)

        # PumpNa pool concentration and activation steady-state
        INa_eq = self.currNa(meq, heq, Vm)
        CNa_eq = self.K_Na * (-INa_eq)
        ANa_eq = CNa_eq
        # print('initial Na current: {:.2f} mA/m2'.format(INa_eq))
        # print('initial Na concentration in pool: {:.2f} arb. unit'.format(CNa_eq))

        # KCa current pool concentration and activation steady-state
        ICa_eq = self.currCa(seq, Vm)
        CCa_eq = self.K_Ca * (-ICa_eq)
        ACa_eq = CCa_eq
        # print('initial Ca current: {:.2f} mA/m2'.format(ICa_eq))
        # print('initial Ca concentration in pool: {:.2f} arb. unit'.format(CCa_eq))

        return np.array([meq, heq, neq, seq, CNa_eq, ANa_eq, CCa_eq, ACa_eq])


    def derStates(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        # Unpack states
        m, h, n, s, C_Na, A_Na, C_Ca, A_Ca = states

        # Standard gating states derivatives
        dmdt = self.derM(Vm, m)
        dhdt = self.derH(Vm, h)
        dndt = self.derN(Vm, n)
        dsdt = self.derS(Vm, s)

        # PumpNa current pool concentration and activation state
        I_Na = self.currNa(m, h, Vm)
        dCNa_dt = self.derC_Na(C_Na, I_Na)
        dANa_dt = self.derA_Na(A_Na, C_Na)

        # KCa current pool concentration and activation state
        I_Ca = self.currCa(s, Vm)
        dCCa_dt = self.derC_Ca(C_Ca, I_Ca)
        dACa_dt = self.derA_Ca(A_Ca, C_Ca)

        # Pack derivatives and return
        return [dmdt, dhdt, dndt, dsdt, dCNa_dt, dANa_dt, dCCa_dt, dACa_dt]


    def getEffRates(self, Vm):
        ''' Concrete implementation of the abstract API method. '''

        # Compute average cycle value for rate constants
        Tm = self.taum
        minf = self.minf(Vm)
        am_avg = np.mean(minf / Tm)
        bm_avg = np.mean(1 / Tm) - am_avg

        Th = self.tauh(Vm)
        hinf = self.hinf(Vm)
        ah_avg = np.mean(hinf / Th)
        bh_avg = np.mean(1 / Th) - ah_avg

        Tn = self.taun(Vm)
        ninf = self.ninf(Vm)
        an_avg = np.mean(ninf / Tn)
        bn_avg = np.mean(1 / Tn) - an_avg

        Ts = self.taus
        sinf = self.sinf(Vm)
        as_avg = np.mean(sinf / Ts)
        bs_avg = np.mean(1 / Ts) - as_avg

        # Return array of coefficients
        return np.array([am_avg, bm_avg, ah_avg, bh_avg, an_avg, bn_avg, as_avg, bs_avg])



    def derStatesEff(self, Qm, states, interp_data):
        ''' Concrete implementation of the abstract API method. '''

        rates = np.array([np.interp(Qm, interp_data['Q'], interp_data[rn])
                          for rn in self.coeff_names])
        Vmeff = np.interp(Qm, interp_data['Q'], interp_data['V'])

        # Unpack states
        m, h, n, s, C_Na, A_Na, C_Ca, A_Ca = states

        # Standard gating states derivatives
        dmdt = rates[0] * (1 - m) - rates[1] * m
        dhdt = rates[2] * (1 - h) - rates[3] * h
        dndt = rates[4] * (1 - n) - rates[5] * n
        dsdt = rates[6] * (1 - m) - rates[7] * s

        # PumpNa current pool concentration and activation state
        I_Na = self.currNa(m, h, Vmeff)
        dCNa_dt = self.derC_Na(C_Na, I_Na)
        dANa_dt = self.derA_Na(A_Na, C_Na)

        # KCa current pool concentration and activation state
        I_Ca_eff = self.currCa(s, Vmeff)
        dCCa_dt = self.derC_Ca(C_Ca, I_Ca_eff)
        dACa_dt = self.derA_Ca(A_Ca, C_Ca)

        # Pack derivatives and return
        return [dmdt, dhdt, dndt, dsdt, dCNa_dt, dANa_dt, dCCa_dt, dACa_dt]
