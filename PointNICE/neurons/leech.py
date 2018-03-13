#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-31 15:20:54
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-13 15:04:00

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
    Vm0 = -53.58  # Cell membrane resting potential (mV)
    VNa = 45.0  # Sodium Nernst potential (mV)
    VK = -62.0  # Potassium Nernst potential (mV)
    VCa = 60.0  # Calcium Nernst potential (mV)
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

    # Original conversion constants from inward ion current (nA) to build-up of
    # intracellular ion concentration (arb.)
    K_Na_original = 0.016  # iNa to intracellular [Na+]
    K_Ca_original = 0.1  # iCa to intracellular [Ca2+]

    # Constants needed to convert K from original model (soma compartment)
    # to current model (point-neuron)
    surface = 6434.0e-12  # surface of cell assumed as a single soma (m2)
    curr_factor = 1e6  # mA to nA

    # Time constants for the removal of ions from intracellular pools (s)
    tau_Na_removal = 16.0  # Na+ removal
    tau_Ca_removal = 1.25  # Ca2+ removal

    # Time constants for the iPumpNa and iKCa currents activation
    # from specific intracellular ions (s)
    tau_PumpNa_act = 0.1  # iPumpNa activation from intracellular Na+
    tau_KCa_act = 0.01  # iKCa activation from intracellular Ca2+

    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h', 'm3h'],
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

        self.K_Na = self.K_Na_original * self.surface * self.curr_factor
        self.K_Ca = self.K_Ca_original * self.surface * self.curr_factor

        # Define initial channel probabilities (solving dx/dt = 0 at resting potential)
        self.states0 = self.steadyStates(self.Vm0)

        # Charge interval bounds for lookup creation
        self.Qbounds = (np.round(self.Vm0 - 10.0) * 1e-5, 50.0e-5)


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
        return self.GKCaMax * A_Ca * (Vm - self.VK)


    def currPumpNa(self, A_Na, Vm):
        ''' Outward current mimicking the activity of the NaK-ATPase pump. '''
        return self.GPumpNa * A_Na * (Vm - self.VPumpNa)


    def currL(self, Vm):
        ''' Leakage current. '''
        return self.GL * (Vm - self.VL)


    def currNet(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        m, h, n, s, _, A_Na, _, A_Ca = states
        return (self.currNa(m, h, Vm) + self.currK(n, Vm) + self.currCa(s, Vm) +
                self.currL(Vm) + self.currPumpNa(A_Na, Vm) + self.currKCa(A_Ca, Vm))  # mA/m2


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

        # KCa current pool concentration and activation steady-state
        ICa_eq = self.currCa(seq, Vm)
        CCa_eq = self.K_Ca * (-ICa_eq)
        ACa_eq = CCa_eq

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
        dsdt = rates[6] * (1 - s) - rates[7] * s

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





class LeechMech(BaseMech):
    ''' Class defining the basic dynamics of Sodium, Potassium and Calcium channels for several
        neurons of the leech.

        Reference:
        *Baccus, S.A. (1998). Synaptic facilitation by reflected action potentials: enhancement
        of transmission when nerve impulses reverse direction at axon branch points. Proc. Natl.
        Acad. Sci. U.S.A. 95, 8345–8350.*
    '''

    alphaC_sf = 1e-5  # Calcium activation rate constant scaling factor (M)
    betaC = 0.1e3  # beta rate for the open-probability of Ca2+-dependent Potassium channels (s-1)

    T = 293.15  # Room temperature (K)
    Rg = 8.314  # Universal gas constant (J.mol^-1.K^-1)
    Faraday = 9.6485e4  # Faraday constant (C/mol)


    def alpham(self, Vm):
        ''' Compute the alpha rate for the open-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        alpha = -0.03 * (Vm + 28) / (np.exp(- (Vm + 28) / 15) - 1)  # ms-1
        return alpha * 1e3  # s-1


    def betam(self, Vm):
        ''' Compute the beta rate for the open-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        beta = 2.7 * np.exp(-(Vm + 53) / 18)  # ms-1
        return beta * 1e3  # s-1


    def alphah(self, Vm):
        ''' Compute the alpha rate for the inactivation-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        alpha = 0.045 * np.exp(-(Vm + 58) / 18)  # ms-1
        return alpha * 1e3  # s-1


    def betah(self, Vm):
        ''' Compute the beta rate for the inactivation-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)

            .. warning:: the original paper contains an error (multiplication) in the
            expression of this rate constant, corrected in the mod file on ModelDB (division).
        '''

        beta = 0.72 / (np.exp(-(Vm + 23) / 14) + 1)  # ms-1
        return beta * 1e3  # s-1


    def alphan(self, Vm):
        ''' Compute the alpha rate for the open-probability of delayed-rectifier Potassium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        alpha = -0.024 * (Vm - 17) / (np.exp(-(Vm - 17) / 8) - 1)  # ms-1
        return alpha * 1e3  # s-1


    def betan(self, Vm):
        ''' Compute the beta rate for the open-probability of delayed-rectifier Potassium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        beta = 0.2 * np.exp(-(Vm + 48) / 35)  # ms-1
        return beta * 1e3  # s-1


    def alphas(self, Vm):
        ''' Compute the alpha rate for the open-probability of Calcium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        alpha = -1.5 * (Vm - 20) / (np.exp(-(Vm - 20) / 5) - 1)  # ms-1
        return alpha * 1e3  # s-1


    def betas(self, Vm):
        ''' Compute the beta rate for the open-probability of Calcium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        beta = 1.5 * np.exp(-(Vm + 25) / 10)  # ms-1
        return beta * 1e3  # s-1


    def alphaC(self, C_Ca_in):
        ''' Compute the alpha rate for the open-probability of Calcium-dependent Potassium channels.

            :param C_Ca_in: intracellular Calcium concentration (M)
            :return: rate constant (s-1)
        '''

        alpha = 0.1 * C_Ca_in / self.alphaC_sf  # ms-1
        return alpha * 1e3  # s-1


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


    def derS(self, Vm, s):
        ''' Compute the evolution of the open-probability of Calcium channels.

            :param Vm: membrane potential (mV)
            :param s: open-probability of Calcium channels (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''

        return self.alphas(Vm) * (1 - s) - self.betas(Vm) * s


    def derC(self, c, C_Ca_in):
        ''' Compute the evolution of the open-probability of Calcium-dependent Potassium channels.

            :param c: open-probability of Calcium-dependent Potassium channels (prob)
            :param C_Ca_in: intracellular Calcium concentration (M)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''

        return self.alphaC(C_Ca_in) * (1 - c) - self.betaC * c


    def currNa(self, m, h, Vm, C_Na_in):
        ''' Compute the inward Sodium current per unit area.

            :param m: open-probability of Sodium channels
            :param h: inactivation-probability of Sodium channels
            :param Vm: membrane potential (mV)
            :param C_Na_in: intracellular Sodium concentration (M)
            :return: current per unit area (mA/m2)
        '''

        GNa = self.GNaMax * m**4 * h
        VNa = self.nernst(self.Z_Na, C_Na_in, self.C_Na_out)  # Sodium Nernst potential
        return GNa * (Vm - VNa)


    def currK(self, n, Vm):
        ''' Compute the outward, delayed-rectifier Potassium current per unit area.

            :param n: open-probability of delayed-rectifier Potassium channels
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        GK = self.GKMax * n**2
        return GK * (Vm - self.VK)


    def currCa(self, s, Vm, C_Ca_in):
        ''' Compute the inward Calcium current per unit area.

            :param s: open-probability of Calcium channels
            :param Vm: membrane potential (mV)
            :param C_Ca_in: intracellular Calcium concentration (M)
            :return: current per unit area (mA/m2)
        '''

        GCa = self.GCaMax * s
        VCa = self.nernst(self.Z_Ca, C_Ca_in, self.C_Ca_out)  # Calcium Nernst potential
        return GCa * (Vm - VCa)


    def currKCa(self, c, Vm):
        ''' Compute the outward Calcium-dependent Potassium current per unit area.

            :param c: open-probability of Calcium-dependent Potassium channels
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        GKCa = self.GKCaMax * c
        return GKCa * (Vm - self.VK)


    def currL(self, Vm):
        ''' Compute the non-specific leakage current per unit area.

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        return self.GL * (Vm - self.VL)


class LeechPressure(LeechMech):
    ''' Class defining the membrane channel dynamics of a leech pressure sensory neuron.
        with 7 different current types:
            - Inward Sodium current
            - Outward Potassium current
            - Inward high-voltage-activated Calcium current
            - Non-specific leakage current
            - Calcium-dependent, outward Potassium current
            - Sodium pump current
            - Calcium pump current

        Reference:
        *Baccus, S.A. (1998). Synaptic facilitation by reflected action potentials: enhancement
        of transmission when nerve impulses reverse direction at axon branch points. Proc. Natl.
        Acad. Sci. U.S.A. 95, 8345–8350.*
    '''

    # Name of channel mechanism
    name = 'LeechP'

    # Cell-specific biophysical parameters
    Cm0 = 1e-2  # Cell membrane resting capacitance (F/m2)
    Vm0 = -48.865  # Cell membrane resting potential (mV)

    C_Na_out = 0.11  # Sodium extracellular concentration (M)
    C_Ca_out = 1.8e-3  # Calcium extracellular concentration (M)
    C_Na_in0 = 0.01  # Initial Sodium intracellular concentration (M)
    C_Ca_in0 = 1e-7  # Initial Calcium intracellular concentration (M)

    # VNa = 60  # Sodium Nernst potential, from MOD file on ModelDB (mV)
    # VCa = 125  # Calcium Nernst potential, from MOD file on ModelDB (mV)
    VK = -68.0  # Potassium Nernst potential (mV)
    VL = -49.0  # Non-specific leakage Nernst potential (mV)
    INaPmax = 70.0  # Maximum pump rate of the NaK-ATPase (mA/m2)
    khalf_Na = 0.012  # Sodium concentration at which NaK-ATPase is at half its maximum rate (M)
    ksteep_Na = 1e-3  # Sensitivity of NaK-ATPase to varying Sodium concentrations (M)
    iCaS = 0.1  # Calcium pump current parameter (mA/m2)
    GNaMax = 3500.0  # Max. conductance of Sodium current (S/m^2)
    GKMax = 60.0  # Max. conductance of Potassium current (S/m^2)
    GCaMax = 0.02  # Max. conductance of Calcium current (S/m^2)
    GKCaMax = 8.0  # Max. conductance of Calcium-dependent Potassium current (S/m^2)
    GL = 5.0  # Conductance of non-specific leakage current (S/m^2)

    diam = 50e-6  # Cell soma diameter (m)
    Z_Na = 1  # Sodium valence
    Z_Ca = 2  # Calcium valence


    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h', 'm4h'],
        'i_K\ kin.': ['n'],
        'i_{Ca}\ kin.': ['s'],
        'i_{KCa}\ kin.': ['c'],
        'pools': ['C_Na', 'C_Ca'],
        'I': ['iNa2', 'iK', 'iCa2', 'iKCa2', 'iPumpNa2', 'iPumpCa2', 'iL', 'iNet']
    }


    def __init__(self):
        ''' Constructor of the class. '''

        SV_ratio = 6 / self.diam  # surface to volume ratio of the (spherical) cell soma

        # Conversion constant from membrane ionic currents into
        # change rate of intracellular ionic concentrations
        self.K_Na = SV_ratio / (self.Z_Na * self.Faraday) * 1e-6  # Sodium (M/s)
        self.K_Ca = SV_ratio / (self.Z_Ca * self.Faraday) * 1e-6  # Calcium (M/s)

        # Names and initial states of the channels state probabilities
        self.states_names = ['m', 'h', 'n', 's', 'c', 'C_Na', 'C_Ca']
        self.states0 = np.array([])

        # Names of the channels effective coefficients
        self.coeff_names = ['alpham', 'betam', 'alphah', 'betah', 'alphan', 'betan',
                            'alphas', 'betas']

        # Define initial channel probabilities (solving dx/dt = 0 at resting potential)
        self.states0 = self.steadyStates(self.Vm0)

        # Charge interval bounds for lookup creation
        self.Qbounds = (np.round(self.Vm0 - 10.0) * 1e-5, 60.0e-5)


    def nernst(self, z_ion, C_ion_in, C_ion_out):
        ''' Return the Nernst potential of a specific ion given its intra and extracellular
            concentrations.

            :param z_ion: ion valence
            :param C_ion_in: intracellular ion concentration (M)
            :param C_ion_out: extracellular ion concentration (M)
            :return: ion Nernst potential (mV)
        '''

        return (self.Rg * self.T) / (z_ion * self.Faraday) * np.log(C_ion_out / C_ion_in) * 1e3


    def currPumpNa(self, C_Na_in):
        ''' Outward current mimicking the activity of the NaK-ATPase pump.

            :param C_Na_in: intracellular Sodium concentration (M)
            :return: current per unit area (mA/m2)
        '''

        return self.INaPmax / (1 + np.exp((self.khalf_Na - C_Na_in) / self.ksteep_Na))


    def currPumpCa(self, C_Ca_in):
        ''' Outward current representing the activity of a Calcium pump.

            :param C_Ca_in: intracellular Calcium concentration (M)
            :return: current per unit area (mA/m2)
        '''

        return self.iCaS * (C_Ca_in - self.C_Ca_in0) / 1.5


    def currNet(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        m, h, n, s, c, C_Na_in, C_Ca_in = states
        return (self.currNa(m, h, Vm, C_Na_in) + self.currK(n, Vm) + self.currCa(s, Vm, C_Ca_in) +
                self.currKCa(c, Vm) + self.currL(Vm) +
                (self.currPumpNa(C_Na_in) / 3.) + self.currPumpCa(C_Ca_in))  # mA/m2


    def steadyStates(self, Vm):
        ''' Concrete implementation of the abstract API method. '''

        # Intracellular concentrations
        C_Na_eq = self.C_Na_in0
        C_Ca_eq = self.C_Ca_in0

        # Standard gating dynamics: Solve the equation dx/dt = 0 at Vm for each x-state
        meq = self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm))
        heq = self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm))
        neq = self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm))
        seq = self.alphas(Vm) / (self.alphas(Vm) + self.betas(Vm))
        ceq = self.alphaC(C_Ca_eq) / (self.alphaC(C_Ca_eq) + self.betaC)

        return np.array([meq, heq, neq, seq, ceq, C_Na_eq, C_Ca_eq])


    def derStates(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        # Unpack states
        m, h, n, s, c, C_Na_in, C_Ca_in = states

        # Standard gating states derivatives
        dmdt = self.derM(Vm, m)
        dhdt = self.derH(Vm, h)
        dndt = self.derN(Vm, n)
        dsdt = self.derS(Vm, s)
        dcdt = self.derC(c, C_Ca_in)

        # Intracellular concentrations
        dCNa_dt = - (self.currNa(m, h, Vm, C_Na_in) + self.currPumpNa(C_Na_in)) * self.K_Na  # M/s
        dCCa_dt = -(self.currCa(s, Vm, C_Ca_in) + self.currPumpCa(C_Ca_in)) * self.K_Ca  # M/s

        # Pack derivatives and return
        return [dmdt, dhdt, dndt, dsdt, dcdt, dCNa_dt, dCCa_dt]


    def getEffRates(self, Vm):
        ''' Concrete implementation of the abstract API method. '''

        # Compute average cycle value for rate constants
        am_avg = np.mean(self.alpham(Vm))
        bm_avg = np.mean(self.betam(Vm))
        ah_avg = np.mean(self.alphah(Vm))
        bh_avg = np.mean(self.betah(Vm))
        an_avg = np.mean(self.alphan(Vm))
        bn_avg = np.mean(self.betan(Vm))
        as_avg = np.mean(self.alphas(Vm))
        bs_avg = np.mean(self.betas(Vm))

        # Return array of coefficients
        return np.array([am_avg, bm_avg, ah_avg, bh_avg, an_avg, bn_avg, as_avg, bs_avg])



    def derStatesEff(self, Qm, states, interp_data):
        ''' Concrete implementation of the abstract API method. '''

        rates = np.array([np.interp(Qm, interp_data['Q'], interp_data[rn])
                          for rn in self.coeff_names])
        Vmeff = np.interp(Qm, interp_data['Q'], interp_data['V'])

        # Unpack states
        m, h, n, s, c, C_Na_in, C_Ca_in = states

        # Standard gating states derivatives
        dmdt = rates[0] * (1 - m) - rates[1] * m
        dhdt = rates[2] * (1 - h) - rates[3] * h
        dndt = rates[4] * (1 - n) - rates[5] * n
        dsdt = rates[6] * (1 - s) - rates[7] * s

        # KCa current gating state derivative
        dcdt = self.derC(c, C_Ca_in)

        # Intracellular concentrations
        dCNa_dt = - (self.currNa(m, h, Vmeff, C_Na_in) + self.currPumpNa(C_Na_in)) * self.K_Na  # M/s
        dCCa_dt = -(self.currCa(s, Vmeff, C_Ca_in) + self.currPumpCa(C_Ca_in)) * self.K_Ca  # M/s

        # Pack derivatives and return
        return [dmdt, dhdt, dndt, dsdt, dcdt, dCNa_dt, dCCa_dt]


class LeechRetzius(LeechMech):
    ''' Class defining the membrane channel dynamics of a leech Retzius neuron.
        with 5 different current types:
            - Inward Sodium current
            - Outward Potassium current
            - Inward high-voltage-activated Calcium current
            - Non-specific leakage current
            - Calcium-dependent, outward Potassium current

        References:
        *Vazquez, Y., Mendez, B., Trueta, C., and De-Miguel, F.F. (2009). Summation of excitatory
        postsynaptic potentials in electrically-coupled neurones. Neuroscience 163, 202–212.*

        *ModelDB link: https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=120910*
    '''

    # Name of channel mechanism
    # name = 'LeechR'

    # Cell-specific biophysical parameters
    Cm0 = 5e-2  # Cell membrane resting capacitance (F/m2)
    Vm0 = -44.45  # Cell membrane resting potential (mV)

    VNa = 50.0  # Sodium Nernst potential, from retztemp.ses file on ModelDB (mV)
    VCa = 125.0  # Calcium Nernst potential, from cachdend.mod file on ModelDB (mV)
    VK = -79.0  # Potassium Nernst potential, from retztemp.ses file on ModelDB (mV)
    VL = -30.0  # Non-specific leakage Nernst potential, from leakdend.mod file on ModelDB (mV)

    GNaMax = 1250.0  # Max. conductance of Sodium current (S/m^2)
    GKMax = 10.0  # Max. conductance of Potassium current (S/m^2)
    GAMax = 100.0  # Max. conductance of transient Potassium current (S/m^2)
    GCaMax = 4.0  # Max. conductance of Calcium current (S/m^2)
    GKCaMax = 130.0  # Max. conductance of Calcium-dependent Potassium current (S/m^2)
    GL = 1.25  # Conductance of non-specific leakage current (S/m^2)

    Vhalf = -73.1  # mV

    C_Ca_in = 5e-8  # Calcium intracellular concentration, from retztemp.ses file (M)


    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h', 'm4h'],
        'i_K\ kin.': ['n'],
        'i_A\ kin.': ['a', 'b', 'ab'],
        'i_{Ca}\ kin.': ['s'],
        'i_{KCa}\ kin.': ['c'],
        'I': ['iNa', 'iK', 'iCa', 'iKCa2', 'iA', 'iL', 'iNet']
    }


    def __init__(self):
        ''' Constructor of the class. '''

        # Names and initial states of the channels state probabilities
        self.states_names = ['m', 'h', 'n', 's', 'c', 'a', 'b']
        self.states0 = np.array([])

        # Names of the channels effective coefficients
        self.coeff_names = ['alpham', 'betam', 'alphah', 'betah', 'alphan', 'betan',
                            'alphas', 'betas', 'alphac', 'betac', 'alphaa', 'betaa'
                            'alphab', 'betab']

        # Define initial channel probabilities (solving dx/dt = 0 at resting potential)
        self.states0 = self.steadyStates(self.Vm0)


    def ainf(self, Vm):
        ''' Steady-state activation probability of transient Potassium channels.

            Source:
            *Beck, H., Ficker, E., and Heinemann, U. (1992). Properties of two voltage-activated
            potassium currents in acutely isolated juvenile rat dentate gyrus granule cells.
            J. Neurophysiol. 68, 2086–2099.*

            :param Vm: membrane potential (mV)
            :return: time constant (s)
        '''

        Vth = -55.0  # mV
        return 0 if Vm <= Vth else min(1, 2 * (Vm - Vth)**3 / ((11 - Vth)**3 + (Vm - Vth)**3))


    def taua(self, Vm):
        ''' Activation time constant of transient Potassium channels. (assuming T = 20°C).

            Source:
            *Beck, H., Ficker, E., and Heinemann, U. (1992). Properties of two voltage-activated
            potassium currents in acutely isolated juvenile rat dentate gyrus granule cells.
            J. Neurophysiol. 68, 2086–2099.*

            :param Vm: membrane potential (mV)
            :return: time constant (s)
        '''

        x = -1.5 * (Vm - self.Vhalf) * 1e-3 * self.Faraday / (self.Rg * self.T)  # [-]
        alpha = np.exp(x)  # ms-1
        beta = np.exp(0.7 * x)  # ms-1
        return max(0.5, beta / (0.3 * (1 + alpha))) * 1e-3  # s


    def binf(self, Vm):
        ''' Steady-state inactivation probability of transient Potassium channels.

            Source:
            *Beck, H., Ficker, E., and Heinemann, U. (1992). Properties of two voltage-activated
            potassium currents in acutely isolated juvenile rat dentate gyrus granule cells.
            J. Neurophysiol. 68, 2086–2099.*

            :param Vm: membrane potential (mV)
            :return: time constant (s)
        '''

        return 1. / (1 + np.exp((self.Vhalf - Vm) / -6.3))


    def taub(self, Vm):
        ''' Inactivation time constant of transient Potassium channels. (assuming T = 20°C).

            Source:
            *Beck, H., Ficker, E., and Heinemann, U. (1992). Properties of two voltage-activated
            potassium currents in acutely isolated juvenile rat dentate gyrus granule cells.
            J. Neurophysiol. 68, 2086–2099.*

            :param Vm: membrane potential (mV)
            :return: time constant (s)
        '''

        x = 2 * (Vm - self.Vhalf) * 1e-3 * self.Faraday / (self.Rg * self.T)
        alpha = np.exp(x)
        beta = np.exp(0.65 * x)
        return max(7.5, beta / (0.02 * (1 + alpha))) * 1e-3  # s


    def derA(self, Vm, a):
        ''' Compute the evolution of the activation-probability of transient Potassium channels.

            :param Vm: membrane potential (mV)
            :param a: activation-probability of transient Potassium channels (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''

        return (self.ainf(Vm) - a) / self.taua(Vm)


    def derB(self, Vm, b):
        ''' Compute the evolution of the inactivation-probability of transient Potassium channels.

            :param Vm: membrane potential (mV)
            :param b: inactivation-probability of transient Potassium channels (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''

        return (self.binf(Vm) - b) / self.taub(Vm)


    def currA(self, a, b, Vm):
        ''' Compute the outward, transient Potassium current per unit area.

            :param a: open-probability of transient Potassium channels
            :param b: inactivation-probability of transient Potassium channels
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        GK = self.GAMax * a * b
        return GK * (Vm - self.VK)


    def currNet(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        m, h, n, s, c, a, b = states
        return (self.currNa(m, h, Vm) + self.currK(n, Vm) + self.currCa(s, Vm) + self.currL(Vm) +
                self.currKCa(c, Vm) + self.currA(a, b, Vm))  # mA/m2


    def steadyStates(self, Vm):
        ''' Concrete implementation of the abstract API method. '''

        # Standard gating dynamics: Solve the equation dx/dt = 0 at Vm for each x-state
        meq = self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm))
        heq = self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm))
        neq = self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm))
        seq = self.alphas(Vm) / (self.alphas(Vm) + self.betas(Vm))
        ceq = self.alphaC(self.C_Ca_in) / (self.alphaC(self.C_Ca_in) + self.betaC)
        aeq = self.ainf(Vm)
        beq = self.binf(Vm)

        return np.array([meq, heq, neq, seq, ceq, aeq, beq])


    def derStates(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        # Unpack states
        m, h, n, s, c, a, b = states

        # Standard gating states derivatives
        dmdt = self.derM(Vm, m)
        dhdt = self.derH(Vm, h)
        dndt = self.derN(Vm, n)
        dsdt = self.derS(Vm, s)
        dcdt = self.derC(c, self.C_Ca_in)
        dadt = self.derA(Vm, a)
        dbdt = self.derB(Vm, b)

        # Pack derivatives and return
        return [dmdt, dhdt, dndt, dsdt, dcdt, dadt, dbdt]


    def getEffRates(self, Vm):
        ''' Concrete implementation of the abstract API method. '''

        # Compute average cycle value for rate constants
        am_avg = np.mean(self.alpham(Vm))
        bm_avg = np.mean(self.betam(Vm))
        ah_avg = np.mean(self.alphah(Vm))
        bh_avg = np.mean(self.betah(Vm))
        an_avg = np.mean(self.alphan(Vm))
        bn_avg = np.mean(self.betan(Vm))
        as_avg = np.mean(self.alphas(Vm))
        bs_avg = np.mean(self.betas(Vm))

        Ta = self.taua(Vm)
        ainf = self.ainf(Vm)
        aa_avg = np.mean(ainf / Ta)
        ba_avg = np.mean(1 / Ta) - aa_avg

        Tb = self.taub(Vm)
        binf = self.binf(Vm)
        ab_avg = np.mean(binf / Tb)
        bb_avg = np.mean(1 / Tb) - ab_avg

        # Return array of coefficients
        return np.array([am_avg, bm_avg, ah_avg, bh_avg, an_avg, bn_avg, as_avg, bs_avg,
                         aa_avg, ba_avg, ab_avg, bb_avg])



    def derStatesEff(self, Qm, states, interp_data):
        ''' Concrete implementation of the abstract API method. '''

        rates = np.array([np.interp(Qm, interp_data['Q'], interp_data[rn])
                          for rn in self.coeff_names])

        # Unpack states
        m, h, n, s, c, a, b = states

        # Standard gating states derivatives
        dmdt = rates[0] * (1 - m) - rates[1] * m
        dhdt = rates[2] * (1 - h) - rates[3] * h
        dndt = rates[4] * (1 - n) - rates[5] * n
        dsdt = rates[6] * (1 - s) - rates[7] * s
        dadt = rates[8] * (1 - a) - rates[9] * a
        dbdt = rates[10] * (1 - b) - rates[11] * b

        # KCa current gating state derivative
        dcdt = self.derC(c, self.C_Ca_in)

        # Pack derivatives and return
        return [dmdt, dhdt, dndt, dsdt, dcdt, dadt, dbdt]
