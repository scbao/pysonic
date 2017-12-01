#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-31 15:20:54
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-12-01 08:59:02

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



class LeechPressure(BaseMech):
    ''' Class defining the membrane channel dynamics of a leech pressure sensory neuron.
        with 4 different current types:
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
    Vm0 = -53.9  # Cell membrane resting potential (mV)
    C_Na_out = 0.11  # Sodium extracellular concentration (M)
    C_Ca_out = 1.8e-3  # Calcium extracellular concentration (M)
    C_Na_in0 = 0.01  # Initial Sodium intracellular concentration (M)
    C_Ca_in0 = 1e-7  # Initial Calcium intracellular concentration (M)
    VK = -68.0  # Potassium Nernst potential (mV)
    VL = -49.0  # Non-specific leakage Nernst potential (mV)
    INaPmax = 70.0  # Sodium pump current parameter (mA/m2)
    khalf_Na = 0.012  # Sodium pump current parameter (M)
    ksteep_Na = 1e-3  # Sodium pump current parameter (M)
    iCaS = 0.1  # Calcium pump current parameter (mA/m2)
    C_Ca_rest = 1e-7  # Calcium pump current parameter (M)
    GNaMax = 3500.0  # Max. conductance of Sodium current (S/m^2)
    GKMax = 60.0  # Max. conductance of Potassium current (S/m^2)
    GCaMax = 0.02  # Max. conductance of Calcium current (S/m^2)
    GKCaMax = 8.0  # Max. conductance of Calcium-dependent Potassium current (S/m^2)
    GL = 5.0  # Conductance of non-specific leakage current (S/m^2)
    betaw = 1e-4  # beta rate for the open-probability of Calcium-dependent Potassium channels (s-1)

    T = 309.15  # Temperature (K, same as in the BilayerSonophore class)
    Rg = 8.314  # Universal gas constant (J.mol^-1.K^-1)
    F = 9.6485e4  # Faraday constant (C/mol)
    diam = 50e-6  # Cell soma diameter (m)


    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h', 'm4h'],
        'i_K\ kin.': ['n'],
        'i_{Ca}\ kin.': ['s'],
        'i_{KCa}\ kin.': ['w'],
        'pools': ['C_Na', 'C_Ca'],
        'I': ['iNa2', 'iK', 'iCa2', 'iKCa2', 'iPumpNa2', 'iPumpCa2', 'iL', 'iNet']
    }


    def __init__(self):
        ''' Constructor of the class. '''

        # Conversion constant from membrane ionic current into
        # change rate of intracellular ionic concentration
        self.K = 4 / (self.diam * self.F) * 10  # M/s

        # Names and initial states of the channels state probabilities
        self.states_names = ['m', 'h', 'n', 's', 'w', 'C_Na', 'C_Ca']
        self.states0 = np.array([])

        # Names of the channels effective coefficients
        self.coeff_names = ['alpham', 'betam', 'alphah', 'betah', 'alphan', 'betan',
                            'alphas', 'betas', 'alphaw', 'betaw']

        # Define initial channel probabilities (solving dx/dt = 0 at resting potential)
        self.states0 = self.steadyStates(self.Vm0)


    def nernst(self, z_ion, C_ion_in, C_ion_out):
        ''' Return the Nernst potential of a specific ion given its intra and extracellular
            concentrations.

            :param z_ion: ion valence
            :param C_ion_in: intracellular ion concentration (M)
            :param C_ion_out: extracellular ion concentration (M)
            :return: ion Nernst potential (mV)
        '''

        return (self.Rg * self.T) / (z_ion * self.F) * np.log(C_ion_out / C_ion_in) * 1e3


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
        '''

        beta = 0.72 * (np.exp(-(Vm + 23) / 14) + 1)  # ms-1
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


    def alphaw(self, C_Ca_in):
        ''' Compute the alpha rate for the open-probability of Calcium-dependent Potassium channels.

            :param C_Ca_in: intracellular Calcium concentration (M)
            :return: rate constant (s-1)
        '''

        alpha = 0.1 * C_Ca_in / 10  # ms-1
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


    def derW(self, w, C_Ca_in):
        ''' Compute the evolution of the open-probability of Calcium-dependent Potassium channels.

            :param w: open-probability of Calcium-dependent Potassium channels (prob)
            :param C_Ca_in: intracellular Calcium concentration (M)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''

        return self.alphaw(C_Ca_in) * (1 - w) - self.betaw * w


    def currNa(self, m, h, Vm, C_Na_in):
        ''' Compute the inward Sodium current per unit area.

            :param m: open-probability of Sodium channels
            :param h: inactivation-probability of Sodium channels
            :param Vm: membrane potential (mV)
            :param C_Na_in: intracellular Sodium concentration (M)
            :return: current per unit area (mA/m2)
        '''

        GNa = self.GNaMax * m**4 * h
        VNa = self.nernst(1, C_Na_in, self.C_Na_out)  # Sodium Nernst potential
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
        VCa = self.nernst(1, C_Ca_in, self.C_Ca_out)  # Calcium Nernst potential
        return GCa * (Vm - VCa)


    def currKCa(self, w, Vm):
        ''' Compute the outward Calcium-dependent Potassium current per unit area.

            :param w: open-probability of Calcium-dependent Potassium channels
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        GKCa = self.GKCaMax * w
        return GKCa * (Vm - self.VK)


    def currL(self, Vm):
        ''' Compute the non-specific leakage current per unit area.

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        return self.GL * (Vm - self.VL)


    def currPumpNa(self, C_Na_in):
        ''' Outward current mimicking the activity of the NaK-ATPase pump. '''

        INaPump = self.INaPmax / (1 + np.exp((self.khalf_Na - C_Na_in) / self.ksteep_Na))
        return INaPump / 3


    def currPumpCa(self, C_Ca_in):
        ''' Outward current representing the intracellular Calcium removal. '''

        return self.iCaS * (C_Ca_in - self.C_Ca_rest) / 1.5


    def currNet(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        m, h, n, s, w, C_Na_in, C_Ca_in = states
        return (self.currNa(m, h, Vm, C_Na_in) + self.currK(n, Vm) + self.currCa(s, Vm, C_Ca_in)
                + self.currKCa(w, Vm) + self.currL(Vm)
                + self.currPumpNa(C_Na_in) + self.currPumpCa(C_Ca_in))  # mA/m2


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
        weq = self.alphaw(C_Ca_eq) / (self.alphaw(C_Ca_eq) + self.betaw)

        return np.array([meq, heq, neq, seq, weq, C_Na_eq, C_Ca_eq])


    def derStates(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        # Unpack states
        m, h, n, s, w, C_Na_in, C_Ca_in = states

        # Standard gating states derivatives
        dmdt = self.derM(Vm, m)
        dhdt = self.derH(Vm, h)
        dndt = self.derN(Vm, n)
        dsdt = self.derS(Vm, s)
        dwdt = self.derW(w, C_Ca_in)

        # Intracellular concentrations
        dCNa_dt = -self.currNa(m, h, Vm, C_Na_in) * self.K  # M/s
        dCCa_dt = -self.currCa(s, Vm, C_Ca_in) * self.K  # M/s

        # Pack derivatives and return
        return [dmdt, dhdt, dndt, dsdt, dwdt, dCNa_dt, dCCa_dt]


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
        m, h, n, s, w, C_Na_in, C_Ca_in = states

        # Standard gating states derivatives
        dmdt = rates[0] * (1 - m) - rates[1] * m
        dhdt = rates[2] * (1 - h) - rates[3] * h
        dndt = rates[4] * (1 - n) - rates[5] * n
        dsdt = rates[6] * (1 - s) - rates[7] * s

        # KCa current gating state derivative
        dwdt = self.derW(w, C_Ca_in)

        # Intracellular concentrations
        dCNa_dt = -self.currNa(m, h, Vmeff, C_Na_in) * self.K  # M/s
        dCCa_dt = -self.currCa(s, Vmeff, C_Ca_in) * self.K  # M/s

        # Pack derivatives and return
        return [dmdt, dhdt, dndt, dsdt, dwdt, dCNa_dt, dCCa_dt]
