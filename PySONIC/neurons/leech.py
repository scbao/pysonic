#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-31 15:20:54
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-16 15:25:06


from functools import partialmethod
import numpy as np

from ..core import PointNeuron
from ..constants import FARADAY, Rg, Z_Na, Z_Ca


class LeechTouch(PointNeuron):
    ''' Leech touch sensory neuron

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
    ENa = 45.0  # Sodium Nernst potential (mV)
    EK = -62.0  # Potassium Nernst potential (mV)
    ECa = 60.0  # Calcium Nernst potential (mV)
    ELeak = -48.0  # Non-specific leakage Nernst potential (mV)
    EPumpNa = -300.0  # Sodium pump current reversal potential (mV)
    gNabar = 3500.0  # Max. conductance of Sodium current (S/m^2)
    gKbar = 900.0  # Max. conductance of Potassium current (S/m^2)
    gCabar = 20.0  # Max. conductance of Calcium current (S/m^2)
    gKCabar = 236.0  # Max. conductance of Calcium-dependent Potassium current (S/m^2)
    gLeak = 1.0  # Conductance of non-specific leakage current (S/m^2)
    gPumpNa = 20.0  # Max. conductance of Sodium pump current (S/m^2)
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
    taur_Na = 16.0  # Na+ removal
    taur_Ca = 1.25  # Ca2+ removal

    # Time constants for the iPumpNa and iKCa currents activation
    # from specific intracellular ions (s)
    taua_PumpNa = 0.1  # iPumpNa activation from intracellular Na+
    taua_KCa = 0.01  # iKCa activation from intracellular Ca2+


    def __init__(self):
        self.states = ['m', 'h', 'n', 's', 'Nai', 'ANai', 'Cai', 'ACai']
        self.rates = self.getRatesNames(['m', 'h', 'n', 's'])
        self.K_Na = self.K_Na_original * self.surface * self.curr_factor
        self.K_Ca = self.K_Ca_original * self.surface * self.curr_factor


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


    def _derCion(self, Cion, Iion, Kion, tau):
        ''' Generic function computing the time derivative of the concentration
            of a specific ion in its intracellular pool.

            :param Cion: ion concentration in the pool (arbitrary unit)
            :param Iion: ionic current (mA/m2)
            :param Kion: scaling factor for current contribution to pool (arb. unit / nA???)
            :param tau: time constant for removal of ions from the pool (s)
            :return: variation of ionic concentration in the pool (arbitrary unit /s)
        '''

        return (Kion * (-Iion) - Cion) / tau


    def _derAion(self, Aion, Cion, tau):
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


    def derNai(self, Nai, m, h, Vm):
        ''' Derivative of Sodium concentration in intracellular pool. '''
        return self._derCion(Nai, self.iNa(m, h, Vm), self.K_Na, self.taur_Na)


    def derANa(self, ANa, Nai):
        ''' Derivative of Sodium pool-dependent activation function for iPumpNa. '''
        return self._derAion(ANa, Nai, self.taua_PumpNa)


    def derCai(self, Cai, s, Vm):
        ''' Derivative of Calcium concentration in intracellular pool. '''
        return self._derCion(Cai, self.iCa(s, Vm), self.K_Ca, self.taur_Ca)


    def derACa(self, ACa, Cai):
        ''' Derivative of Calcium pool-dependent activation function for iKCa. '''
        return self._derAion(ACa, Cai, self.taua_KCa)


    # ------------------ Currents -------------------

    def iNa(self, m, h, Vm):
        ''' Sodium current

            :param m: open-probability of m-gate
            :param h: open-probability of h-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gNabar * m**3 * h * (Vm - self.ENa)


    def iK(self, n, Vm):
        ''' Delayed-rectifier Potassium current

            :param n: open-probability of n-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gKbar * n**2 * (Vm - self.EK)


    def iCa(self, s, Vm):
        ''' Calcium current

            :param s: open-probability of s-gate
            :param u: open-probability of u-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        ''' Calcium inward current. '''
        return self.gCabar * s * (Vm - self.ECa)


    def iKCa(self, ACa, Vm):
        ''' Calcium-activated Potassium current

            :param ACa: Calcium pool-dependent gate opening
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gKCabar * ACa * (Vm - self.EK)


    def iPumpNa(self, ANa, Vm):
        ''' NaK-ATPase pump current

            :param ANa: Sodium pool-dependent gate opening.
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gPumpNa * ANa * (Vm - self.EPumpNa)


    def iLeak(self, Vm):
        ''' Non-specific leakage current

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gLeak * (Vm - self.ELeak)


    def currents(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, s, _, ANa, _, ACa = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iK': self.iK(n, Vm),
            'iCa': self.iCa(s, Vm),
            'iLeak': self.iLeak(Vm),
            'iPumpNa': self.iPumpNa(ANa, Vm),
            'iKCa': self.iKCa(ACa, Vm)
        }  # mA/m2


    def steadyStates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Standard gating dynamics: Solve the equation dx/dt = 0 at Vm for each x-state
        sstates = {
            'm': self.minf(Vm),
            'h': self.hinf(Vm),
            'n': self.ninf(Vm),
            's': self.sinf(Vm)
        }

        # PumpNa pool concentration and activation steady-state
        sstates['CNa'] = - self.K_Na * self.iNa(sstates['m'], sstates['h'], Vm)
        sstates['ANa'] = sstates['CNa']

        # KCa current pool concentration and activation steady-state
        sstates['CCa'] = - self.K_Ca * self.iCa(sstates['s'], Vm)
        sstates['ACa'] = sstates['CCa']

        return sstates


    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, s, Nai, ANa, Cai, ACa = states

        # Standard gating states derivatives
        dstates = {
            'm': self.derM(Vm, m),
            'h': self.derH(Vm, h),
            'n': self.derN(Vm, n),
            's': self.derS(Vm, s)
        }

        # PumpNa current pool concentration and activation state
        dstates['CNa'] = self.derNai(Nai, m, h, Vm)
        dstates['ANa'] = self.derANa(ANa, Nai)

        # KCa current pool concentration and activation state
        dstates['CCa'] = self.derCai(Cai, s, Vm)
        dstates['ACa'] = self.derACa(ACa, Cai)

        # Pack derivatives and return
        return dstates


    def computeEffRates(self, Vm):
        ''' Overriding of abstract parent method. '''

        return {
            'alpham': np.mean(self.minf(Vm) / self.taum(Vm)),
            'betam': np.mean((1 - self.minf(Vm)) / self.taum(Vm)),
            'alphah': np.mean(self.hinf(Vm) / self.tauh(Vm)),
            'betah': np.mean((1 - self.hinf(Vm)) / self.tauh(Vm)),
            'alphan': np.mean(self.ninf(Vm) / self.taun(Vm)),
            'betan': np.mean((1 - self.ninf(Vm)) / self.taun(Vm)),
            'alphas': np.mean(self.sinf(Vm) / self.taus(Vm)),
            'betas': np.mean((1 - self.sinf(Vm)) / self.taus(Vm))
        }


    def derEffStates(self, Qm, states, lkp):
        ''' Overriding of abstract parent method. '''

        rates = self.interpEffRates(Qm, lkp)
        Vmeff = self.interpVmeff(Qm, lkp)
        m, h, n, s, Nai, ANa, Cai, ACa = states

        # Standard gating states derivatives
        dstates = {
            'm': rates['alpham'] * (1 - m) - rates['betam'] * m,
            'h': rates['alphah'] * (1 - h) - rates['betah'] * h,
            'n': rates['alphan'] * (1 - n) - rates['betan'] * n,
            's': rates['alphas'] * (1 - s) - rates['betas'] * s
        }

        # PumpNa current pool concentration and activation state
        dstates['CNa'] = self.derNai(Nai, m, h, Vmeff)
        dstates['ANa'] = self.derANa(ANa, Nai)

        # KCa current pool concentration and activation state
        dstates['CCa'] = self.derCai(Cai, s, Vmeff)
        dstates['ACa'] = self.derACa(ACa, Cai)

        # Pack derivatives and return
        return dstates


    def quasiSteadyStates(self, lkp):
        ''' Overriding of abstract parent method. '''
        qsstates = self.qsStates(lkp, ['m', 'h', 'n', 's'])

        # PumpNa pool concentration and activation steady-state
        qsstates['CNa'] = - self.K_Na * self.iNa(qsstates['m'], qsstates['h'], lkp['V'])
        qsstates['ANa'] = qsstates['CNa']

        # KCa current pool concentration and activation steady-state
        qsstates['CCa'] = - self.K_Ca * self.iCa(qsstates['s'], lkp['V'])
        qsstates['ACa'] = qsstates['CCa']

        return qsstates



class LeechMech(PointNeuron):
    ''' Generic leech neuron

        Reference:
        *Baccus, S.A. (1998). Synaptic facilitation by reflected action potentials: enhancement
        of transmission when nerve impulses reverse direction at axon branch points. Proc. Natl.
        Acad. Sci. U.S.A. 95, 8345–8350.*
    '''

    alphaC_sf = 1e-5  # Calcium activation rate constant scaling factor (M)
    betaC = 0.1e3  # beta rate for the open-probability of Ca2+-dependent Potassium channels (s-1)
    T = 293.15  # Room temperature (K)


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


    def alphaC(self, Cai):
        ''' Compute the alpha rate for the open-probability of Calcium-dependent Potassium channels.

            :param Cai: intracellular Calcium concentration (M)
            :return: rate constant (s-1)
        '''
        alpha = 0.1 * Cai / self.alphaC_sf  # ms-1
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


    def derC(self, c, Cai):
        ''' Compute the evolution of the open-probability of Calcium-dependent Potassium channels.

            :param c: open-probability of Calcium-dependent Potassium channels (prob)
            :param Cai: intracellular Calcium concentration (M)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''
        return self.alphaC(Cai) * (1 - c) - self.betaC * c


    def iNa(self, m, h, Vm, Nai):
        ''' Sodium current

            :param m: open-probability of m-gate
            :param h: open-probability of Sodium channels
            :param Vm: membrane potential (mV)
            :param Nai: intracellular Sodium concentration (M)
            :return: current per unit area (mA/m2)
        '''
        GNa = self.gNabar * m**4 * h
        ENa = self.nernst(Z_Na, Nai, self.C_Na_out, self.T)  # mV
        return GNa * (Vm - ENa)


    def iK(self, n, Vm):
        ''' Delayed-rectifier Potassium current

            :param n: open-probability of n-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        GK = self.gKbar * n**2
        return GK * (Vm - self.EK)


    def iCa(self, s, Vm, Cai):
        ''' Calcium current

            :param s: open-probability of s-gate
            :param Vm: membrane potential (mV)
            :param Cai: intracellular Calcium concentration (M)
            :return: current per unit area (mA/m2)
        '''
        GCa = self.gCabar * s
        ECa = self.nernst(Z_Ca, Cai, self.C_Ca_out, self.T)  # mV
        return GCa * (Vm - ECa)


    def iKCa(self, c, Vm):
        ''' Calcium-activated Potassium current

            :param c: open-probability of c-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        GKCa = self.gKCabar * c
        return GKCa * (Vm - self.EK)


    def iLeak(self, Vm):
        ''' Non-specific leakage current

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gLeak * (Vm - self.ELeak)


class LeechPressure(LeechMech):
    ''' Leech pressure sensory neuron

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
    Nai0 = 0.01  # Initial Sodium intracellular concentration (M)
    Cai0 = 1e-7  # Initial Calcium intracellular concentration (M)

    # ENa = 60  # Sodium Nernst potential, from MOD file on ModelDB (mV)
    # ECa = 125  # Calcium Nernst potential, from MOD file on ModelDB (mV)
    EK = -68.0  # Potassium Nernst potential (mV)
    ELeak = -49.0  # Non-specific leakage Nernst potential (mV)
    INaPmax = 70.0  # Maximum pump rate of the NaK-ATPase (mA/m2)
    khalf_Na = 0.012  # Sodium concentration at which NaK-ATPase is at half its maximum rate (M)
    ksteep_Na = 1e-3  # Sensitivity of NaK-ATPase to varying Sodium concentrations (M)
    iCaS = 0.1  # Calcium pump current parameter (mA/m2)
    gNabar = 3500.0  # Max. conductance of Sodium current (S/m^2)
    gKbar = 60.0  # Max. conductance of Potassium current (S/m^2)
    gCabar = 0.02  # Max. conductance of Calcium current (S/m^2)
    gKCabar = 8.0  # Max. conductance of Calcium-dependent Potassium current (S/m^2)
    gLeak = 5.0  # Conductance of non-specific leakage current (S/m^2)

    diam = 50e-6  # Cell soma diameter (m)


    def __init__(self):
        ''' Constructor of the class. '''

        SV_ratio = 6 / self.diam  # surface to volume ratio of the (spherical) cell soma

        # Conversion constant from membrane ionic currents into
        # change rate of intracellular ionic concentrations
        self.K_Na = SV_ratio / (Z_Na * FARADAY) * 1e-6  # Sodium (M/s)
        self.K_Ca = SV_ratio / (Z_Ca * FARADAY) * 1e-6  # Calcium (M/s)

        # Names and initial states of the channels state probabilities
        self.states = ['m', 'h', 'n', 's', 'c', 'Nai', 'Cai']

        # Names of the channels effective coefficients
        self.rates = self.getRatesNames(['m', 'h', 'n', 's'])


    def iPumpNa(self, Nai):
        ''' NaK-ATPase pump current

            :param Nai: intracellular Sodium concentration (M)
            :return: current per unit area (mA/m2)
        '''
        return self.INaPmax / (1 + np.exp((self.khalf_Na - Nai) / self.ksteep_Na))


    def iPumpCa(self, Cai):
        ''' Calcium pump current

            :param Cai: intracellular Calcium concentration (M)
            :return: current per unit area (mA/m2)
        '''
        return self.iCaS * (Cai - self.Cai0) / 1.5


    def currents(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, s, c, Nai, Cai = states
        return {
            'iNa': self.iNa(m, h, Vm, Nai),
            'iK': self.iK(n, Vm),
            'iCa': self.iCa(s, Vm, Cai),
            'iKCa': self.iKCa(c, Vm),
            'iLeak': self.iLeak(Vm),
            'iPumpNa': self.iPumpNa(Nai) / 3.,
            'iPumpCa': self.iPumpCa(Cai)
        }  # mA/m2


    def steadyStates(self, Vm):
        ''' Overriding of abstract parent method. '''
        sstates = {
            'm': self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            's': self.alphas(Vm) / (self.alphas(Vm) + self.betas(Vm)),
            'Nai': self.Nai0,
            'Cai': self.Cai0
        }
        sstates['c'] = self.alphaC(sstates['Cai']) / (self.alphaC(sstates['Cai']) + self.betaC)
        return sstates


    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, s, c, Nai, Cai = states
        return {
            'm': self.derM(Vm, m),
            'h': self.derH(Vm, h),
            'n': self.derN(Vm, n),
            's': self.derS(Vm, s),
            'c': self.derC(c, Cai),
            'Nai': -(self.iNa(m, h, Vm, Nai) + self.iPumpNa(Nai)) * self.K_Na,  # M/s
            'Cai': -(self.iCa(s, Vm, Cai) + self.iPumpCa(Cai)) * self.K_Ca  # M/s'
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
            'alphas': np.mean(self.alphas(Vm)),
            'betas': np.mean(self.betas(Vm))
        }


    def derEffStates(self, Qm, states, lkp):
        ''' Overriding of abstract parent method. '''

        rates = self.interpEffRates(Qm, lkp)
        Vmeff = self.interpVmeff(Qm, lkp)
        m, h, n, s, c, Nai, Cai = states

        return {
            'm': rates['alpham'] * (1 - m) - rates['betam'] * m,
            'h': rates['alphah'] * (1 - h) - rates['betah'] * h,
            'n': rates['alphan'] * (1 - n) - rates['betan'] * n,
            's': rates['alphas'] * (1 - s) - rates['betas'] * s,
            'c': self.derC(c, Cai),
            'Nai': -(self.iNa(m, h, Vmeff, Nai) + self.iPumpNa(Nai)) * self.K_Na,  # M/s
            'Cai': -(self.iCa(s, Vmeff, Cai) + self.iPumpCa(Cai)) * self.K_Ca  # M/s
        }

    def quasiSteadyStates(self, lkp):
        ''' Overriding of abstract parent method. '''
        qsstates = self.qsStates(lkp, ['m', 'h', 'n', 's'])
        qsstates.update({'Nai': self.Nai0, 'Cai': self.Cai0})
        qsstates['c'] = self.alphaC(qsstates['Cai']) / (self.alphaC(qsstates['Cai']) + self.betaC)

        return qsstates


class LeechRetzius(LeechMech):
    ''' Leech Retzius neuron

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

    ENa = 50.0  # Sodium Nernst potential, from retztemp.ses file on ModelDB (mV)
    ECa = 125.0  # Calcium Nernst potential, from cachdend.mod file on ModelDB (mV)
    EK = -79.0  # Potassium Nernst potential, from retztemp.ses file on ModelDB (mV)
    ELeak = -30.0  # Non-specific leakage Nernst potential, from leakdend.mod file on ModelDB (mV)

    gNabar = 1250.0  # Max. conductance of Sodium current (S/m^2)
    gKbar = 10.0  # Max. conductance of Potassium current (S/m^2)
    GAMax = 100.0  # Max. conductance of transient Potassium current (S/m^2)
    gCabar = 4.0  # Max. conductance of Calcium current (S/m^2)
    gKCabar = 130.0  # Max. conductance of Calcium-dependent Potassium current (S/m^2)
    gLeak = 1.25  # Conductance of non-specific leakage current (S/m^2)

    Vhalf = -73.1  # mV

    Cai = 5e-8  # Calcium intracellular concentration, from retztemp.ses file (M)


    def __init__(self):
        ''' Constructor of the class. '''
        self.states = ['m', 'h', 'n', 's', 'c', 'a', 'b']
        self.rates = self.getRatesNames([self.states])


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
        x = -1.5 * (Vm - self.Vhalf) * 1e-3 * FARADAY / (Rg * self.T)  # [-]
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
        x = 2 * (Vm - self.Vhalf) * 1e-3 * FARADAY / (Rg * self.T)
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


    def iA(self, a, b, Vm):
        ''' Transient Potassium current

            :param a: open-probability of a-gate
            :param b: open-probability of b-gate
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        GK = self.GAMax * a * b
        return GK * (Vm - self.EK)


    def currents(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, s, c, a, b = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iK': self.iK(n, Vm),
            'iCa': self.iCa(s, Vm),
            'iLeak': self.iLeak(Vm),
            'iKCa': self.iKCa(c, Vm),
            'iA': self.iA(a, b, Vm)
        }  # mA/m2


    def steadyStates(self, Vm):
        ''' Overriding of abstract parent method. '''
        return {
            'm': self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            's': self.alphas(Vm) / (self.alphas(Vm) + self.betas(Vm)),
            'c': self.alphaC(self.Cai) / (self.alphaC(self.Cai) + self.betaC),
            'a': self.ainf(Vm),
            'b': self.binf(Vm)
        }


    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, s, c, a, b = states

        return {
            'm': self.derM(Vm, m),
            'h': self.derH(Vm, h),
            'n': self.derN(Vm, n),
            's': self.derS(Vm, s),
            'c': self.derC(c, self.Cai),
            'a': self.derA(Vm, a),
            'b': self.derB(Vm, b)
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
            'alphas': np.mean(self.alphas(Vm)),
            'betas': np.mean(self.betas(Vm)),
            'alphaa': np.mean(self.ainf(Vm) / self.taua(Vm)),
            'betaa': np.mean((1 - self.ainf(Vm)) / self.taua(Vm)),
            'alphab': np.mean(self.binf(Vm) / self.taub(Vm)),
            'betab': np.mean((1 - self.binf(Vm)) / self.taub(Vm))
        }


    def derEffStates(self, Qm, states, lkp):
        ''' Overriding of abstract parent method. '''

        rates = self.interpEffRates(Qm, lkp)
        m, h, n, s, c, a, b = states

        # Standard gating states derivatives
        return {
            'm': rates['alpham'] * (1 - m) - rates['betam'] * m,
            'h': rates['alphah'] * (1 - h) - rates['betah'] * h,
            'n': rates['alpham'] * (1 - n) - rates['betam'] * n,
            's': rates['alphas'] * (1 - s) - rates['betas'] * s,
            'a': rates['alphaa'] * (1 - a) - rates['betaa'] * a,
            'b': rates['alphab'] * (1 - b) - rates['betab'] * b,
            'c': self.derC(c, self.Cai)
        }


    def quasiSteadyStates(self, lkp):
        ''' Overriding of abstract parent method. '''
        qsstates = self.qsStates(lkp, ['m', 'h', 'n', 's', 'a', 'b'])
        qsstates['c'] = self.alphaC(self.Cai) / (self.alphaC(self.Cai) + self.betaC),

        return qsstates
