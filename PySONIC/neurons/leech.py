# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-31 15:20:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-18 18:20:51

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

    # Neuron name
    name = 'LeechT'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2  # Membrane capacitance (F/m2)
    Vm0 = -53.58  # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 45.0        # Sodium
    EK = -62.0        # Potassium
    ECa = 60.0        # Calcium
    ELeak = -48.0     # Non-specific leakage
    EPumpNa = -300.0  # Sodium pump

    # Maximal channel conductances (S/m2)
    gNabar = 3500.0  # Sodium
    gKdbar = 900.0   # Delayed-rectifier Potassium
    gCabar = 20.0    # Calcium
    gKCabar = 236.0  # Calcium-dependent Potassium
    gLeak = 1.0      # Non-specific leakage
    gPumpNa = 20.0   # Sodium pump

    # Activation time constants (s)
    taum = 0.1e-3  # Sodium
    taus = 0.6e-3  # Calcium

    # Original conversion constants from inward ionic current (nA) to build-up of
    # intracellular ion concentration (arb.)
    K_Na_original = 0.016  # iNa to intracellular [Na+]
    K_Ca_original = 0.1    # iCa to intracellular [Ca2+]

    # Constants needed to convert K from original model (soma compartment)
    # to current model (point-neuron)
    surface = 6434.0e-12  # surface of cell assumed as a single soma (m2)
    curr_factor = 1e6     # mA to nA

    # Time constants for the removal of ions from intracellular pools (s)
    taur_Na = 16.0  # Sodium
    taur_Ca = 1.25  # Calcium

    # Time constants for the PumpNa and KCa currents activation
    # from specific intracellular ions (s)
    taua_PumpNa = 0.1  # PumpNa current activation from intracellular Na+
    taua_KCa = 0.01    # KCa current activation from intracellular Ca2+

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n', 's', 'Nai', 'ANai', 'Cai', 'ACai')

    def __init__(self):
        super().__init__()
        self.rates = self.getRatesNames(['m', 'h', 'n', 's'])
        self.K_Na = self.K_Na_original * self.surface * self.curr_factor
        self.K_Ca = self.K_Ca_original * self.surface * self.curr_factor

    # ------------------------------ Gating states kinetics ------------------------------

    def _xinf(self, Vm, halfmax, slope, power):
        ''' Generic function computing the steady-state open-probability of a
            particular ion channel gate at a given voltage.

            :param Vm: membrane potential (mV)
            :param halfmax: half-activation voltage (mV)
            :param slope: slope parameter of activation function (mV)
            :param power: power exponent multiplying the exponential expression (integer)
            :return: steady-state open-probability (-)
        '''
        return 1 / (1 + np.exp((Vm - halfmax) / slope))**power

    def _taux(self, Vm, halfmax, slope, tauMax, tauMin):
        ''' Generic function computing the voltage-dependent, adaptation time constant
            of a particular ion channel gate at a given voltage.

            :param Vm: membrane potential (mV)
            :param halfmax: voltage at which adaptation time constant is half-maximal (mV)
            :param slope: slope parameter of adaptation time constant function (mV)
            :return: adptation time constant (s)
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

    minf = partialmethod(_xinf, halfmax=-35.0, slope=-5.0, power=1)
    hinf = partialmethod(_xinf, halfmax=-50.0, slope=9.0, power=2)
    tauh = partialmethod(_taux, halfmax=-36.0, slope=3.5, tauMax=14.0e-3, tauMin=0.2e-3)
    ninf = partialmethod(_xinf, halfmax=-22.0, slope=-9.0, power=1)
    taun = partialmethod(_taux, halfmax=-10.0, slope=10.0, tauMax=6.0e-3, tauMin=1.0e-3)
    sinf = partialmethod(_xinf, halfmax=-10.0, slope=-2.8, power=1)

    # ------------------------------ States derivatives ------------------------------

    def derNai(self, Nai, m, h, Vm):
        ''' Evolution of submembrane Sodium concentration.

            :param Nai: submembrane Sodium concentration (M)
            :param m: open-probability of m-gate (-)
            :param h: open-probability of h-gate (-)
            :param Vm: membrane potential (mV)
            :return: time derivative of submembrane Sodium concentration (M/s)
        '''
        return self._derCion(Nai, self.iNa(m, h, Vm), self.K_Na, self.taur_Na)

    def derCai(self, Cai, s, Vm):
        ''' Evolution of submembrane Calcium concentration.

            :param Cai: submembrane Calcium concentration (M)
            :param s: open-probability of s-gate (-)
            :param Vm: membrane potential (mV)
            :return: time derivative of submembrane Calcium concentration (M/s)
        '''
        return self._derCion(Cai, self.iCa(s, Vm), self.K_Ca, self.taur_Ca)

    def derANa(self, ANa, Nai):
        ''' Evolution of Sodium pool-dependent activation function for iPumpNa. '''
        return self._derAion(ANa, Nai, self.taua_PumpNa)

    def derACa(self, ACa, Cai):
        ''' Evolution of Calcium pool-dependent activation function for iKCa. '''
        return self._derAion(ACa, Cai, self.taua_KCa)

    def derStates(self, Vm, states):
        return {
            'm': (self.minf(Vm) - states['m']) / self.taum,
            'h': (self.hinf(Vm) - states['h']) / self.tauh(Vm),
            'n': (self.ninf(Vm) - states['n']) / self.taun(Vm),
            's': (self.sinf(Vm) - states['s']) / self.taus,
            'Nai': self.derNai(states['Nai'], states['m'], states['h'], Vm),
            'ANai': self.derANa(states['ANai'], states['Nai']),
            'Cai': self.derCai(states['Cai'], states['s'], Vm),
            'ACai': self.derACa(states['ACai'], states['Cai'])
        }

    def derEffStates(self, Vm, states, rates):
        return {
            'm': rates['alpham'] * (1 - states['m']) - rates['betam'] * states['m'],
            'h': rates['alphah'] * (1 - states['h']) - rates['betah'] * states['h'],
            'n': rates['alphan'] * (1 - states['n']) - rates['betan'] * states['n'],
            's': rates['alphas'] * (1 - states['s']) - rates['betas'] * states['s'],
            'Nai': self.derNai(states['Nai'], states['m'], states['h'], Vm),
            'ANai': self.derANa(states['ANai'], states['Nai']),
            'Cai': self.derCai(states['Cai'], states['s'], Vm),
            'ACai': self.derACa(states['ACai'], states['Cai'])
        }

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
        # Standard gating dynamics: Solve the equation dx/dt = 0 at Vm for each x-state
        sstates = {
            'm': self.minf(Vm),
            'h': self.hinf(Vm),
            'n': self.ninf(Vm),
            's': self.sinf(Vm)
        }
        # pools concentrations and activation steady-states
        sstates['Nai'] = - self.K_Na * self.iNa(sstates['m'], sstates['h'], Vm)
        sstates['ANai'] = sstates['Nai']
        sstates['Cai'] = - self.K_Ca * self.iCa(sstates['s'], Vm)
        sstates['ACai'] = sstates['Cai']

        return sstates

    def quasiSteadyStates(self, lkp):
        # Standard gating dynamics
        qsstates = self.qsStates(lkp, ['m', 'h', 'n', 's'])

        # pool concentrations and activation steady-states
        qsstates['Nai'] = - self.K_Na * self.iNa(qsstates['m'], qsstates['h'], lkp['V'])
        qsstates['ANai'] = qsstates['Nai']
        qsstates['Cai'] = - self.K_Ca * self.iCa(qsstates['s'], lkp['V'])
        qsstates['ACai'] = qsstates['Cai']

        return qsstates

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm):
        ''' Sodium current

            :param m: open-probability of m-gate
            :param h: open-probability of h-gate
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
        return self.gKdbar * n**2 * (Vm - self.EK)

    def iCa(self, s, Vm):
        ''' Calcium current

            :param s: open-probability of s-gate (-)
            :param u: open-probability of u-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gCabar * s * (Vm - self.ECa)

    def iKCa(self, ACa, Vm):
        ''' Calcium-activated Potassium current

            :param ACa: Calcium pool-dependent gate opening (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gKCabar * ACa * (Vm - self.EK)

    def iPumpNa(self, ANa, Vm):
        ''' NaK-ATPase pump current

            :param ANa: Sodium pool-dependent gate opening (-)
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
        return {
            'iNa': self.iNa(states['m'], states['h'], Vm),
            'iKd': self.iKd(states['n'], Vm),
            'iCa': self.iCa(states['s'], Vm),
            'iLeak': self.iLeak(Vm),
            'iPumpNa': self.iPumpNa(states['ANai'], Vm),
            'iKCa': self.iKCa(states['ACai'], Vm)
        }  # mA/m2

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
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


class LeechMech(PointNeuron):
    ''' Generic leech neuron

        Reference:
        *Baccus, S.A. (1998). Synaptic facilitation by reflected action potentials: enhancement
        of transmission when nerve impulses reverse direction at axon branch points. Proc. Natl.
        Acad. Sci. U.S.A. 95, 8345–8350.*
    '''

    # ------------------------------ Biophysical parameters ------------------------------

    alphaC_sf = 1e-5  # Calcium activation rate constant scaling factor (M)
    betaC = 0.1e3     # beta rate for the open-probability of iKCa channels (s-1)
    T = 293.15        # Room temperature (K)

    # ------------------------------ Gating states kinetics ------------------------------

    def alpham(self, Vm):
        ''' Voltage-dependent activation rate of m-gate

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        alpha = -0.03 * (Vm + 28) / (np.exp(- (Vm + 28) / 15) - 1)  # ms-1
        return alpha * 1e3  # s-1

    def betam(self, Vm):
        ''' Voltage-dependent inactivation rate of m-gate

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        beta = 2.7 * np.exp(-(Vm + 53) / 18)  # ms-1
        return beta * 1e3  # s-1

    def alphah(self, Vm):
        ''' Voltage-dependent activation rate of h-gate

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        alpha = 0.045 * np.exp(-(Vm + 58) / 18)  # ms-1
        return alpha * 1e3  # s-1

    def betah(self, Vm):
        ''' Voltage-dependent inactivation rate of h-gate

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)

            .. warning:: the original paper contains an error (multiplication) in the
            expression of this rate constant, corrected in the mod file on ModelDB (division).
        '''
        beta = 0.72 / (np.exp(-(Vm + 23) / 14) + 1)  # ms-1
        return beta * 1e3  # s-1

    def alphan(self, Vm):
        ''' Voltage-dependent activation rate of n-gate

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        alpha = -0.024 * (Vm - 17) / (np.exp(-(Vm - 17) / 8) - 1)  # ms-1
        return alpha * 1e3  # s-1

    def betan(self, Vm):
        ''' Voltage-dependent inactivation rate of n-gate

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        beta = 0.2 * np.exp(-(Vm + 48) / 35)  # ms-1
        return beta * 1e3  # s-1

    def alphas(self, Vm):
        ''' Voltage-dependent activation rate of s-gate

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        alpha = -1.5 * (Vm - 20) / (np.exp(-(Vm - 20) / 5) - 1)  # ms-1
        return alpha * 1e3  # s-1

    def betas(self, Vm):
        ''' Voltage-dependent inactivation rate of s-gate

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''
        beta = 1.5 * np.exp(-(Vm + 25) / 10)  # ms-1
        return beta * 1e3  # s-1

    def alphaC(self, Cai):
        ''' Voltage-dependent activation rate of C-gate

            :param Cai: intracellular Calcium concentration (M)
            :return: rate constant (s-1)
        '''
        alpha = 0.1 * Cai / self.alphaC_sf  # ms-1
        return alpha * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    def derC(self, c, Cai):
        ''' Evolution of the c-gate open-probability

            :param c: open-probability of c-gate (-)
            :param Cai: intracellular Calcium concentration (M)
            :return: derivative of open-probability w.r.t. time (s-1)
        '''
        return self.alphaC(Cai) * (1 - c) - self.betaC * c

    def derStates(self, Vm, states):
        return {
            'm': self.alpham(Vm) * (1 - states['m']) - self.betam(Vm) * states['m'],
            'h': self.alphah(Vm) * (1 - states['h']) - self.betah(Vm) * states['h'],
            'n': self.alphan(Vm) * (1 - states['n']) - self.betan(Vm) * states['n'],
            's': self.alphas(Vm) * (1 - states['s']) - self.betas(Vm) * states['s'],
            'c': self.derC(states['c'], states['Cai'])
        }

    def derEffStates(self, Vm, states, rates):
        return {
            'm': rates['alpham'] * (1 - states['m']) - rates['betam'] * states['m'],
            'h': rates['alphah'] * (1 - states['h']) - rates['betah'] * states['h'],
            'n': rates['alphan'] * (1 - states['n']) - rates['betan'] * states['n'],
            's': rates['alphas'] * (1 - states['s']) - rates['betas'] * states['s'],
            'c': self.derC(states['c'], states['Cai']),
        }

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm, Nai):
        ''' Sodium current

            :param m: open-probability of m-gate (-)
            :param h: open-probability of h-gate (-)
            :param Vm: membrane potential (mV)
            :param Nai: intracellular Sodium concentration (M)
            :return: current per unit area (mA/m2)
        '''
        GNa = self.gNabar * m**4 * h
        ENa = self.nernst(Z_Na, Nai, self.Nao, self.T)  # mV
        return GNa * (Vm - ENa)

    def iKd(self, n, Vm):
        ''' Delayed-rectifier Potassium current

            :param n: open-probability of n-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        GK = self.gKdbar * n**2
        return GK * (Vm - self.EK)

    def iCa(self, s, Vm, Cai):
        ''' Calcium current

            :param s: open-probability of s-gate (-)
            :param Vm: membrane potential (mV)
            :param Cai: intracellular Calcium concentration (M)
            :return: current per unit area (mA/m2)
        '''
        GCa = self.gCabar * s
        ECa = self.nernst(Z_Ca, Cai, self.Cao, self.T)  # mV
        return GCa * (Vm - ECa)

    def iKCa(self, c, Vm):
        ''' Calcium-activated Potassium current

            :param c: open-probability of c-gate (-)
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

    def currents(self, Vm, states):
        return {
            'iNa': self.iNa(states['m'], states['h'], Vm, states['Nai']),
            'iKd': self.iKd(states['n'], Vm),
            'iCa': self.iCa(states['s'], Vm, states['Cai']),
            'iKCa': self.iKCa(states['c'], Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2


class LeechPressure(LeechMech):
    ''' Leech pressure sensory neuron

        Reference:
        *Baccus, S.A. (1998). Synaptic facilitation by reflected action potentials: enhancement
        of transmission when nerve impulses reverse direction at axon branch points. Proc. Natl.
        Acad. Sci. U.S.A. 95, 8345–8350.*
    '''

    # Neuron name
    name = 'LeechP'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2     # Membrane capacitance (F/m2)
    Vm0 = -48.865  # Membrane potential (mV)
    Nai0 = 0.01    # Intracellular Sodium concentration (M)
    Cai0 = 1e-7    # Intracellular Calcium concentration (M)

    # Reversal potentials (mV)
    # ENa = 60      # Sodium (from MOD file on ModelDB)
    # ECa = 125     # Calcium (from MOD file on ModelDB)
    EK = -68.0     # Potassium
    ELeak = -49.0  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 3500.0  # Sodium
    gKdbar = 60.0    # Delayed-rectifier Potassium
    gCabar = 0.02    # Calcium
    gKCabar = 8.0    # Calcium-dependent Potassium
    gLeak = 5.0      # Non-specific leakage

    # Ionic concentrations (M)
    Nao = 0.11    # Extracellular Sodium
    Cao = 1.8e-3  # Extracellular Calcium

    # Additional parameters
    INaPmax = 70.0    # Maximum pump rate of the NaK-ATPase (mA/m2)
    khalf_Na = 0.012  # Sodium concentration at which NaK-ATPase is at half its maximum rate (M)
    ksteep_Na = 1e-3  # Sensitivity of NaK-ATPase to varying Sodium concentrations (M)
    iCaS = 0.1        # Calcium pump current parameter (mA/m2)
    diam = 50e-6      # Cell soma diameter (m)

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n', 's', 'c', 'Nai', 'Cai')

    def __init__(self):
        ''' Constructor of the class. '''
        super().__init__()
        self.rates = self.getRatesNames(['m', 'h', 'n', 's'])

        # Surface to volume ratio of the (spherical) cell soma (m-1)
        SV_ratio = 6 / self.diam

        # Conversion constants from membrane ionic currents into
        # change rate of intracellular ionic concentrations (M/s)
        self.K_Na = SV_ratio / (Z_Na * FARADAY) * 1e-6  # Sodium
        self.K_Ca = SV_ratio / (Z_Ca * FARADAY) * 1e-6  # Calcium

    # ------------------------------ States derivatives ------------------------------

    def derStates(self, Vm, states):
        dstates = super().derStates(Vm, states)
        dstates.update({
            'Nai': -(self.iNa(states['m'], states['h'], Vm, states['Nai']) +
                     self.iPumpNa(states['Nai'])) * self.K_Na,
            'Cai': -(self.iCa(states['s'], Vm, states['Cai']) +
                     self.iPumpCa(states['Cai'])) * self.K_Ca
        })
        return dstates

    def derEffStates(self, Vm, states, rates):
        dstates = super().derEffStates(Vm, states, rates)
        dstates.update({
            'Nai': -(self.iNa(states['m'], states['h'], Vm, states['Nai']) +
                     self.iPumpNa(states['Nai'])) * self.K_Na,
            'Cai': -(self.iCa(states['s'], Vm, states['Cai']) +
                     self.iPumpCa(states['Cai'])) * self.K_Ca
        })
        return dstates

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
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

    def quasiSteadyStates(self, lkp):
        qsstates = self.qsStates(lkp, ['m', 'h', 'n', 's'])
        qsstates.update({
            'Nai': self.Nai0,
            'Cai': self.Cai0
        })
        qsstates['c'] = self.alphaC(qsstates['Cai']) / (self.alphaC(qsstates['Cai']) + self.betaC)
        return qsstates

    # ------------------------------ Membrane currents ------------------------------

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
        currents = super().currents(Vm, states)
        currents.update({
            'iPumpNa': self.iPumpNa(states['Nai']) / 3.,
            'iPumpCa': self.iPumpCa(states['Cai'])
        })  # mA/m2
        return currents

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
            'alphas': np.mean(self.alphas(Vm)),
            'betas': np.mean(self.betas(Vm))
        }


class LeechRetzius(LeechMech):
    ''' Leech Retzius neuron

        References:
        *Vazquez, Y., Mendez, B., Trueta, C., and De-Miguel, F.F. (2009). Summation of excitatory
        postsynaptic potentials in electrically-coupled neurones. Neuroscience 163, 202–212.*

        *ModelDB link: https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=120910*
    '''

    # Neuron name
    # name = 'LeechR'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 5e-2    # Membrane capacitance (F/m2)
    Vm0 = -44.45  # Membrane resting potential (mV)

    # Reversal potentials (mV)
    ENa = 50.0     # Sodium (from retztemp.ses file on ModelDB)
    EK = -79.0     # Potassium (from retztemp.ses file on ModelDB)
    ECa = 125.0    # Calcium (from cachdend.mod file on ModelDB)
    ELeak = -30.0  # Non-specific leakage (from leakdend.mod file on ModelDB)

    # Maximal channel conductances (S/m2)
    gNabar = 1250.0  # Sodium current
    gKdbar = 10.0    # Delayed-rectifier Potassium
    GAMax = 100.0    # Transient Potassium
    gCabar = 4.0     # Calcium current
    gKCabar = 130.0  # Calcium-dependent Potassium
    gLeak = 1.25     # Non-specific leakage

    # Ionic concentrations (M)
    Cai = 5e-8  # Intracellular Calcium (from retztemp.ses file)

    # Additional parameters
    Vhalf = -73.1  # half-activation voltage (mV)

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n', 's', 'c', 'a', 'b')

    # ------------------------------ Gating states kinetics ------------------------------

    def ainf(self, Vm):
        ''' Steady-state open-probability of a-gate.

            Source:
            *Beck, H., Ficker, E., and Heinemann, U. (1992). Properties of two voltage-activated
            potassium currents in acutely isolated juvenile rat dentate gyrus granule cells.
            J. Neurophysiol. 68, 2086–2099.*

            :param Vm: membrane potential (mV)
            :return: steady-state open-probability (-)
        '''
        Vth = -55.0  # mV
        return 0 if Vm <= Vth else min(1, 2 * (Vm - Vth)**3 / ((11 - Vth)**3 + (Vm - Vth)**3))

    def taua(self, Vm):
        ''' Adaptation time constant of a-gate (assuming T = 20°C).

            Source:
            *Beck, H., Ficker, E., and Heinemann, U. (1992). Properties of two voltage-activated
            potassium currents in acutely isolated juvenile rat dentate gyrus granule cells.
            J. Neurophysiol. 68, 2086–2099.*

            :param Vm: membrane potential (mV)
            :return: adaptation time constant (s)
        '''
        x = -1.5 * (Vm - self.Vhalf) * 1e-3 * FARADAY / (Rg * self.T)  # [-]
        alpha = np.exp(x)  # ms-1
        beta = np.exp(0.7 * x)  # ms-1
        return max(0.5, beta / (0.3 * (1 + alpha))) * 1e-3  # s

    def binf(self, Vm):
        ''' Steady-state open-probability of b-gate

            Source:
            *Beck, H., Ficker, E., and Heinemann, U. (1992). Properties of two voltage-activated
            potassium currents in acutely isolated juvenile rat dentate gyrus granule cells.
            J. Neurophysiol. 68, 2086–2099.*

            :param Vm: membrane potential (mV)
            :return: steady-state open-probability (-)
        '''
        return 1. / (1 + np.exp((self.Vhalf - Vm) / -6.3))

    def taub(self, Vm):
        ''' Adaptation time constant of b-gate (assuming T = 20°C).

            Source:
            *Beck, H., Ficker, E., and Heinemann, U. (1992). Properties of two voltage-activated
            potassium currents in acutely isolated juvenile rat dentate gyrus granule cells.
            J. Neurophysiol. 68, 2086–2099.*

            :param Vm: membrane potential (mV)
            :return: adaptation time constant (s)
        '''
        x = 2 * (Vm - self.Vhalf) * 1e-3 * FARADAY / (Rg * self.T)
        alpha = np.exp(x)
        beta = np.exp(0.65 * x)
        return max(7.5, beta / (0.02 * (1 + alpha))) * 1e-3  # s

    # ------------------------------ States derivatives ------------------------------

    def derStates(self, Vm, states):
        dstates = super().derStates(Vm, states)
        dstates.update({
            'a': (self.ainf(Vm) - states['a']) / self.taua(Vm),
            'b': (self.binf(Vm) - states['b']) / self.taub(Vm)
        })
        return dstates

    def derEffStates(self, Vm, states, rates):
        dstates = super().derStates(Vm, states, rates)
        dstates.update({
            'a': rates['alphaa'] * (1 - states['a']) - rates['betaa'] * states['a'],
            'b': rates['alphab'] * (1 - states['b']) - rates['betab'] * states['b']
        })
        return dstates

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
        return {
            'm': self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            's': self.alphas(Vm) / (self.alphas(Vm) + self.betas(Vm)),
            'c': self.alphaC(self.Cai) / (self.alphaC(self.Cai) + self.betaC),
            'a': self.ainf(Vm),
            'b': self.binf(Vm)
        }

    def quasiSteadyStates(self, lkp):
        qsstates = self.qsStates(lkp, ['m', 'h', 'n', 's', 'a', 'b'])
        qsstates['c'] = self.alphaC(self.Cai) / (self.alphaC(self.Cai) + self.betaC),
        return qsstates

    # ------------------------------ Membrane currents ------------------------------

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
        currents = super().currents(Vm, states)
        currents['iA'] = self.iA(states['a'], states['b'], Vm)  # mA/m2
        return currents

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
            'alphas': np.mean(self.alphas(Vm)),
            'betas': np.mean(self.betas(Vm)),
            'alphaa': np.mean(self.ainf(Vm) / self.taua(Vm)),
            'betaa': np.mean((1 - self.ainf(Vm)) / self.taua(Vm)),
            'alphab': np.mean(self.binf(Vm) / self.taub(Vm)),
            'betab': np.mean((1 - self.binf(Vm)) / self.taub(Vm))
        }
