# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-31 15:20:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-25 14:54:27

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

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        's': 'iCa gate',
        'Nai': 'submembrane Na+ concentration (arbitrary unit)',
        'ANa': 'Na+ dependent iPumpNa gate',
        'Cai': 'submembrane Ca2+ concentration (arbitrary unit)',
        'ACa': 'Ca2+ dependent iKCa gate'
    }

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
        ''' Evolution of submembrane Sodium concentration '''
        return self._derCion(Nai, self.iNa(m, h, Vm), self.K_Na, self.taur_Na)  # M/s

    def derCai(self, Cai, s, Vm):
        ''' Evolution of submembrane Calcium concentration '''
        return self._derCion(Cai, self.iCa(s, Vm), self.K_Ca, self.taur_Ca)  # M/s

    def derANa(self, ANa, Nai):
        ''' Evolution of Na+ dependent iPumpNa gate '''
        return self._derAion(ANa, Nai, self.taua_PumpNa)

    def derACa(self, ACa, Cai):
        ''' Evolution of Ca2+ dependent iKCa gate '''
        return self._derAion(ACa, Cai, self.taua_KCa)

    def derStates(self):
        return {
            'm': lambda Vm, x: (self.minf(Vm) - x['m']) / self.taum,
            'h': lambda Vm, x: (self.hinf(Vm) - x['h']) / self.tauh(Vm),
            'n': lambda Vm, x: (self.ninf(Vm) - x['n']) / self.taun(Vm),
            's': lambda Vm, x: (self.sinf(Vm) - x['s']) / self.taus,
            'Nai': lambda Vm, x: self.derNai(x['Nai'], x['m'], x['h'], Vm),
            'ANa': lambda Vm, x: self.derANa(x['ANa'], x['Nai']),
            'Cai': lambda Vm, x: self.derCai(x['Cai'], x['s'], Vm),
            'ACa': lambda Vm, x: self.derACa(x['ACa'], x['Cai'])
        }

    # ------------------------------ Steady states ------------------------------

    def Naiinf(self, Vm):
        ''' Steady.state Sodium intracellular concentration. '''
        return -self.K_Na * self.iNa(self.minf(Vm), self.hinf(Vm), Vm)

    def Caiinf(self, Vm):
        ''' Steady.state Calcium intracellular concentration. '''
        return -self.K_Ca * self.iCa(self.sinf(Vm), Vm)

    def steadyStates(self):
        return {
            'm': lambda Vm: self.minf(Vm),
            'h': lambda Vm: self.hinf(Vm),
            'n': lambda Vm: self.ninf(Vm),
            's': lambda Vm: self.sinf(Vm),
            'Nai': lambda Vm: self.Naiinf(Vm),
            'ANa': lambda Vm: self.Naiinf(Vm),
            'Cai': lambda Vm: self.Caiinf(Vm),
            'ACa': lambda Vm: self.Caiinf(Vm)
        }

    # def quasiSteadyStates(self, lkp):
    #     qsstates = self.qsStates(lkp, ['m', 'h', 'n', 's'])
    #     qsstates['Nai'] = - self.K_Na * self.iNa(qsstates['m'], qsstates['h'], lkp['V'])
    #     qsstates['ANa'] = qsstates['Nai']
    #     qsstates['Cai'] = - self.K_Ca * self.iCa(qsstates['s'], lkp['V'])
    #     qsstates['ACa'] = qsstates['Cai']
    #     return qsstates

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm):
        ''' Sodium current '''
        return self.gNabar * m**3 * h * (Vm - self.ENa)  # mA/m2

    def iKd(self, n, Vm):
        ''' Delayed-rectifier Potassium current '''
        return self.gKdbar * n**2 * (Vm - self.EK)  # mA/m2

    def iCa(self, s, Vm):
        ''' Calcium current '''
        return self.gCabar * s * (Vm - self.ECa)  # mA/m2

    def iKCa(self, ACa, Vm):
        ''' Calcium-activated Potassium current '''
        return self.gKCabar * ACa * (Vm - self.EK)  # mA/m2

    def iPumpNa(self, ANa, Vm):
        ''' NaK-ATPase pump current '''
        return self.gPumpNa * ANa * (Vm - self.EPumpNa)  # mA/m2

    def iLeak(self, Vm):
        ''' Non-specific leakage current '''
        return self.gLeak * (Vm - self.ELeak)  # mA/m2

    def currents(self):
        return {
            'iNa': lambda Vm, x: self.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: self.iKd(x['n'], Vm),
            'iCa': lambda Vm, x: self.iCa(x['s'], Vm),
            'iPumpNa': lambda Vm, x: self.iPumpNa(x['ANa'], Vm),
            'iKCa': lambda Vm, x: self.iKCa(x['ACa'], Vm),
            'iLeak': lambda Vm, _: self.iLeak(Vm)
        }

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
        return -0.03 * (Vm + 28) / (np.exp(- (Vm + 28) / 15) - 1) * 1e3  # s-1

    def betam(self, Vm):
        return 2.7 * np.exp(-(Vm + 53) / 18) * 1e3  # s-1

    def alphah(self, Vm):
        return 0.045 * np.exp(-(Vm + 58) / 18) * 1e3  # s-1

    def betah(self, Vm):
        ''' .. warning:: the original paper contains an error (multiplication) in the
            expression of this rate constant, corrected in the mod file on ModelDB (division).
        '''
        return 0.72 / (np.exp(-(Vm + 23) / 14) + 1) * 1e3  # s-1

    def alphan(self, Vm):
        return -0.024 * (Vm - 17) / (np.exp(-(Vm - 17) / 8) - 1) * 1e3  # s-1

    def betan(self, Vm):
        return 0.2 * np.exp(-(Vm + 48) / 35) * 1e3  # s-1

    def alphas(self, Vm):
        return -1.5 * (Vm - 20) / (np.exp(-(Vm - 20) / 5) - 1) * 1e3  # s-1

    def betas(self, Vm):
        return 1.5 * np.exp(-(Vm + 25) / 10) * 1e3  # s-1

    def alphaC(self, Cai):
        return 0.1 * Cai / self.alphaC_sf * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    def derC(self, c, Cai):
        ''' Evolution of the c-gate open-probability '''
        return self.alphaC(Cai) * (1 - c) - self.betaC * c  # s-1

    def derStates(self):
        return {
            'm': lambda Vm, x: self.alpham(Vm) * (1 - x['m']) - self.betam(Vm) * x['m'],
            'h': lambda Vm, x: self.alphah(Vm) * (1 - x['h']) - self.betah(Vm) * x['h'],
            'n': lambda Vm, x: self.alphan(Vm) * (1 - x['n']) - self.betan(Vm) * x['n'],
            's': lambda Vm, x: self.alphas(Vm) * (1 - x['s']) - self.betas(Vm) * x['s'],
            'c': lambda Vm, x: self.derC(x['c'], x['Cai'])
        }

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self):
        return {
            'm': lambda Vm: self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': lambda Vm: self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': lambda Vm: self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            's': lambda Vm: self.alphas(Vm) / (self.alphas(Vm) + self.betas(Vm)),
        }

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm, Nai):
        ''' Sodium current '''
        ENa = self.nernst(Z_Na, Nai, self.Nao, self.T)  # mV
        return self.gNabar * m**4 * h * (Vm - ENa)  # mA/m2

    def iKd(self, n, Vm):
        ''' Delayed-rectifier Potassium current '''
        return self.gKdbar * n**2 * (Vm - self.EK)  # mA/m2

    def iCa(self, s, Vm, Cai):
        ''' Calcium current '''
        ECa = self.nernst(Z_Ca, Cai, self.Cao, self.T)  # mV
        return self.gCabar * s * (Vm - ECa)  # mA/m2

    def iKCa(self, c, Vm):
        ''' Calcium-activated Potassium current '''
        return self.gKCabar * c * (Vm - self.EK)  # mA/m2

    def iLeak(self, Vm):
        ''' Non-specific leakage current '''
        return self.gLeak * (Vm - self.ELeak)  # mA/m2

    def currents(self):
        return {
            'iNa': lambda Vm, x: self.iNa(x['m'], x['h'], Vm, x['Nai']),
            'iKd': lambda Vm, x: self.iKd(x['n'], Vm),
            'iCa': lambda Vm, x: self.iCa(x['s'], Vm, x['Cai']),
            'iKCa': lambda Vm, x: self.iKCa(x['c'], Vm),
            'iLeak': lambda Vm, _: self.iLeak(Vm)
        }

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
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

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        's': 'iCa gate',
        'c': 'iKCa gate',
        'Nai': 'submembrane Na+ concentration (M)',
        'Cai': 'submembrane Ca2+ concentration (M)'
    }

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

    def derStates(self):
        return {**super().derStates(), **{
            'Nai': lambda Vm, x: -(self.iNa(x['m'], x['h'], Vm, x['Nai']) +
                                   self.iPumpNa(x['Nai'])) * self.K_Na,
            'Cai': lambda Vm, x: -(self.iCa(x['s'], Vm, x['Cai']) +
                                   self.iPumpCa(x['Cai'])) * self.K_Ca
        }}

    # ------------------------------ Steady states ------------------------------

    def cinf(self, Cai):
        return self.alphaC(Cai) / (self.alphaC(Cai) + self.betaC)

    def steadyStates(self):
        return {**super().steadyStates(), **{
            'Nai': lambda _: self.Nai0,
            'Cai': lambda _: self.Cai0,
            'c': lambda _: self.cinf(self.Cai0)
        }}

    # def quasiSteadyStates(self, lkp):
    #     qsstates = self.qsStates(lkp, ['m', 'h', 'n', 's'])
    #     qsstates.update({
    #         'Nai': self.Nai0,
    #         'Cai': self.Cai0
    #     })
    #     qsstates['c'] = self.alphaC(qsstates['Cai']) / (self.alphaC(qsstates['Cai']) + self.betaC)
    #     return qsstates

    # ------------------------------ Membrane currents ------------------------------

    def iPumpNa(self, Nai):
        ''' NaK-ATPase pump current '''
        return self.INaPmax / (1 + np.exp((self.khalf_Na - Nai) / self.ksteep_Na))  # mA/m2

    def iPumpCa(self, Cai):
        ''' Calcium pump current '''
        return self.iCaS * (Cai - self.Cai0) / 1.5  # mA/m2

    def currents(self):
        return {**super().currents(), **{
            'iPumpNa': lambda Vm, x: self.iPumpNa(x['Nai']) / 3.,
            'iPumpCa': lambda Vm, x: self.iPumpCa(x['Cai'])
        }}


class LeechRetzius(LeechMech):
    ''' Leech Retzius neuron

        References:
        *Vazquez, Y., Mendez, B., Trueta, C., and De-Miguel, F.F. (2009). Summation of excitatory
        postsynaptic potentials in electrically-coupled neurones. Neuroscience 163, 202–212.*

        *ModelDB link: https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=120910*

        iA current reference:
        *Beck, H., Ficker, E., and Heinemann, U. (1992). Properties of two voltage-activated
        potassium currents in acutely isolated juvenile rat dentate gyrus granule cells.
        J. Neurophysiol. 68, 2086–2099.*
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

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        's': 'iCa gate',
        'c': 'iKCa gate',
        'a': 'iA activation gate',
        'b': 'iA inactivation gate',
    }

    # ------------------------------ Gating states kinetics ------------------------------

    def ainf(self, Vm):
        Vth = -55.0  # mV
        return 0 if Vm <= Vth else min(1, 2 * (Vm - Vth)**3 / ((11 - Vth)**3 + (Vm - Vth)**3))

    def taua(self, Vm):
        x = -1.5 * (Vm - self.Vhalf) * 1e-3 * FARADAY / (Rg * self.T)  # [-]
        alpha = np.exp(x)  # ms-1
        beta = np.exp(0.7 * x)  # ms-1
        return max(0.5, beta / (0.3 * (1 + alpha))) * 1e-3  # s

    def binf(self, Vm):
        return 1. / (1 + np.exp((self.Vhalf - Vm) / -6.3))

    def taub(self, Vm):
        x = 2 * (Vm - self.Vhalf) * 1e-3 * FARADAY / (Rg * self.T)  # [-]
        alpha = np.exp(x)  # ms-1
        beta = np.exp(0.65 * x)  # ms-1
        return max(7.5, beta / (0.02 * (1 + alpha))) * 1e-3  # s

    # ------------------------------ States derivatives ------------------------------

    def derStates(self, Vm, states):
        return {**super().derStates(Vm, states), **{
            'a': lambda Vm, x: (self.ainf(Vm) - x['a']) / self.taua(Vm),
            'b': lambda Vm, x: (self.binf(Vm) - x['b']) / self.taub(Vm)
        }}

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self):
        return {**super().steadyStates(), **{
            'a': lambda Vm: self.ainf(Vm),
            'b': lambda Vm: self.binf(Vm)
        }}

    # def quasiSteadyStates(self, lkp):
    #     qsstates = self.qsStates(lkp, ['m', 'h', 'n', 's', 'a', 'b'])
    #     qsstates['c'] = self.alphaC(self.Cai) / (self.alphaC(self.Cai) + self.betaC),
    #     return qsstates

    # ------------------------------ Membrane currents ------------------------------

    def iA(self, a, b, Vm):
        ''' Transient Potassium current '''
        return self.GAMax * a * b * (Vm - self.EK)  # mA/m2

    def currents(self):
        return {**super().currents(), **{
            'iA': lambda Vm, x: self.iA(x['a'], x['b'], Vm)
        }}

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        return {**super().computeEffRates(Vm), **{
            'alphaa': np.mean(self.ainf(Vm) / self.taua(Vm)),
            'betaa': np.mean((1 - self.ainf(Vm)) / self.taua(Vm)),
            'alphab': np.mean(self.binf(Vm) / self.taub(Vm)),
            'betab': np.mean((1 - self.binf(Vm)) / self.taub(Vm))
        }}
