# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-31 15:20:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-18 15:34:56

import numpy as np
from ..core import PointNeuron
from ..constants import Z_Ca


class Thalamic(PointNeuron):
    ''' Generic thalamic neuron

        Reference:
        *Plaksin, M., Kimmel, E., and Shoham, S. (2016). Cell-Type-Selective Effects of
        Intramembrane Cavitation as a Unifying Theoretical Framework for Ultrasonic
        Neuromodulation. eNeuro 3.*
    '''

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2  # Membrane capacitance (F/m2)

    # Reversal potentials (mV)
    ENa = 50.0   # Sodium
    EK = -90.0   # Potassium
    ECa = 120.0  # Calcium

    # ------------------------------ Gating states kinetics ------------------------------

    def alpham(self, Vm):
        ''' Voltage-dependent activation rate of m-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        alpha = 0.32 * self.vtrap(13 - Vdiff, 4)  # ms-1
        return alpha * 1e3  # s-1

    def betam(self, Vm):
        ''' Voltage-dependent inactivation rate of m-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        beta = 0.28 * self.vtrap(Vdiff - 40, 5)  # ms-1
        return beta * 1e3  # s-1

    def alphah(self, Vm):
        ''' Voltage-dependent activation rate of h-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        alpha = (0.128 * np.exp(-(Vdiff - 17) / 18))  # ms-1
        return alpha * 1e3  # s-1

    def betah(self, Vm):
        ''' Voltage-dependent inactivation rate of h-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        beta = (4 / (1 + np.exp(-(Vdiff - 40) / 5)))  # ms-1
        return beta * 1e3  # s-1

    def alphan(self, Vm):
        ''' Voltage-dependent activation rate of n-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        alpha = 0.032 * self.vtrap(15 - Vdiff, 5)  # ms-1
        return alpha * 1e3  # s-1

    def betan(self, Vm):
        ''' Voltage-dependent inactivation rate of n-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        beta = (0.5 * np.exp(-(Vdiff - 10) / 40))  # ms-1
        return beta * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    def derStates(self, Vm, states):
        return {
            'm': self.alpham(Vm) * (1 - states['m']) - self.betam(Vm) * states['m'],
            'h': self.alphah(Vm) * (1 - states['h']) - self.betah(Vm) * states['h'],
            'n': self.alphan(Vm) * (1 - states['n']) - self.betan(Vm) * states['n'],
            's': (self.sinf(Vm) - states['s']) / self.taus(Vm),
            'u': (self.uinf(Vm) - states['u']) / self.tauu(Vm)
        }

    def derEffStates(self, Vm, states, rates):
        return {
            'm': rates['alpham'] * (1 - states['m']) - rates['betam'] * states['m'],
            'h': rates['alphah'] * (1 - states['h']) - rates['betah'] * states['h'],
            'n': rates['alphan'] * (1 - states['n']) - rates['betan'] * states['n'],
            's': rates['alphas'] * (1 - states['s']) - rates['betas'] * states['s'],
            'u': rates['alphau'] * (1 - states['u']) - rates['betau'] * states['u']
        }

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
        # Voltage-gated steady-states
        return {
            'm': self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            's': self.sinf(Vm),
            'u': self.uinf(Vm)
        }

    # ------------------------------ Membrane currents ------------------------------

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

    def iCaT(self, s, u, Vm):
        ''' low-threshold (Ts-type) Calcium current

            :param s: open-probability of s-gate (-)
            :param u: open-probability of u-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gCaTbar * s**2 * u * (Vm - self.ECa)

    def iLeak(self, Vm):
        ''' non-specific leakage current

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gLeak * (Vm - self.ELeak)

    def currents(self, Vm, states):
        m, h, n, s, u = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iKd': self.iKd(n, Vm),
            'iCaT': self.iCaT(s, u, Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2

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
            'alphas': np.mean(self.sinf(Vm) / self.taus(Vm)),
            'betas': np.mean((1 - self.sinf(Vm)) / self.taus(Vm)),
            'alphau': np.mean(self.uinf(Vm) / self.tauu(Vm)),
            'betau': np.mean((1 - self.uinf(Vm)) / self.tauu(Vm))
        }


class ThalamicRE(Thalamic):
    ''' Thalamic reticular neuron

        References:
        *Destexhe, A., Contreras, D., Steriade, M., Sejnowski, T.J., and Huguenard, J.R. (1996).
        In vivo, in vitro, and computational analysis of dendritic calcium currents in thalamic
        reticular neurons. J. Neurosci. 16, 169–185.*

        *Huguenard, J.R., and Prince, D.A. (1992). A novel T-type current underlies prolonged
        Ca(2+)-dependent burst firing in GABAergic neurons of rat thalamic reticular nucleus.
        J. Neurosci. 12, 3804–3817.*

    '''

    # Neuron name
    name = 'RE'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Vm0 = -89.5  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -90.0  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 2000.0  # Sodium
    gKdbar = 200.0   # Delayed-rectifier Potassium
    gCaTbar = 30.0   # Low-threshold Calcium
    gLeak = 0.5      # Non-specific leakage

    # Additional parameters
    VT = -67.0  # Spike threshold adjustment parameter (mV)

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n', 's', 'u')

    # ------------------------------ Gating states kinetics ------------------------------

    def sinf(self, Vm):
        ''' Voltage-dependent steady-state opening of s-gate

            :param Vm: membrane potential (mV)
            :return: steady-state opening (-)
        '''
        return 1.0 / (1.0 + np.exp(-(Vm + 52.0) / 7.4))  # prob

    def taus(self, Vm):
        ''' Voltage-dependent adaptation time for adaptation of s-gate

            :param Vm: membrane potential (mV)
            :return: adaptation time (s)
        '''
        return (1 + 0.33 / (np.exp((Vm + 27.0) / 10.0) + np.exp(-(Vm + 102.0) / 15.0))) * 1e-3  # s

    def uinf(self, Vm):
        ''' Voltage-dependent steady-state opening of u-gate

            :param Vm: membrane potential (mV)
            :return: steady-state opening (-)
        '''
        return 1.0 / (1.0 + np.exp((Vm + 80.0) / 5.0))  # prob

    def tauu(self, Vm):
        ''' Voltage-dependent adaptation time for adaptation of u-gate

            :param Vm: membrane potential (mV)
            :return: adaptation time (s)
        '''
        return (28.3 + 0.33 / (np.exp((Vm + 48.0) / 4.0) + np.exp(-(Vm + 407.0) / 50.0))) * 1e-3  # s


class ThalamoCortical(Thalamic):
    ''' Thalamo-cortical neuron

        References:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac, Y.,
        Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for different
        classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*
        *Destexhe, A., Bal, T., McCormick, D.A., and Sejnowski, T.J. (1996). Ionic mechanisms
        underlying synchronized oscillations and propagating waves in a model of ferret
        thalamic slices. J. Neurophysiol. 76, 2049–2070.*
        *McCormick, D.A., and Huguenard, J.R. (1992). A model of the electrophysiological
        properties of thalamocortical relay neurons. J. Neurophysiol. 68, 1384–1400.*
    '''

    # Neuron name
    name = 'TC'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    # Vm0 = -63.4  # Membrane potential (mV)
    Vm0 = -61.93  # Membrane potential (mV)

    # Reversal potentials (mV)
    EH = -40.0     # Mixed cationic current
    ELeak = -70.0  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 900.0  # Sodium
    gKdbar = 100.0  # Delayed-rectifier Potassium
    gCaTbar = 20.0  # Low-threshold Calcium
    gKLeak = 0.138  # Leakage Potassium
    gHbar = 0.175   # Mixed cationic current
    gLeak = 0.1     # Non-specific leakage

    # Additional parameters
    VT = -52.0       # Spike threshold adjustment parameter (mV)
    Vx = 0.0         # Voltage-dependence uniform shift factor at 36°C (mV)
    taur_Cai = 5e-3  # decay time constant for intracellular Ca2+ dissolution (s)
    Cai_min = 50e-9  # minimal intracellular Calcium concentration (M)
    deff = 100e-9    # effective depth beneath membrane for intracellular [Ca2+] calculation
    nCa = 4          # number of Calcium binding sites on regulating factor
    k1 = 2.5e22      # intracellular Ca2+ regulation factor (M-4 s-1)
    k2 = 0.4         # intracellular Ca2+ regulation factor (s-1)
    k3 = 100.0       # intracellular Ca2+ regulation factor (s-1)
    k4 = 1.0         # intracellular Ca2+ regulation factor (s-1)

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n', 's', 'u', 'O', 'C', 'P0', 'Cai')

    def __init__(self):
        super().__init__()
        self.rates = self.getRatesNames(['m', 'h', 'n', 's', 'u', 'O'])
        self.iCa_to_Cai_rate = self.currentToConcentrationRate(Z_Ca, self.deff)
        # self.states += ['O', 'C', 'P0', 'Cai']

    def getPltScheme(self):
        pltscheme = super().getPltScheme()
        pltscheme['i_{H}\\ kin.'] = ['O', 'OL', 'P0']
        pltscheme['[Ca^{2+}]_i'] = ['Cai']
        return pltscheme

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        pltvars = super().getPltVars(wrapleft, wrapright)
        pltvars.update({
            'Cai': {
                'desc': 'sumbmembrane Ca2+ concentration',
                'label': '[Ca^{2+}]_i',
                'unit': 'uM',
                'factor': 1e6
            },
            'OL': {
                'desc': 'iH O-gate locked-opening',
                'label': 'O_L',
                'bounds': (-0.1, 1.1),
                'func': 'OL({0}O{1}, {0}C{1})'.format(wrapleft, wrapright)
            },
            'P0': {
                'desc': 'iH regulating factor activation',
                'label': 'P_0',
                'bounds': (-0.1, 1.1)
            }
        })
        return pltvars

    # ------------------------------ Gating states kinetics ------------------------------

    def sinf(self, Vm):
        ''' Voltage-dependent steady-state opening of s-gate

            :param Vm: membrane potential (mV)
            :return: steady-state opening (-)
        '''
        return 1.0 / (1.0 + np.exp(-(Vm + self.Vx + 57.0) / 6.2))  # prob

    def taus(self, Vm):
        ''' Voltage-dependent adaptation time for adaptation of s-gate

            :param Vm: membrane potential (mV)
            :return: adaptation time (s)
        '''
        x = np.exp(-(Vm + self.Vx + 132.0) / 16.7) + np.exp((Vm + self.Vx + 16.8) / 18.2)
        return 1.0 / 3.7 * (0.612 + 1.0 / x) * 1e-3  # s

    def uinf(self, Vm):
        ''' Voltage-dependent steady-state opening of u-gate

            :param Vm: membrane potential (mV)
            :return: steady-state opening (-)
        '''
        return 1.0 / (1.0 + np.exp((Vm + self.Vx + 81.0) / 4.0))  # prob

    def tauu(self, Vm):
        ''' Voltage-dependent adaptation time for adaptation of u-gate

            :param Vm: membrane potential (mV)
            :return: adaptation time (s)
        '''
        if Vm + self.Vx < -80.0:
            return 1.0 / 3.7 * np.exp((Vm + self.Vx + 467.0) / 66.6) * 1e-3  # s
        else:
            return 1 / 3.7 * (np.exp(-(Vm + self.Vx + 22) / 10.5) + 28.0) * 1e-3  # s

    def oinf(self, Vm):
        ''' Voltage-dependent steady-state opening of O-gate

            :param Vm: membrane potential (mV)
            :return: steady-state opening (-)
        '''
        return 1.0 / (1.0 + np.exp((Vm + 75.0) / 5.5))

    def tauo(self, Vm):
        ''' Voltage-dependent adaptation time for adaptation of O-gate

            :param Vm: membrane potential (mV)
            :return: adaptation time (s)
        '''
        return 1 / (np.exp(-14.59 - 0.086 * Vm) + np.exp(-1.87 + 0.0701 * Vm)) * 1e-3

    def alphao(self, Vm):
        ''' Voltage-dependent transition rate between closed and open forms of O-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        return self.oinf(Vm) / self.tauo(Vm)

    def betao(self, Vm):
        ''' Voltage-dependent transition rate between open and closed forms of O-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        return (1 - self.oinf(Vm)) / self.tauo(Vm)

    # ------------------------------ States derivatives ------------------------------

    def derS(self, Vm, s):
        ''' Evolution of s-gate open-probability

            :param Vm: membrane potential (mV)
            :param s: open-probability of s-gate (-)
            :return: time derivative of s-gate open-probability (s-1)
        '''
        return (self.sinf(Vm) - s) / self.taus(Vm)

    def derU(self, Vm, u):
        ''' Evolution of u-gate open-probability

            :param Vm: membrane potential (mV)
            :param u: open-probability of u-gate (-)
            :return: time derivative of u-gate open-probability (s-1)
        '''
        return (self.uinf(Vm) - u) / self.tauu(Vm)

    def derC(self, C, O, Vm):
        ''' Evolution of O-gate closed-probability

            :param C: closed-probability of O-gate (-)
            :param O: open-probability of O-gate (-)
            :param Vm: membrane potential (mV)
            :return: time derivative of O-gate closed-probability (s-1)
        '''
        return self.betao(Vm) * O - self.alphao(Vm) * C

    def derO(self, C, O, P0, Vm):
        ''' Evolution of O-gate open-probability

            :param C: closed-probability of O-gate (-)
            :param O: open-probability of O-gate (-)
            :param P0: proportion of Ih channels regulating factor in unbound state (-)
            :param Vm: membrane potential (mV)
            :return: time derivative of O-gate open-probability (s-1)
        '''
        return - self.derC(C, O, Vm) - self.k3 * O * (1 - P0) + self.k4 * (1 - O - C)

    def OL(self, O, C):
        ''' O-gate locked-open probability.

            :param O: open-probability of O-gate (-)
            :param C: closed-probability of O-gate (-)
            :return: loked-open-probability of O-gate (-)

        '''
        return 1 - O - C

    def derP0(self, P0, Cai):
        ''' Evolution of unbound probability of Ih regulating factor.

            :param P0: unbound probability of Ih regulating factor (-)
            :param Cai: submembrane Calcium concentration (M)
            :return: time derivative of ubnound probability (s-1)
        '''
        return self.k2 * (1 - P0) - self.k1 * P0 * Cai**self.nCa

    def derCai(self, Cai, s, u, Vm):
        ''' Evolution of submembrane Calcium concentration.

            Model of Ca2+ buffering and contribution from iCaT derived from:
            *McCormick, D.A., and Huguenard, J.R. (1992). A model of the electrophysiological
            properties of thalamocortical relay neurons. J. Neurophysiol. 68, 1384–1400.*

            :param Cai: submembrane Calcium concentration (M)
            :param s: open-probability of s-gate (-)
            :param u: open-probability of u-gate (-)
            :param Vm: membrane potential (mV)
            :return: time derivative of submembrane Calcium concentration (M/s)
        '''
        return (self.Cai_min - Cai) / self.taur_Cai - self.iCa_to_Cai_rate * self.iCaT(s, u, Vm)

    def derStates(self, Vm, states):
        dstates = super().derStates(Vm, states)
        dstates.update({
            'O': self.derO(states['C'], states['O'], states['P0'], Vm),
            'C': self.derC(states['C'], states['O'], Vm),
            'P0': self.derP0(states['P0'], states['Cai']),
            'Cai': self.derCai(states['Cai'], states['s'], states['u'], Vm)
        })
        return dstates

    def derEffStates(self, Vm, states, rates):
        dstates = super().derEffStates(Vm, states, rates)
        dstates['C'] = rates['betao'] * states['O'] - rates['alphao'] * states['C']
        dstates['O'] = (- dstates['C'] - self.k3 * states['O'] * (1 - states['P0']) +
                        self.k4 * (1 - states['O'] - states['C']))
        dstates['P0'] = self.derP0(states['P0'], states['Cai'])
        dstates['Cai'] = self.derCai(states['Cai'], states['s'], states['u'], Vm)
        return dstates

    # ------------------------------ Steady states ------------------------------

    def Caiinf(self, Vm, s, u):
        ''' Find the steady-state intracellular Calcium concentration for a
            specific membrane potential and voltage-gated channel states.

            :param Vm: membrane potential (mV)
            :param s: open-probability of s-gate
            :param u: open-probability of u-gate
            :return: steady-state Calcium concentration in submembrane space (M)
        '''
        return self.Cai_min - self.taur_Cai * self.iCa_to_Cai_rate * self.iCaT(s, u, Vm)

    def P0inf(self, Cai):
        ''' Find the steady-state unbound probability of Ih regulating factor
            for a specific intracellular Calcium concentration.

            :param Cai : Calcium concentration in submembrane space (M)
            :return: steady-state unbound probability of Ih regulating factor
        '''
        return self.k2 / (self.k2 + self.k1 * Cai**self.nCa)

    def Oinf(self, Cai, Vm):
        ''' Find the steady-state O-gate open-probability for specific
            membrane potential and intracellular Calcium concentration.

            :param Cai : Calcium concentration in submembrane space (M)
            :param Vm: membrane potential (mV)
            :return: steady-state O-gate open-probability
        '''
        BA = self.betao(Vm) / self.alphao(Vm)
        return self.k4 / (self.k3 * (1 - self.P0inf(Cai)) + self.k4 * (1 + BA))

    def Cinf(self, Cai, Vm):
        ''' Find the steady-state O-gate closed-probability for specific
            membrane potential and intracellular Calcium concentration.

            :param Cai : Calcium concentration in submembrane space (M)
            :param Vm: membrane potential (mV)
            :return: steady-state O-gate closed-probability
        '''
        BA = self.betao(Vm) / self.alphao(Vm)
        return BA * self.Oinf(Cai, Vm)

    def steadyStates(self, Vm):
        # Voltage-gated steady-states
        sstates = super().steadyStates(Vm)

        # Other steady-states
        sstates['Cai'] = self.Caiinf(Vm, sstates['s'], sstates['u'])
        sstates['P0'] = self.P0inf(sstates['Cai'])
        sstates['O'] = self.Oinf(sstates['Cai'], Vm)
        sstates['C'] = self.Cinf(sstates['Cai'], Vm)

        return sstates

    def quasiSteadyStates(self, lkp):
        qsstates = super().quasiSteadyStates(lkp)
        qsstates['Cai'] = self.Caiinf(lkp['V'], qsstates['s'], qsstates['u'])
        qsstates['P0'] = self.P0inf(qsstates['Cai'])
        qsstates['O'] = self.Oinf(qsstates['Cai'], lkp['V'])
        qsstates['C'] = self.Cinf(qsstates['Cai'], lkp['V'])

        return qsstates

    # ------------------------------ Membrane currents ------------------------------

    def iKLeak(self, Vm):
        ''' Potassium leakage current

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gKLeak * (Vm - self.EK)

    def iH(self, O, C, Vm):
        ''' outward mixed cationic current

            :param C: closed-probability of O-gate (-)
            :param O: open-probability of O-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gHbar * (O + 2 * self.OL(O, C)) * (Vm - self.EH)

    def currents(self, Vm, states):
        m, h, n, s, u, O, C, _, _ = states
        currents = super().currents(Vm, [m, h, n, s, u])
        currents['iKLeak'] = self.iKLeak(Vm)  # mA/m2
        currents['iH'] = self.iH(O, C, Vm)  # mA/m2
        return currents

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        # Compute effective coefficients for Sodium, Potassium and Calcium conductances
        effrates = super().computeEffRates(Vm)

        # Compute effective coefficients for Ih conductance
        effrates['alphao'] = np.mean(self.alphao(Vm))
        effrates['betao'] = np.mean(self.betao(Vm))

        return effrates
