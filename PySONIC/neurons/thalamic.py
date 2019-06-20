# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-31 15:20:54
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-20 10:03:59

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
        return 0.32 * self.vtrap(13 - (Vm - self.VT), 4) * 1e3  # s-1

    def betam(self, Vm):
        return 0.28 * self.vtrap((Vm - self.VT) - 40, 5) * 1e3  # s-1

    def alphah(self, Vm):
        return 0.128 * np.exp(-((Vm - self.VT) - 17) / 18) * 1e3  # s-1

    def betah(self, Vm):
        return 4 / (1 + np.exp(-((Vm - self.VT) - 40) / 5)) * 1e3  # s-1

    def alphan(self, Vm):
        return 0.032 * self.vtrap(15 - (Vm - self.VT), 5) * 1e3  # s-1

    def betan(self, Vm):
        return 0.5 * np.exp(-((Vm - self.VT) - 10) / 40) * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    def derStates(self):
        return {
            'm': lambda Vm, s: self.alpham(Vm) * (1 - s['m']) - self.betam(Vm) * s['m'],
            'h': lambda Vm, s: self.alphah(Vm) * (1 - s['h']) - self.betah(Vm) * s['h'],
            'n': lambda Vm, s: self.alphan(Vm) * (1 - s['n']) - self.betan(Vm) * s['n'],
            's': lambda Vm, s: (self.sinf(Vm) - s['s']) / self.taus(Vm),
            'u': lambda Vm, s: (self.uinf(Vm) - s['u']) / self.tauu(Vm)
        }

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self):
        return {
            'm': lambda Vm: self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': lambda Vm: self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': lambda Vm: self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            's': lambda Vm: self.sinf(Vm),
            'u': lambda Vm: self.uinf(Vm)
        }

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm):
        ''' Sodium current '''
        return self.gNabar * m**3 * h * (Vm - self.ENa)  # mA/m2

    def iKd(self, n, Vm):
        ''' delayed-rectifier Potassium current '''
        return self.gKdbar * n**4 * (Vm - self.EK)

    def iCaT(self, s, u, Vm):
        ''' low-threshold (Ts-type) Calcium current '''
        return self.gCaTbar * s**2 * u * (Vm - self.ECa)  # mA/m2

    def iLeak(self, Vm):
        ''' non-specific leakage current '''
        return self.gLeak * (Vm - self.ELeak)  # mA/m2

    def currents(self):
        return {
            'iNa': lambda Vm, states: self.iNa(states['m'], states['h'], Vm),
            'iKd': lambda Vm, states: self.iKd(states['n'], Vm),
            'iCaT': lambda Vm, states: self.iCaT(states['s'], states['u'], Vm),
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

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        's': 'iCaT activation gate',
        'u': 'iCaT inactivation gate'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    def sinf(self, Vm):
        return 1.0 / (1.0 + np.exp(-(Vm + 52.0) / 7.4))

    def taus(self, Vm):
        return (1 + 0.33 / (np.exp((Vm + 27.0) / 10.0) + np.exp(-(Vm + 102.0) / 15.0))) * 1e-3  # s

    def uinf(self, Vm):
        return 1.0 / (1.0 + np.exp((Vm + 80.0) / 5.0))

    def tauu(self, Vm):
        return (28.3 + 0.33 / (
            np.exp((Vm + 48.0) / 4.0) + np.exp(-(Vm + 407.0) / 50.0))) * 1e-3  # s


class ThalamoCortical(Thalamic):
    ''' Thalamo-cortical neuron

        References:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac, Y.,
        Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for different
        classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

        *Destexhe, A., Bal, T., McCormick, D.A., and Sejnowski, T.J. (1996). Ionic mechanisms
        underlying synchronized oscillations and propagating waves in a model of ferret
        thalamic slices. J. Neurophysiol. 76, 2049–2070.*

        Model of Ca2+ buffering and contribution from iCaT derived from:
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

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        's': 'iCaT activation gate',
        'u': 'iCaT inactivation gate',
        'C': 'iH gate closed state',
        'O': 'iH gate open state',
        'Cai': 'submembrane Ca2+ concentration (M)',
        'P0': 'proportion of unbound iH regulating factor',
    }

    def __init__(self):
        super().__init__()
        self.rates = self.getRatesNames(['m', 'h', 'n', 's', 'u', 'O'])
        self.iCa_to_Cai_rate = self.currentToConcentrationRate(Z_Ca, self.deff)
        # self.states += ['O', 'C', 'P0', 'Cai']

    def OL(self, O, C):
        ''' O-gate locked-open probability '''
        return 1 - O - C

    def getPltScheme(self):
        pltscheme = super().getPltScheme()
        pltscheme['i_{H}\\ kin.'] = ['O', 'OL', 'P0']
        pltscheme['[Ca^{2+}]_i'] = ['Cai']
        return pltscheme

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        return {**super().getPltVars(wrapleft, wrapright), **{
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
        }}

    # ------------------------------ Gating states kinetics ------------------------------

    def sinf(self, Vm):
        return 1.0 / (1.0 + np.exp(-(Vm + self.Vx + 57.0) / 6.2))

    def taus(self, Vm):
        x = np.exp(-(Vm + self.Vx + 132.0) / 16.7) + np.exp((Vm + self.Vx + 16.8) / 18.2)
        return 1.0 / 3.7 * (0.612 + 1.0 / x) * 1e-3  # s

    def uinf(self, Vm):
        return 1.0 / (1.0 + np.exp((Vm + self.Vx + 81.0) / 4.0))

    def tauu(self, Vm):
        if Vm + self.Vx < -80.0:
            return 1.0 / 3.7 * np.exp((Vm + self.Vx + 467.0) / 66.6) * 1e-3  # s
        else:
            return 1 / 3.7 * (np.exp(-(Vm + self.Vx + 22) / 10.5) + 28.0) * 1e-3  # s

    def oinf(self, Vm):
        return 1.0 / (1.0 + np.exp((Vm + 75.0) / 5.5))

    def tauo(self, Vm):
        return 1 / (np.exp(-14.59 - 0.086 * Vm) + np.exp(-1.87 + 0.0701 * Vm)) * 1e-3  # s

    def alphao(self, Vm):
        return self.oinf(Vm) / self.tauo(Vm)  # s-1

    def betao(self, Vm):
        return (1 - self.oinf(Vm)) / self.tauo(Vm)  # s-1

    # ------------------------------ States derivatives ------------------------------

    def derCai(self, Cai, s, u, Vm):
        return (self.Cai_min - Cai) / self.taur_Cai -\
            self.iCa_to_Cai_rate * self.iCaT(s, u, Vm)  # M/s

    def derC(self, O, C, Vm):
        return self.betao(Vm) * O - self.alphao(Vm) * C  # s-1

    def derO(self, O, C, P0, Vm):
        return -self.derC(O, C, Vm) - self.k3 * O * (1 - P0) + self.k4 * (1 - O - C)  # s-1

    def derStates(self):
        return {**super().derStates(), **{
            'C': lambda Vm, s: self.derC(s['O'], s['C'], Vm),
            'O': lambda Vm, s: self.derO(s['O'], s['C'], s['P0'], Vm),
            'P0': lambda Vm, s: self.k2 * (1 - s['P0']) - self.k1 * s['P0'] * s['Cai']**self.nCa,
            'Cai': lambda Vm, s: self.derCai(s['Cai'], s['s'], s['u'], Vm),
        }}

    # def derEffStates(self, Vm, states, rates):
    #     dstates = super().derEffStates(Vm, states, rates)
    #     states.update({
    #         'C': rates['betao'] * states['O'] - rates['alphao'] * states['C'],
    #         'P0': self.k2 * (1 - states['P0']) - self.k1 * states['P0'] * states['Cai']**self.nCa,
    #         'Cai': self.derCai(states['Cai'], states['s'], states['u'], Vm)
    #     })
    #     dstates['O'] = -dstates['C'] + self.tmp(states['C'], states['O'], states['P0'])
    #     return dstates

    # ------------------------------ Steady states ------------------------------

    def Cinf(self, Cai, Vm):
        ''' Steady-state O-gate closed-probability '''
        return self.betao(Vm) / self.alphao(Vm) * self.Oinf(Cai, Vm)

    def Oinf(self, Cai, Vm):
        ''' Steady-state O-gate open-probability '''
        return self.k4 / (self.k3 * (1 - self.P0inf(Cai)) + self.k4 * (
            1 + self.betao(Vm) / self.alphao(Vm)))

    def P0inf(self, Cai):
        ''' Steady-state unbound probability of Ih regulating factor '''
        return self.k2 / (self.k2 + self.k1 * Cai**self.nCa)

    def Caiinf(self, Vm):
        ''' Steady-state intracellular Calcium concentration '''
        return self.Cai_min - self.taur_Cai * self.iCa_to_Cai_rate * self.iCaT(
            self.sinf(Vm), self.uinf(Vm), Vm)  # M

    def steadyStates(self):
        return {**super().steadyStates(), **{
            'O': lambda Vm: self.Oinf(self.Caiinf(Vm), Vm),
            'C': lambda Vm: self.Cinf(self.Caiinf(Vm), Vm),
            'P0': lambda Vm: self.P0inf(self.Caiinf(Vm)),
            'Cai': lambda Vm: self.Caiinf(Vm)
        }}

    # def quasiSteadyStates(self, lkp):
    #     qsstates = super().quasiSteadyStates(lkp)
    #     qsstates.update({
    #         'O': self.Oinf(self.Caiinf(lkp['V']), lkp['V']),
    #         'C': self.Cinf(lkp['V']),
    #         'P0': self.P0inf(self.Caiinf(lkp['V'])),
    #         'Cai': self.Caiinf(lkp['V'])
    #     })
    #     return qsstates

    # ------------------------------ Membrane currents ------------------------------

    def iKLeak(self, Vm):
        ''' Potassium leakage current '''
        return self.gKLeak * (Vm - self.EK)  # mA/m2

    def iH(self, O, C, Vm):
        ''' outward mixed cationic current '''
        return self.gHbar * (O + 2 * self.OL(O, C)) * (Vm - self.EH)  # mA/m2

    def currents(self):
        return {**super().currents(), **{
            'iKLeak': lambda Vm, states: self.iKLeak(Vm),
            'iH': lambda Vm, states: self.iH(states['O'], states['C'], Vm)
        }}

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        return {**super().computeEffRates(Vm), **{
            'alphao': np.mean(self.alphao(Vm)),
            'betao': np.mean(self.betao(Vm))
        }}
