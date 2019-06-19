# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-31 15:19:51
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-19 09:46:54

import numpy as np
from ..core import PointNeuron


class Cortical(PointNeuron):
    ''' Generic cortical neuron

        Reference:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*
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

    def pinf(self, Vm):
        return 1.0 / (1 + np.exp(-(Vm + 35) / 10))

    def taup(self, Vm):
        return self.TauMax / (3.3 * np.exp((Vm + 35) / 20) + np.exp(-(Vm + 35) / 20))  # s

    # ------------------------------ States derivatives ------------------------------

    def derStates(self, Vm, states):
        return {
            'm': self.alpham(Vm) * (1 - states['m']) - self.betam(Vm) * states['m'],
            'h': self.alphah(Vm) * (1 - states['h']) - self.betah(Vm) * states['h'],
            'n': self.alphan(Vm) * (1 - states['n']) - self.betan(Vm) * states['n'],
            'p': (self.pinf(Vm) - states['p']) / self.taup(Vm)
        }

    def derEffStates(self, Vm, states, rates):
        return {
            'm': rates['alpham'] * (1 - states['m']) - rates['betam'] * states['m'],
            'h': rates['alphah'] * (1 - states['h']) - rates['betah'] * states['h'],
            'n': rates['alphan'] * (1 - states['n']) - rates['betan'] * states['n'],
            'p': rates['alphap'] * (1 - states['p']) - rates['betap'] * states['p']
        }

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
        return {
            'm': self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            'p': self.pinf(Vm)
        }

    # ------------------------------ Membrane currents ------------------------------

    def iNa(self, m, h, Vm):
        ''' Sodium current '''
        return self.gNabar * m**3 * h * (Vm - self.ENa)  # mA/m2

    def iKd(self, n, Vm):
        ''' delayed-rectifier Potassium current '''
        return self.gKdbar * n**4 * (Vm - self.EK)  # mA/m2

    def iM(self, p, Vm):
        ''' slow non-inactivating Potassium current '''
        return self.gMbar * p * (Vm - self.EK)  # mA/m2

    def iLeak(self, Vm):
        ''' non-specific leakage current '''
        return self.gLeak * (Vm - self.ELeak)  # mA/m2

    def currents(self):
        return {
            'iNa': lambda Vm, states: self.iNa(states['m'], states['h'], Vm),
            'iKd': lambda Vm, states: self.iKd(states['n'], Vm),
            'iM': lambda Vm, states: self.iM(states['p'], Vm),
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
            'alphap': np.mean(self.pinf(Vm) / self.taup(Vm)),
            'betap': np.mean((1 - self.pinf(Vm)) / self.taup(Vm))
        }


class CorticalRS(Cortical):
    ''' Cortical regular spiking neuron

        Reference:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*
    '''

    # Neuron name
    name = 'RS'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Vm0 = -71.9  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -70.3  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 560.0  # Sodium
    gKdbar = 60.0   # Delayed-rectifier Potassium
    gMbar = 0.75    # Slow non-inactivating Potassium
    gLeak = 0.205   # Non-specific leakage

    # Additional parameters
    VT = -56.2      # Spike threshold adjustment parameter (mV)
    TauMax = 0.608  # Max. adaptation decay of slow non-inactivating Potassium current (s)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'p': 'iM gate'
    }


class CorticalFS(Cortical):
    ''' Cortical fast-spiking neuron

        Reference:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*
    '''

    # Neuron name
    name = 'FS'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Vm0 = -71.4  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -70.4  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 580.0  # Sodium
    gKdbar = 39.0   # Delayed-rectifier Potassium
    gMbar = 0.787   # Slow non-inactivating Potassium
    gLeak = 0.38    # Non-specific leakage

    # Additional parameters
    VT = -57.9      # Spike threshold adjustment parameter (mV)
    TauMax = 0.502  # Max. adaptation decay of slow non-inactivating Potassium current (s)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'p': 'iM gate'
    }


class CorticalLTS(Cortical):
    ''' Cortical low-threshold spiking neuron

        References:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

        *Huguenard, J.R., and McCormick, D.A. (1992). Simulation of the currents involved in
        rhythmic oscillations in thalamic relay neurons. J. Neurophysiol. 68, 1373–1383.*

    '''

    # Neuron name
    name = 'LTS'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Vm0 = -54.0  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -50.0  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 500.0  # Sodium
    gKdbar = 40.0   # Delayed-rectifier Potassium
    gMbar = 0.28    # Slow non-inactivating Potassium
    gCaTbar = 4.0   # Low-threshold Calcium
    gLeak = 0.19    # Non-specific leakage

    # Additional parameters
    VT = -50.0    # Spike threshold adjustment parameter (mV)
    TauMax = 4.0  # Max. adaptation decay of slow non-inactivating Potassium current (s)
    Vx = -7.0     # Voltage-dependence uniform shift factor at 36°C (mV)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'p': 'iM gate',
        's': 'iCaT activation gate',
        'u': 'iCaT inactivation gate'
    }

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
            return 1.0 / 3.7 * (np.exp(-(Vm + self.Vx + 22) / 10.5) + 28.0) * 1e-3  # s

    # ------------------------------ States derivatives ------------------------------

    def derStates(self, Vm, states):
        dstates = super().derStates(Vm, states)
        dstates.update({
            's': (self.sinf(Vm) - states['s']) / self.taus(Vm),
            'u': (self.uinf(Vm) - states['u']) / self.tauu(Vm)
        })
        return dstates

    def derEffStates(self, Vm, states, rates):
        dstates = super().derEffStates(Vm, states, rates)
        dstates.update({
            's': rates['alphas'] * (1 - states['s']) - rates['betas'] * states['s'],
            'u': rates['alphau'] * (1 - states['u']) - rates['betau'] * states['u']
        })
        return dstates

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
        sstates = super().steadyStates(Vm)
        sstates['s'] = self.sinf(Vm)
        sstates['u'] = self.uinf(Vm)
        return sstates

    # ------------------------------ Membrane currents ------------------------------

    def iCaT(self, s, u, Vm):
        ''' low-threshold (T-type) Calcium current '''
        return self.gCaTbar * s**2 * u * (Vm - self.ECa)  # mA/m2

    def currents(self):
        currents = super().currents()
        currents['iCaT'] = lambda Vm, states: self.iCaT(states['s'], states['u'], Vm)
        return currents

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        effrates = super().computeEffRates(Vm)
        effrates.update({
            'alphas': np.mean(self.sinf(Vm) / self.taus(Vm)),
            'betas': np.mean((1 - self.sinf(Vm)) / self.taus(Vm)),
            'alphau': np.mean(self.uinf(Vm) / self.tauu(Vm)),
            'betau': np.mean((1 - self.uinf(Vm)) / self.tauu(Vm))
        })
        return effrates


class CorticalIB(Cortical):
    ''' Cortical intrinsically bursting neuron

        References:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

        *Reuveni, I., Friedman, A., Amitai, Y., and Gutnick, M.J. (1993). Stepwise
        repolarization from Ca2+ plateaus in neocortical pyramidal cells: evidence
        for nonhomogeneous distribution of HVA Ca2+ channels in dendrites.
        J. Neurosci. 13, 4609–4621.*
    '''

    # Neuron name
    name = 'IB'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Vm0 = -71.4  # Membrane potential (mV)

    # Reversal potentials (mV)
    ELeak = -70  # Non-specific leakage

    # Maximal channel conductances (S/m2)
    gNabar = 500   # Sodium
    gKdbar = 50    # Delayed-rectifier Potassium
    gMbar = 0.3    # Slow non-inactivating Potassium
    gCaLbar = 1.0  # High-threshold Calcium
    gLeak = 0.1    # Non-specific leakage

    # Additional parameters
    VT = -56.2      # Spike threshold adjustment parameter (mV)
    TauMax = 0.608  # Max. adaptation decay of slow non-inactivating Potassium current (s)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd gate',
        'p': 'iM gate',
        'q': 'iCaL activation gate',
        'r': 'iCaL inactivation gate'
    }

    # ------------------------------ Gating states kinetics ------------------------------

    def alphaq(self, Vm):
        return 0.055 * self.vtrap(-(Vm + 27), 3.8) * 1e3  # s-1

    def betaq(self, Vm):
        return 0.94 * np.exp(-(Vm + 75) / 17) * 1e3  # s-1

    def alphar(self, Vm):
        return 0.000457 * np.exp(-(Vm + 13) / 50) * 1e3  # s-1

    def betar(self, Vm):
        return 0.0065 / (np.exp(-(Vm + 15) / 28) + 1) * 1e3  # s-1

    # ------------------------------ States derivatives ------------------------------

    def derStates(self, Vm, states):
        dstates = super().derStates(Vm, states)
        dstates.update({
            'q': self.alphaq(Vm) * (1 - states['q']) - self.betaq(Vm) * states['q'],
            'r': self.alphar(Vm) * (1 - states['r']) - self.betar(Vm) * states['r']
        })
        return dstates

    def derEffStates(self, Vm, states, rates):
        dstates = super().derEffStates(Vm, states, rates)
        dstates.update({
            'q': rates['alphaq'] * (1 - states['q']) - rates['betaq'] * states['q'],
            'r': rates['alphar'] * (1 - states['r']) - rates['betar'] * states['r']
        })
        return dstates

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
        sstates = super().steadyStates(Vm)
        sstates['q'] = self.alphaq(Vm) / (self.alphaq(Vm) + self.betaq(Vm))
        sstates['r'] = self.alphar(Vm) / (self.alphar(Vm) + self.betar(Vm))
        return sstates

    # ------------------------------ Membrane currents ------------------------------

    def iCaL(self, q, r, Vm):
        ''' high-threshold (L-type) Calcium current '''
        return self.gCaLbar * q**2 * r * (Vm - self.ECa)  # mA/m2

    def currents(self, Vm, states):
        currents = super().currents(Vm, states)
        currents['iCaL'] = lambda Vm, states: self.iCaL(states['q'], states['r'], Vm)
        return currents

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        effrates = super().computeEffRates(Vm)
        effrates.update({
            'alphaq': np.mean(self.alphaq(Vm)),
            'betaq': np.mean(self.betaq(Vm)),
            'alphar': np.mean(self.alphar(Vm)),
            'betar': np.mean(self.betar(Vm))
        })
        return effrates
