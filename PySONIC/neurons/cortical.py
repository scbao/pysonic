# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-31 15:19:51
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-20 09:54:27

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

    def derStates(self):
        return {
            'm': lambda Vm, x: self.alpham(Vm) * (1 - x['m']) - self.betam(Vm) * x['m'],
            'h': lambda Vm, x: self.alphah(Vm) * (1 - x['h']) - self.betah(Vm) * x['h'],
            'n': lambda Vm, x: self.alphan(Vm) * (1 - x['n']) - self.betan(Vm) * x['n'],
            'p': lambda Vm, x: (self.pinf(Vm) - x['p']) / self.taup(Vm)
        }

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self):
        return {
            'm': lambda Vm: self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm)),
            'h': lambda Vm: self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm)),
            'n': lambda Vm: self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm)),
            'p': lambda Vm: self.pinf(Vm)
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
            'iNa': lambda Vm, x: self.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: self.iKd(x['n'], Vm),
            'iM': lambda Vm, x: self.iM(x['p'], Vm),
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

    def derStates(self):
        return {**super().derStates(), **{
            's': lambda Vm, x: (self.sinf(Vm) - x['s']) / self.taus(Vm),
            'u': lambda Vm, x: (self.uinf(Vm) - x['u']) / self.tauu(Vm)
        }}

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self):
        return {**super().steadyStates(), **{
            's': lambda Vm: self.sinf(Vm),
            'u': lambda Vm: self.uinf(Vm)
        }}

    # ------------------------------ Membrane currents ------------------------------

    def iCaT(self, s, u, Vm):
        ''' low-threshold (T-type) Calcium current '''
        return self.gCaTbar * s**2 * u * (Vm - self.ECa)  # mA/m2

    def currents(self):
        return {**super().currents(), **{
            'iCaT': lambda Vm, x: self.iCaT(x['s'], x['u'], Vm)
        }}

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        return {**super().computeEffRates(Vm), **{
            'alphas': np.mean(self.sinf(Vm) / self.taus(Vm)),
            'betas': np.mean((1 - self.sinf(Vm)) / self.taus(Vm)),
            'alphau': np.mean(self.uinf(Vm) / self.tauu(Vm)),
            'betau': np.mean((1 - self.uinf(Vm)) / self.tauu(Vm))
        }}


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

    def derStates(self):
        return {**super().derStates(), **{
            'q': lambda Vm, x: self.alphaq(Vm) * (1 - x['q']) - self.betaq(Vm) * x['q'],
            'r': lambda Vm, x: self.alphar(Vm) * (1 - x['r']) - self.betar(Vm) * x['r']
        }}

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self):
        return {**super().steadyStates(), **{
            'q': lambda Vm: self.alphaq(Vm) / (self.alphaq(Vm) + self.betaq(Vm)),
            'r': lambda Vm: self.alphar(Vm) / (self.alphar(Vm) + self.betar(Vm))
        }}

    # ------------------------------ Membrane currents ------------------------------

    def iCaL(self, q, r, Vm):
        ''' high-threshold (L-type) Calcium current '''
        return self.gCaLbar * q**2 * r * (Vm - self.ECa)  # mA/m2

    def currents(self):
        return {**super().currents(), **{
            'iCaL': lambda Vm, x: self.iCaL(x['q'], x['r'], Vm)
        }}

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        return {**super().computeEffRates(Vm), **{
            'alphaq': np.mean(self.alphaq(Vm)),
            'betaq': np.mean(self.betaq(Vm)),
            'alphar': np.mean(self.alphar(Vm)),
            'betar': np.mean(self.betar(Vm))
        }}
