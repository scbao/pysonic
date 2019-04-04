#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-31 15:20:54
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-04-03 16:33:28

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

    # Generic biophysical parameters of thalamic cells
    Cm0 = 1e-2  # Cell membrane resting capacitance (F/m2)
    Vm0 = 0.0  # Dummy value for membrane potential (mV)
    ENa = 50.0  # Sodium Nernst potential (mV)
    EK = -90.0  # Potassium Nernst potential (mV)
    ECa = 120.0  # Calcium Nernst potential (mV)

    def __init__(self):
        self.states = ['m', 'h', 'n', 's', 'u']
        self.rates = self.getRatesNames(self.states)

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

    def derM(self, Vm, m):
        ''' Evolution of m-gate open-probability

            :param Vm: membrane potential (mV)
            :param m: open-probability of m-gate (-)
            :return: time derivative of m-gate open-probability (s-1)
        '''
        return self.alpham(Vm) * (1 - m) - self.betam(Vm) * m

    def derH(self, Vm, h):
        ''' Evolution of h-gate open-probability

            :param Vm: membrane potential (mV)
            :param h: open-probability of h-gate (-)
            :return: time derivative of h-gate open-probability (s-1)
        '''
        return self.alphah(Vm) * (1 - h) - self.betah(Vm) * h

    def derN(self, Vm, n):
        ''' Evolution of n-gate open-probability

            :param Vm: membrane potential (mV)
            :param n: open-probability of n-gate (-)
            :return: time derivative of n-gate open-probability (s-1)
        '''
        return self.alphan(Vm) * (1 - n) - self.betan(Vm) * n

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
        ''' Overriding of abstract parent method. '''
        m, h, n, s, u = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iKd': self.iKd(n, Vm),
            'iCaT': self.iCaT(s, u, Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2

    def steadyStates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Voltage-gated steady-states
        m_eq = self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm))
        h_eq = self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm))
        n_eq = self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm))
        s_eq = self.sinf(Vm)
        u_eq = self.uinf(Vm)

        return np.array([m_eq, h_eq, n_eq, s_eq, u_eq])

    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''

        m, h, n, s, u = states
        dmdt = self.derM(Vm, m)
        dhdt = self.derH(Vm, h)
        dndt = self.derN(Vm, n)
        dsdt = self.derS(Vm, s)
        dudt = self.derU(Vm, u)

        return [dmdt, dhdt, dndt, dsdt, dudt]

    def getEffRates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Compute average cycle value for rate constants
        am_avg = np.mean(self.alpham(Vm))
        bm_avg = np.mean(self.betam(Vm))
        ah_avg = np.mean(self.alphah(Vm))
        bh_avg = np.mean(self.betah(Vm))
        an_avg = np.mean(self.alphan(Vm))
        bn_avg = np.mean(self.betan(Vm))
        Ts = self.taus(Vm)
        as_avg = np.mean(self.sinf(Vm) / Ts)
        bs_avg = np.mean(1 / Ts) - as_avg
        Tu = np.array([self.tauu(v) for v in Vm])
        au_avg = np.mean(self.uinf(Vm) / Tu)
        bu_avg = np.mean(1 / Tu) - au_avg

        # Return array of coefficients
        return np.array([am_avg, bm_avg, ah_avg, bh_avg, an_avg, bn_avg,
                         as_avg, bs_avg, au_avg, bu_avg])

    def derStatesEff(self, Qm, states, interp_data):
        ''' Overriding of abstract parent method. '''

        rates = {rn: np.interp(Qm, interp_data['Q'], interp_data[rn]) for rn in self.rates}

        m, h, n, s, u = states
        dmdt = rates['alpham'] * (1 - m) - rates['betam'] * m
        dhdt = rates['alphah'] * (1 - h) - rates['betah'] * h
        dndt = rates['alphan'] * (1 - n) - rates['betan'] * n
        dsdt = rates['alphas'] * (1 - s) - rates['betas'] * s
        dudt = rates['alphau'] * (1 - u) - rates['betau'] * u

        return [dmdt, dhdt, dndt, dsdt, dudt]


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

    # Name of channel mechanism
    name = 'RE'

    # Cell-specific biophysical parameters
    Vm0 = -89.5  # Cell membrane resting potential (mV)
    gNabar = 2000.0  # Max. conductance of Sodium current (S/m^2)
    gKdbar = 200.0  # Max. conductance of Potassium current (S/m^2)
    gCaTbar = 30.0  # Max. conductance of low-threshold Calcium current (S/m^2)
    gLeak = 0.5  # Conductance of non-specific leakage current (S/m^2)
    ELeak = -90.0  # Non-specific leakage Nernst potential (mV)
    VT = -67.0  # Spike threshold adjustment parameter (mV)

    def __init__(self):
        super().__init__()

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
    ''' Thalamo-cortical neuron, with a specific

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

    # Name of channel mechanism
    name = 'TC'

    # Cell-specific biophysical parameters
    # Vm0 = -63.4  # Cell membrane resting potential (mV)
    Vm0 = -61.93  # Cell membrane resting potential (mV)
    gNabar = 900.0  # bar. conductance of Sodium current (S/m^2)
    gKdbar = 100.0  # bar. conductance of Potassium current (S/m^2)
    gCaTbar = 20.0  # Max. conductance of low-threshold Calcium current (S/m^2)
    gKLeak = 0.138  # Conductance of leakage Potassium current (S/m^2)
    gHbar = 0.175  # Max. conductance of mixed cationic current (S/m^2)
    gLeak = 0.1  # Conductance of non-specific leakage current (S/m^2)
    EH = -40.0  # Mixed cationic current reversal potential (mV)
    ELeak = -70.0  # Non-specific leakage Nernst potential (mV)
    VT = -52.0  # Spike threshold adjustment parameter (mV)
    Vx = 0.0  # Voltage-dependence uniform shift factor at 36°C (mV)

    taur_Cai = 5e-3  # decay time constant for intracellular Ca2+ dissolution (s)
    Cai_min = 50e-9  # minimal intracellular Calcium concentration (M)
    deff = 100e-9  # effective depth beneath membrane for intracellular [Ca2+] calculation
    nCa = 4  # number of Calcium binding sites on regulating factor
    k1 = 2.5e22  # intracellular Ca2+ regulation factor (M-4 s-1)
    k2 = 0.4  # intracellular Ca2+ regulation factor (s-1)
    k3 = 100.0  # intracellular Ca2+ regulation factor (s-1)
    k4 = 1.0  # intracellular Ca2+ regulation factor (s-1)

    def __init__(self):
        super().__init__()
        self.iCa_to_Cai_rate = self.currentToConcentrationRate(Z_Ca, self.deff)
        self.states += ['O', 'C', 'P0', 'Cai']
        self.rates += self.getRatesNames(['O'])

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
        ''' Overriding of abstract parent method. '''
        m, h, n, s, u, O, C, _, _ = states
        currents = super().currents(Vm, [m, h, n, s, u])
        currents['iKLeak'] = self.iKLeak(Vm)  # mA/m2
        currents['iH'] = self.iH(O, C, Vm)  # mA/m2
        return currents

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
        ''' Overriding of abstract parent method. '''

        # Voltage-gated steady-states
        NaKCa_eq = super().steadyStates(Vm)
        seq = NaKCa_eq[3]
        ueq = NaKCa_eq[4]

        # Other steady-states
        Cai_eq = self.Caiinf(Vm, seq, ueq)
        P0_eq = self.P0inf(Cai_eq)
        O_eq = self.Oinf(Cai_eq, Vm)
        C_eq = self.Cinf(Cai_eq, Vm)

        kin_eq = np.array([O_eq, C_eq, P0_eq, Cai_eq])

        return np.concatenate((NaKCa_eq, kin_eq))

    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''

        m, h, n, s, u, O, C, P0, Cai = states

        NaKCa_states = [m, h, n, s, u]
        NaKCa_derstates = super().derStates(Vm, NaKCa_states)

        dOdt = self.derO(C, O, P0, Vm)
        dCdt = self.derC(C, O, Vm)
        dP0dt = self.derP0(P0, Cai)
        dCaidt = self.derCai(Cai, s, u, Vm)

        return NaKCa_derstates + [dOdt, dCdt, dP0dt, dCaidt]

    def getEffRates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Compute effective coefficients for Sodium, Potassium and Calcium conductances
        NaKCa_effrates = super().getEffRates(Vm)

        # Compute effective coefficients for Ih conductance
        ao_avg = np.mean(self.alphao(Vm))
        bo_avg = np.mean(self.betao(Vm))
        iH_effrates = np.array([ao_avg, bo_avg])

        # Return array of coefficients
        return np.concatenate((NaKCa_effrates, iH_effrates))

    def derStatesEff(self, Qm, states, interp_data):
        ''' Overriding of abstract parent method. '''

        rates = {rn: np.interp(Qm, interp_data['Q'], interp_data[rn]) for rn in self.rates}
        Vmeff = np.interp(Qm, interp_data['Q'], interp_data['V'])

        # Unpack states
        m, h, n, s, u, O, C, P0, Cai = states

        # INa, IK, iCaT effective states derivatives
        dmdt = rates['alpham'] * (1 - m) - rates['betam'] * m
        dhdt = rates['alphah'] * (1 - h) - rates['betah'] * h
        dndt = rates['alphan'] * (1 - n) - rates['betan'] * n
        dsdt = rates['alphas'] * (1 - s) - rates['betas'] * s
        dudt = rates['alphau'] * (1 - u) - rates['betau'] * u

        # Ih effective states derivatives
        dCdt = rates['betao'] * O - rates['alphao'] * C
        dOdt = - dCdt - self.k3 * O * (1 - P0) + self.k4 * (1 - O - C)
        dP0dt = self.derP0(P0, Cai)
        dCaidt = self.derCai(Cai, s, u, Vmeff)

        # Merge derivatives and return
        return [dmdt, dhdt, dndt, dsdt, dudt, dOdt, dCdt, dP0dt, dCaidt]
