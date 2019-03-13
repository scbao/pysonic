#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-31 15:19:51
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-13 18:15:56

import numpy as np
from ..core import PointNeuron
from ..utils import vtrap


class Cortical(PointNeuron):
    ''' Class defining the generic membrane channel dynamics of a cortical neuron
        with 4 different current types:
            - Inward Sodium current
            - Outward, delayed-rectifier Potassium current
            - Outward, slow non.inactivating Potassium current
            - Non-specific leakage current
        This generic class cannot be used directly as it does not contain any specific parameters.

        Reference:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*
    '''

    # Generic biophysical parameters of cortical cells
    Cm0 = 1e-2  # Cell membrane resting capacitance (F/m2)
    Vm0 = 0.0  # Dummy value for membrane potential (mV)
    ENa = 50.0  # Sodium Nernst potential (mV)
    EK = -90.0  # Potassium Nernst potential (mV)
    ECa = 120.0  # # Calcium Nernst potential (mV)


    def __init__(self):
        self.states_names = ['m', 'h', 'n', 'p']
        self.coeff_names = ['alpham', 'betam', 'alphah', 'betah', 'alphan', 'betan',
                            'alphap', 'betap']
        self.states0 = np.array([])


    def alpham(self, Vm):
        ''' Voltage-dependent activation rate of m-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        alpha = 0.32 * vtrap(13 - Vdiff, 4)  # ms-1
        return alpha * 1e3  # s-1


    def betam(self, Vm):
        ''' Voltage-dependent inactivation rate of m-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        beta = 0.28 * vtrap(Vdiff - 40, 5)  # ms-1
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
        alpha = 0.032 * vtrap(15 - Vdiff, 5)  # ms-1
        return alpha * 1e3  # s-1


    def betan(self, Vm):
        ''' Voltage-dependent inactivation rate of n-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        Vdiff = Vm - self.VT
        beta = (0.5 * np.exp(-(Vdiff - 10) / 40))  # ms-1
        return beta * 1e3  # s-1


    def pinf(self, Vm):
        ''' Voltage-dependent steady-state opening of p-gate

            :param Vm: membrane potential (mV)
            :return: steady-state opening (-)
        '''
        return 1.0 / (1 + np.exp(-(Vm + 35) / 10))


    def taup(self, Vm):
        ''' Voltage-dependent adaptation time for adaptation of p-gate

            :param Vm: membrane potential (mV)
            :return: adaptation time (s)
        '''
        return self.TauMax / (3.3 * np.exp((Vm + 35) / 20) + np.exp(-(Vm + 35) / 20))  # s


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


    def derP(self, Vm, p):
        ''' Evolution of p-gate open-probability

            :param Vm: membrane potential (mV)
            :param p: open-probability of p-gate (-)
            :return: time derivative of p-gate open-probability (s-1)
        '''
        return (self.pinf(Vm) - p) / self.taup(Vm)


    def iNa(self, m, h, Vm):
        ''' Sodium current

            :param m: open-probability of m-gate (-)
            :param h: open-probability of h-gate (-)
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
        return self.gKbar * n**4 * (Vm - self.EK)


    def iM(self, p, Vm):
        ''' Slow non-inactivating Potassium current

            :param p: open-probability of p-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gMbar * p * (Vm - self.EK)


    def iLeak(self, Vm):
        ''' Non-specific leakage current

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gLeak * (Vm - self.ELeak)


    def currents(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, p = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iKd': self.iKd(n, Vm),
            'iM': self.iM(p, Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2


    def steadyStates(self, Vm):
        ''' Overriding of abstract parent method. '''
        meq = self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm))
        heq = self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm))
        neq = self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm))
        peq = self.pinf(Vm)

        return np.array([meq, heq, neq, peq])


    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, p = states
        dmdt = self.derM(Vm, m)
        dhdt = self.derH(Vm, h)
        dndt = self.derN(Vm, n)
        dpdt = self.derP(Vm, p)

        return [dmdt, dhdt, dndt, dpdt]


    def getEffRates(self, Vm):
        ''' Overriding of abstract parent method. '''
        am_avg = np.mean(self.alpham(Vm))
        bm_avg = np.mean(self.betam(Vm))
        ah_avg = np.mean(self.alphah(Vm))
        bh_avg = np.mean(self.betah(Vm))
        an_avg = np.mean(self.alphan(Vm))
        bn_avg = np.mean(self.betan(Vm))
        Tp = self.taup(Vm)
        pinf = self.pinf(Vm)
        ap_avg = np.mean(pinf / Tp)
        bp_avg = np.mean(1 / Tp) - ap_avg

        return np.array([am_avg, bm_avg, ah_avg, bh_avg, an_avg, bn_avg, ap_avg, bp_avg])


    def derStatesEff(self, Qm, states, interp_data):
        ''' Overriding of abstract parent method. '''

        rates = np.array([np.interp(Qm, interp_data['Q'], interp_data[rn])
                          for rn in self.coeff_names])
        m, h, n, p = states
        dmdt = rates[0] * (1 - m) - rates[1] * m
        dhdt = rates[2] * (1 - h) - rates[3] * h
        dndt = rates[4] * (1 - n) - rates[5] * n
        dpdt = rates[6] * (1 - p) - rates[7] * p

        return [dmdt, dhdt, dndt, dpdt]


class CorticalRS(Cortical):
    ''' Specific membrane channel dynamics of a cortical regular spiking, excitatory
        pyramidal neuron.

        Reference:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*
    '''

    # Name of channel mechanism
    name = 'RS'

    # Cell-specific biophysical parameters
    Vm0 = -71.9  # Cell membrane resting potential (mV)
    gNabar = 560.0  # Max. conductance of Sodium current (S/m^2)
    gKbar = 60.0  # Max. conductance of delayed Potassium current (S/m^2)
    gMbar = 0.75  # Max. conductance of slow non-inactivating Potassium current (S/m^2)
    gLeak = 0.205  # Conductance of non-specific leakage current (S/m^2)
    ELeak = -70.3  # Non-specific leakage Nernst potential (mV)
    VT = -56.2  # Spike threshold adjustment parameter (mV)
    TauMax = 0.608  # Max. adaptation decay of slow non-inactivating Potassium current (s)

    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h'],
        'i_{Kd}\ kin.': ['n'],
        'i_M\ kin.': ['p']
    }


    def __init__(self):
        super().__init__()
        self.states0 = self.steadyStates(self.Vm0)



class CorticalFS(Cortical):
    ''' Specific membrane channel dynamics of a cortical fast-spiking, inhibitory neuron.

        Reference:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*
    '''

    # Name of channel mechanism
    name = 'FS'

    # Cell-specific biophysical parameters
    Vm0 = -71.4  # Cell membrane resting potential (mV)
    gNabar = 580.0  # Max. conductance of Sodium current (S/m^2)
    gKbar = 39.0  # Max. conductance of delayed Potassium current (S/m^2)
    gMbar = 0.787  # Max. conductance of slow non-inactivating Potassium current (S/m^2)
    gLeak = 0.38  # Conductance of non-specific leakage current (S/m^2)
    ELeak = -70.4  # Non-specific leakage Nernst potential (mV)
    VT = -57.9  # Spike threshold adjustment parameter (mV)
    TauMax = 0.502  # Max. adaptation decay of slow non-inactivating Potassium current (s)

    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h'],
        'i_{Kd}\ kin.': ['n'],
        'i_M\ kin.': ['p']
    }


    def __init__(self):
        super().__init__()
        self.states0 = self.steadyStates(self.Vm0)



class CorticalLTS(Cortical):
    ''' Specific membrane channel dynamics of a cortical low-threshold spiking, inhibitory
        neuron with an additional inward Calcium current due to the presence of a T-type channel.

        References:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

        *Huguenard, J.R., and McCormick, D.A. (1992). Simulation of the currents involved in
        rhythmic oscillations in thalamic relay neurons. J. Neurophysiol. 68, 1373–1383.*

    '''

    # Name of channel mechanism
    name = 'LTS'

    # Cell-specific biophysical parameters
    Vm0 = -54.0  # Cell membrane resting potential (mV)
    gNabar = 500.0  # Max. conductance of Sodium current (S/m^2)
    gKbar = 40.0  # Max. conductance of delayed Potassium current (S/m^2)
    gMbar = 0.28  # Max. conductance of slow non-inactivating Potassium current (S/m^2)
    gCaTbar = 4.0  # Max. conductance of low-threshold Calcium current (S/m^2)
    gLeak = 0.19  # Conductance of non-specific leakage current (S/m^2)
    ELeak = -50.0  # Non-specific leakage Nernst potential (mV)
    VT = -50.0  # Spike threshold adjustment parameter (mV)
    TauMax = 4.0  # Max. adaptation decay of slow non-inactivating Potassium current (s)
    Vx = -7.0  # Voltage-dependence uniform shift factor at 36°C (mV)

    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h'],
        'i_{Kd}\ kin.': ['n'],
        'i_M\ kin.': ['p'],
        'i_{CaT}\ kin.': ['s', 'u']
    }

    def __init__(self):
        super().__init__()
        self.states_names += ['s', 'u']
        self.coeff_names += ['alphas', 'betas', 'alphau', 'betau']
        self.states0 = self.steadyStates(self.Vm0)


    def sinf(self, Vm):
        ''' Voltage-dependent steady-state opening of s-gate

            :param Vm: membrane potential (mV)
            :return: steady-state opening (-)
        '''
        return 1.0 / (1.0 + np.exp(-(Vm + self.Vx + 57.0) / 6.2))


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
            return 1.0 / 3.7 * (np.exp(-(Vm + self.Vx + 22) / 10.5) + 28.0) * 1e-3  # s


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


    def iCaT(self, s, u, Vm):
        ''' Low-threshold (T-type) Calcium current

            :param s: open-probability of s-gate (-)
            :param u: open-probability of u-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gCaTbar * s**2 * u * (Vm - self.ECa)


    def currents(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, p, s, u = states
        currents = super().currents(Vm, [m, h, n, p])
        currents['iCaT'] = self.iCaT(s, u, Vm)  # mA/m2
        return currents


    def steadyStates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Call parent method to compute Sodium and Potassium channels gates steady-states
        NaK_eqstates = super().steadyStates(Vm)

        # Compute Calcium channel gates steady-states
        Ca_eqstates = np.array([self.sinf(Vm), self.uinf(Vm)])

        # Merge all steady-states and return
        return np.concatenate((NaK_eqstates, Ca_eqstates))


    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''

        # Unpack input states
        *NaK_states, s, u = states

        # Call parent method to compute Sodium and Potassium channels states derivatives
        NaK_derstates = super().derStates(Vm, NaK_states)

        # Compute Calcium channels states derivatives
        dsdt = self.derS(Vm, s)
        dudt = self.derU(Vm, u)

        # Merge all states derivatives and return
        return NaK_derstates + [dsdt, dudt]


    def getEffRates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Call parent method to compute Sodium and Potassium effective rate constants
        NaK_rates = super().getEffRates(Vm)

        # Compute Calcium effective rate constants
        Ts = self.taus(Vm)
        as_avg = np.mean(self.sinf(Vm) / Ts)
        bs_avg = np.mean(1 / Ts) - as_avg
        Tu = np.array([self.tauu(v) for v in Vm])
        au_avg = np.mean(self.uinf(Vm) / Tu)
        bu_avg = np.mean(1 / Tu) - au_avg
        Ca_rates = np.array([as_avg, bs_avg, au_avg, bu_avg])

        # Merge all rates and return
        return np.concatenate((NaK_rates, Ca_rates))


    def derStatesEff(self, Qm, states, interp_data):
        ''' Overriding of abstract parent method. '''

        # Unpack input states
        *NaK_states, s, u = states

        # Call parent method to compute Sodium and Potassium channels states derivatives
        NaK_dstates = super().derStatesEff(Qm, NaK_states, interp_data)

        # Compute Calcium channels states derivatives
        Ca_rates = np.array([np.interp(Qm, interp_data['Q'], interp_data[rn])
                             for rn in self.coeff_names[8:]])
        dsdt = Ca_rates[0] * (1 - s) - Ca_rates[1] * s
        dudt = Ca_rates[2] * (1 - u) - Ca_rates[3] * u

        # Merge all states derivatives and return
        return NaK_dstates + [dsdt, dudt]


class CorticalIB(Cortical):
    ''' Specific membrane channel dynamics of a cortical intrinsically bursting neuron with
        an additional inward Calcium current due to the presence of a L-type channel.

        References:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

        *Reuveni, I., Friedman, A., Amitai, Y., and Gutnick, M.J. (1993). Stepwise repolarization
        from Ca2+ plateaus in neocortical pyramidal cells: evidence for nonhomogeneous distribution
        of HVA Ca2+ channels in dendrites. J. Neurosci. 13, 4609–4621.*
    '''

    # Name of channel mechanism
    name = 'IB'

    # Cell-specific biophysical parameters
    Vm0 = -71.4  # Cell membrane resting potential (mV)
    gNabar = 500  # Max. conductance of Sodium current (S/m^2)
    gKbar = 50  # Max. conductance of delayed Potassium current (S/m^2)
    gMbar = 0.3  # Max. conductance of slow non-inactivating Potassium current (S/m^2)
    gCaLbar = 1.0  # Max. conductance of L-type Calcium current (S/m^2)
    gLeak = 0.1  # Conductance of non-specific leakage current (S/m^2)
    ELeak = -70  # Non-specific leakage Nernst potential (mV)
    VT = -56.2  # Spike threshold adjustment parameter (mV)
    TauMax = 0.608  # Max. adaptation decay of slow non-inactivating Potassium current (s)

    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h'],
        'i_{Kd}\ kin.': ['n'],
        'i_M\ kin.': ['p'],
        'i_{CaL}\ kin.': ['q', 'r']
    }

    def __init__(self):
        super().__init__()
        self.states_names += ['q', 'r']
        self.coeff_names += ['alphaq', 'betaq', 'alphar', 'betar']
        self.states0 = self.steadyStates(self.Vm0)


    def alphaq(self, Vm):
        ''' Voltage-dependent activation rate of q-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        alpha = 0.055 * vtrap(-(Vm + 27), 3.8)  # ms-1
        return alpha * 1e3  # s-1


    def betaq(self, Vm):
        ''' Voltage-dependent inactivation rate of q-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        beta = 0.94 * np.exp(-(Vm + 75) / 17)  # ms-1
        return beta * 1e3  # s-1


    def alphar(self, Vm):
        ''' Voltage-dependent activation rate of r-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        alpha = 0.000457 * np.exp(-(Vm + 13) / 50)  # ms-1
        return alpha * 1e3  # s-1


    def betar(self, Vm):
        ''' Voltage-dependent inactivation rate of r-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        beta = 0.0065 / (np.exp(-(Vm + 15) / 28) + 1)  # ms-1
        return beta * 1e3  # s-1


    def derQ(self, Vm, q):
        ''' Evolution of q-gate open-probability

            :param Vm: membrane potential (mV)
            :param q: open-probability of q-gate (-)
            :return: time derivative of q-gate open-probability (s-1)
        '''
        return self.alphaq(Vm) * (1 - q) - self.betaq(Vm) * q


    def derR(self, Vm, r):
        ''' Evolution of r-gate open-probability

            :param Vm: membrane potential (mV)
            :param r: open-probability of r-gate (-)
            :return: time derivative of r-gate open-probability (s-1)
        '''
        return self.alphar(Vm) * (1 - r) - self.betar(Vm) * r


    def iCaL(self, q, r, Vm):
        ''' High-threshold (L-type) Calcium current

            :param q: open-probability of q-gate (-)
            :param r: open-probability of r-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gCaLbar * q**2 * r * (Vm - self.ECa)


    def currents(self, Vm, states):
        ''' Overriding of abstract parent method. '''
        m, h, n, p, q, r = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iKd': self.iKd(n, Vm),
            'iM': self.iM(p, Vm),
            'iCaL': self.iCaL(q, r, Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2


    def steadyStates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Call parent method to compute Sodium and Potassium channels gates steady-states
        NaK_eqstates = super().steadyStates(Vm)

        # Compute L-type Calcium channel gates steady-states
        qeq = self.alphaq(Vm) / (self.alphaq(Vm) + self.betaq(Vm))
        req = self.alphar(Vm) / (self.alphar(Vm) + self.betar(Vm))
        CaL_eqstates = np.array([qeq, req])

        # Merge all steady-states and return
        return np.concatenate((NaK_eqstates, CaL_eqstates))


    def derStates(self, Vm, states):
        ''' Overriding of abstract parent method. '''

        # Unpack input states
        *NaK_states, q, r = states

        # Call parent method to compute Sodium and Potassium channels states derivatives
        NaK_derstates = super().derStates(Vm, NaK_states)

        # Compute L-type Calcium channels states derivatives
        dqdt = self.derQ(Vm, q)
        drdt = self.derR(Vm, r)

        # Merge all states derivatives and return
        return NaK_derstates + [dqdt, drdt]


    def getEffRates(self, Vm):
        ''' Overriding of abstract parent method. '''

        # Call parent method to compute Sodium and Potassium effective rate constants
        NaK_rates = super().getEffRates(Vm)

        # Compute Calcium effective rate constants
        aq_avg = np.mean(self.alphaq(Vm))
        bq_avg = np.mean(self.betaq(Vm))
        ar_avg = np.mean(self.alphar(Vm))
        br_avg = np.mean(self.betar(Vm))
        CaL_rates = np.array([aq_avg, bq_avg, ar_avg, br_avg])

        # Merge all rates and return
        return np.concatenate((NaK_rates, CaL_rates))


    def derStatesEff(self, Qm, states, interp_data):
        ''' Overriding of abstract parent method. '''

        # Unpack input states
        *NaK_states, q, r = states

        # Call parent method to compute Sodium and Potassium channels states derivatives
        NaK_dstates = super().derStatesEff(Qm, NaK_states, interp_data)

        # Compute Calcium channels states derivatives
        CaL_rates = np.array([np.interp(Qm, interp_data['Q'], interp_data[rn])
                              for rn in self.coeff_names[8:]])
        dqdt = CaL_rates[0] * (1 - q) - CaL_rates[1] * q
        drdt = CaL_rates[2] * (1 - r) - CaL_rates[3] * r

        # Merge all states derivatives and return
        return NaK_dstates + [dqdt, drdt]
