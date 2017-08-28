#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-31 15:20:54
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-28 17:55:32

''' Channels mechanisms for thalamic neurons. '''

import logging
import numpy as np
from .base import BaseMech

# Get package logger
logger = logging.getLogger('PointNICE')


class Thalamic(BaseMech):
    ''' Class defining the generic membrane channel dynamics of a thalamic neuron
        with 4 different current types:
            - Inward Sodium current
            - Outward Potassium current
            - Inward Calcium current
            - Non-specific leakage current
        This generic class cannot be used directly as it does not contain any specific parameters.

        Reference:
        *Plaksin, M., Kimmel, E., and Shoham, S. (2016). Cell-Type-Selective Effects of
        Intramembrane Cavitation as a Unifying Theoretical Framework for Ultrasonic
        Neuromodulation. eNeuro 3.*
    '''

    # Generic biophysical parameters of thalamic cells
    Cm0 = 1e-2  # Cell membrane resting capacitance (F/m2)
    Vm0 = 0.0  # Dummy value for membrane potential (mV)
    VNa = 50.0  # Sodium Nernst potential (mV)
    VK = -90.0  # Potassium Nernst potential (mV)
    VCa = 120.0  # Calcium Nernst potential (mV)

    def __init__(self):
        ''' Constructor of the class '''

        # Names and initial states of the channels state probabilities
        self.states_names = ['m', 'h', 'n', 's', 'u']
        self.states0 = np.array([])

        # Names of the different coefficients to be averaged in a lookup table.
        self.coeff_names = ['alpham', 'betam', 'alphah', 'betah', 'alphan', 'betan',
                            'alphas', 'betas', 'alphau', 'betau']


    def alpham(self, Vm):
        ''' Compute the alpha rate for the open-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        Vdiff = Vm - self.VT
        alpha = (-0.32 * (Vdiff - 13) / (np.exp(- (Vdiff - 13) / 4) - 1))  # ms-1
        return alpha * 1e3  # s-1


    def betam(self, Vm):
        ''' Compute the beta rate for the open-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        Vdiff = Vm - self.VT
        beta = (0.28 * (Vdiff - 40) / (np.exp((Vdiff - 40) / 5) - 1))  # ms-1
        return beta * 1e3  # s-1


    def alphah(self, Vm):
        ''' Compute the alpha rate for the inactivation-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        Vdiff = Vm - self.VT
        alpha = (0.128 * np.exp(-(Vdiff - 17) / 18))  # ms-1
        return alpha * 1e3  # s-1


    def betah(self, Vm):
        ''' Compute the beta rate for the inactivation-probability of Sodium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        Vdiff = Vm - self.VT
        beta = (4 / (1 + np.exp(-(Vdiff - 40) / 5)))  # ms-1
        return beta * 1e3  # s-1


    def alphan(self, Vm):
        ''' Compute the alpha rate for the open-probability of delayed-rectifier Potassium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        Vdiff = Vm - self.VT
        alpha = (-0.032 * (Vdiff - 15) / (np.exp(-(Vdiff - 15) / 5) - 1))  # ms-1
        return alpha * 1e3  # s-1


    def betan(self, Vm):
        ''' Compute the beta rate for the open-probability of delayed-rectifier Potassium channels.

            :param Vm: membrane potential (mV)
            :return: rate constant (s-1)
        '''

        Vdiff = Vm - self.VT
        beta = (0.5 * np.exp(-(Vdiff - 10) / 40))  # ms-1
        return beta * 1e3  # s-1


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
        ''' Compute the evolution of the open-probability of the S-type,
            activation gate of Calcium channels.

            :param Vm: membrane potential (mV)
            :param s: open-probability of S-type Calcium activation gates (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''

        return (self.sinf(Vm) - s) / self.taus(Vm)


    def derU(self, Vm, u):
        ''' Compute the evolution of the open-probability of the U-type,
            inactivation gate of Calcium channels.

            :param Vm: membrane potential (mV)
            :param u: open-probability of U-type Calcium inactivation gates (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''

        return (self.uinf(Vm) - u) / self.tauu(Vm)


    def currNa(self, m, h, Vm):
        ''' Compute the inward Sodium current per unit area.

            :param m: open-probability of Sodium channels
            :param h: inactivation-probability of Sodium channels
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        GNa = self.GNaMax * m**3 * h
        return GNa * (Vm - self.VNa)


    def currK(self, n, Vm):
        ''' Compute the outward delayed-rectifier Potassium current per unit area.

            :param n: open-probability of delayed-rectifier Potassium channels
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        GK = self.GKMax * n**4
        return GK * (Vm - self.VK)


    def currCa(self, s, u, Vm):
        ''' Compute the inward Calcium current per unit area.

            :param s: open-probability of the S-type activation gate of Calcium channels
            :param u: open-probability of the U-type inactivation gate of Calcium channels
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        GT = self.GTMax * s**2 * u
        return GT * (Vm - self.VCa)


    def currL(self, Vm):
        ''' Compute the non-specific leakage current per unit area.

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        return self.GL * (Vm - self.VL)


    def currNet(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        m, h, n, s, u = states
        return (self.currNa(m, h, Vm) + self.currK(n, Vm)
                + self.currCa(s, u, Vm) + self.currL(Vm))  # mA/m2


    def steadyStates(self, Vm):
        ''' Concrete implementation of the abstract API method. '''

        # Solve the equation dx/dt = 0 at Vm for each x-state
        meq = self.alpham(Vm) / (self.alpham(Vm) + self.betam(Vm))
        heq = self.alphah(Vm) / (self.alphah(Vm) + self.betah(Vm))
        neq = self.alphan(Vm) / (self.alphan(Vm) + self.betan(Vm))
        seq = self.sinf(Vm)
        ueq = self.uinf(Vm)
        return np.array([meq, heq, neq, seq, ueq])


    def derStates(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        m, h, n, s, u = states
        dmdt = self.derM(Vm, m)
        dhdt = self.derH(Vm, h)
        dndt = self.derN(Vm, n)
        dsdt = self.derS(Vm, s)
        dudt = self.derU(Vm, u)
        return [dmdt, dhdt, dndt, dsdt, dudt]


    def getEffRates(self, Vm):
        ''' Concrete implementation of the abstract API method. '''

        # Compute average cycle value for rate constants
        am_avg = np.mean(self.alpham(Vm))
        bm_avg = np.mean(self.betam(Vm))
        ah_avg = np.mean(self.alphah(Vm))
        bh_avg = np.mean(self.betah(Vm))
        an_avg = np.mean(self.alphan(Vm))
        bn_avg = np.mean(self.betan(Vm))
        Ts = self.taus(Vm)
        sinf = self.sinf(Vm)
        as_avg = np.mean(sinf / Ts)
        bs_avg = np.mean(1 / Ts) - as_avg
        Tu = np.array([self.tauu(v) for v in Vm])
        uinf = self.uinf(Vm)
        au_avg = np.mean(uinf / Tu)
        bu_avg = np.mean(1 / Tu) - au_avg

        # Return array of coefficients
        return np.array([am_avg, bm_avg, ah_avg, bh_avg, an_avg, bn_avg,
                         as_avg, bs_avg, au_avg, bu_avg])



    def derStatesEff(self, Qm, states, interp_data):
        ''' Concrete implementation of the abstract API method. '''

        rates = np.array([np.interp(Qm, interp_data['Q'], interp_data[rn])
                          for rn in self.coeff_names])

        m, h, n, s, u = states
        dmdt = rates[0] * (1 - m) - rates[1] * m
        dhdt = rates[2] * (1 - h) - rates[3] * h
        dndt = rates[4] * (1 - n) - rates[5] * n
        dsdt = rates[6] * (1 - s) - rates[7] * s
        dudt = rates[8] * (1 - u) - rates[9] * u

        return [dmdt, dhdt, dndt, dsdt, dudt]



class ThalamicRE(Thalamic):
    ''' Specific membrane channel dynamics of a thalamic reticular neuron.

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
    GNaMax = 2000.0  # Max. conductance of Sodium current (S/m^2)
    GKMax = 200.0  # Max. conductance of Potassium current (S/m^2)
    GTMax = 30.0  # Max. conductance of low-threshold Calcium current (S/m^2)
    GL = 0.5  # Conductance of non-specific leakage current (S/m^2)
    VL = -90.0  # Non-specific leakage Nernst potential (mV)
    VT = -67.0  # Spike threshold adjustment parameter (mV)

    # Default plotting scheme
    vars_RE = {
        'i_{Na}\ kin.': ['m', 'h'],
        'i_K\ kin.': ['n'],
        'i_{TS}\ kin.': ['s', 'u'],
        'I': ['iNa', 'iK', 'iTs', 'iL', 'iNet']
    }

    def __init__(self):
        ''' Constructor of the class. '''

        # Instantiate parent class
        super().__init__()

        # Define initial channel probabilities (solving dx/dt = 0 at resting potential)
        self.states0 = self.steadyStates(self.Vm0)


    def sinf(self, Vm):
        ''' Compute the asymptotic value of the open-probability of the S-type,
            activation gate of Calcium channels.

            :param Vm: membrane potential (mV)
            :return: asymptotic probability (-)
        '''

        return 1.0 / (1.0 + np.exp(-(Vm + 52.0) / 7.4))  # prob


    def taus(self, Vm):
        ''' Compute the decay time constant for adaptation of S-type,
            activation gate of Calcium channels.

            :param Vm: membrane potential (mV)
            :return: decayed time constant (s)
        '''
        return (1 + 0.33 / (np.exp((Vm + 27.0) / 10.0) + np.exp(-(Vm + 102.0) / 15.0))) * 1e-3  # s


    def uinf(self, Vm):
        ''' Compute the asymptotic value of the open-probability of the U-type,
            inactivation gate of Calcium channels.

            :param Vm: membrane potential (mV)
            :return: asymptotic probability (-)
        '''

        return 1.0 / (1.0 + np.exp((Vm + 80.0) / 5.0))  # prob


    def tauu(self, Vm):
        ''' Compute the decay time constant for adaptation of U-type,
            inactivation gate of Calcium channels.

            :param Vm: membrane potential (mV)
            :return: decayed time constant (s)
        '''
        return (28.3 + 0.33 / (np.exp((Vm + 48.0) / 4.0) + np.exp(-(Vm + 407.0) / 50.0))) * 1e-3  # s



class ThalamoCortical(Thalamic):
    ''' Specific membrane channel dynamics of a thalamo-cortical neuron, with a specific
        hyperpolarization-activated, mixed cationic current and a leakage Potassium current.

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
    GNaMax = 900.0  # Max. conductance of Sodium current (S/m^2)
    GKMax = 100.0  # Max. conductance of Potassium current (S/m^2)
    GTMax = 20.0  # Max. conductance of low-threshold Calcium current (S/m^2)
    GKL = 0.138  # Conductance of leakage Potassium current (S/m^2)
    GhMax = 0.175  # Max. conductance of mixed cationic current (S/m^2)
    GL = 0.1  # Conductance of non-specific leakage current (S/m^2)
    Vh = -40.0  # Mixed cationic current reversal potential (mV)
    VL = -70.0  # Non-specific leakage Nernst potential (mV)
    VT = -52.0  # Spike threshold adjustment parameter (mV)
    Vx = 0.0  # Voltage-dependence uniform shift factor at 36°C (mV)

    tau_Ca_removal = 5e-3  # decay time constant for intracellular Ca2+ dissolution (s)
    CCa_min = 50e-9  # minimal intracellular Calcium concentration (M)
    deff = 100e-9  # effective depth beneath membrane for intracellular [Ca2+] calculation
    F_Ca = 1.92988e5  # Faraday constant for bivalent ion (Coulomb / mole)
    nCa = 4  # number of Calcium binding sites on regulating factor
    k1 = 2.5e22  # intracellular Ca2+ regulation factor (M-4 s-1)
    k2 = 0.4  # intracellular Ca2+ regulation factor (s-1)
    k3 = 100.0  # intracellular Ca2+ regulation factor (s-1)
    k4 = 1.0  # intracellular Ca2+ regulation factor (s-1)

    # Default plotting scheme
    pltvars_scheme = {
        'i_{Na}\ kin.': ['m', 'h'],
        'i_K\ kin.': ['n'],
        'i_{T}\ kin.': ['s', 'u'],
        'i_{H}\ kin.': ['O', 'OL', 'O + 2OL'],
        'I': ['iNa', 'iK', 'iT', 'iH', 'iKL', 'iL', 'iNet']
    }


    def __init__(self):
        ''' Constructor of the class. '''

        # Instantiate parent class
        super().__init__()

        # Compute current to concentration conversion constant
        self.iT_2_CCa = 1e-6 / (self.deff * self.F_Ca)

        # Define names of the channels state probabilities
        self.states_names += ['O', 'C', 'P0', 'C_Ca']

        # Define the names of the different coefficients to be averaged in a lookup table.
        self.coeff_names += ['alphao', 'betao']

        # Define initial channel probabilities (solving dx/dt = 0 at resting potential)
        self.states0 = self.steadyStates(self.Vm0)


    def sinf(self, Vm):
        ''' Compute the asymptotic value of the open-probability of the S-type,
            activation gate of Calcium channels.

            Reference:
            *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac, Y.,
            Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for different
            classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

            :param Vm: membrane potential (mV)
            :return: asymptotic probability (-)
        '''

        return 1.0 / (1.0 + np.exp(-(Vm + self.Vx + 57.0) / 6.2))  # prob


    def taus(self, Vm):
        ''' Compute the decay time constant for adaptation of S-type,
            activation gate of Calcium channels.

            Reference:
            *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac, Y.,
            Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for different
            classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

            :param Vm: membrane potential (mV)
            :return: decayed time constant (s)
        '''
        tmp = np.exp(-(Vm + self.Vx + 132.0) / 16.7) + np.exp((Vm + self.Vx + 16.8) / 18.2)
        return 1.0 / 3.7 * (0.612 + 1.0 / tmp) * 1e-3  # s


    def uinf(self, Vm):
        ''' Compute the asymptotic value of the open-probability of the U-type,
            inactivation gate of Calcium channels.

            Reference:
            *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac, Y.,
            Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for different
            classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

            :param Vm: membrane potential (mV)
            :return: asymptotic probability (-)
        '''

        return 1.0 / (1.0 + np.exp((Vm + self.Vx + 81.0) / 4.0))  # prob


    def tauu(self, Vm):
        ''' Compute the decay time constant for adaptation of U-type,
            inactivation gate of Calcium channels.

            Reference:
            *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac, Y.,
            Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for different
            classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

            :param Vm: membrane potential (mV)
            :return: decayed time constant (s)
        '''

        if Vm + self.Vx < -80.0:
            return 1.0 / 3.7 * np.exp((Vm + self.Vx + 467.0) / 66.6) * 1e-3  # s
        else:
            return 1 / 3.7 * (np.exp(-(Vm + self.Vx + 22) / 10.5) + 28.0) * 1e-3  # s


    def derS(self, Vm, s):
        ''' Compute the evolution of the open-probability of the S-type,
            activation gate of Calcium channels.

            :param Vm: membrane potential (mV)
            :param s: open-probability of S-type Calcium activation gates (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''

        return (self.sinf(Vm) - s) / self.taus(Vm)


    def derU(self, Vm, u):
        ''' Compute the evolution of the open-probability of the U-type,
            inactivation gate of Calcium channels.

            :param Vm: membrane potential (mV)
            :param u: open-probability of U-type Calcium inactivation gates (prob)
            :return: derivative of open-probability w.r.t. time (prob/s)
        '''

        return (self.uinf(Vm) - u) / self.tauu(Vm)



    def oinf(self, Vm):
        ''' Voltage-dependent steady-state activation of hyperpolarization-activated
            cation current channels.

            Reference:
            *Huguenard, J.R., and McCormick, D.A. (1992). Simulation of the currents involved in
            rhythmic oscillations in thalamic relay neurons. J. Neurophysiol. 68, 1373–1383.*

            :param Vm: membrane potential (mV)
            :return: steady-state activation (-)
        '''

        return 1.0 / (1.0 + np.exp((Vm + 75.0) / 5.5))


    def tauo(self, Vm):
        ''' Time constant for activation of hyperpolarization-activated cation current channels.

            Reference:
            *Huguenard, J.R., and McCormick, D.A. (1992). Simulation of the currents involved in
            rhythmic oscillations in thalamic relay neurons. J. Neurophysiol. 68, 1373–1383.*

            :param Vm: membrane potential (mV)
            :return: time constant (s)
        '''

        return 1 / (np.exp(-14.59 - 0.086 * Vm) + np.exp(-1.87 + 0.0701 * Vm)) * 1e-3


    def alphao(self, Vm):
        ''' Transition rate between closed and open form of hyperpolarization-activated
            cation current channels.

            :param Vm: membrane potential (mV)
            :return: transition rate (s-1)
        '''

        return self.oinf(Vm) / self.tauo(Vm)


    def betao(self, Vm):
        ''' Transition rate between open and closed form of hyperpolarization-activated
            cation current channels.

            :param Vm: membrane potential (mV)
            :return: transition rate (s-1)
        '''

        return (1 - self.oinf(Vm)) / self.tauo(Vm)


    def derC(self, C, O, Vm):
        ''' Compute the evolution of the proportion of hyperpolarization-activated
            cation current channels in closed state.

            Kinetics scheme of Calcium dependent activation derived from:
            *Destexhe, A., Bal, T., McCormick, D.A., and Sejnowski, T.J. (1996). Ionic mechanisms
            underlying synchronized oscillations and propagating waves in a model of ferret
            thalamic slices. J. Neurophysiol. 76, 2049–2070.*

            :param Vm: membrane potential (mV)
            :param C: proportion of Ih channels in closed state (-)
            :param O: proportion of Ih channels in open state (-)
            :return: derivative of proportion w.r.t. time (s-1)
        '''

        return self.betao(Vm) * O - self.alphao(Vm) * C


    def derO(self, C, O, P0, Vm):
        ''' Compute the evolution of the proportion of hyperpolarization-activated
            cation current channels in open state.

            Kinetics scheme of Calcium dependent activation derived from:
            *Destexhe, A., Bal, T., McCormick, D.A., and Sejnowski, T.J. (1996). Ionic mechanisms
            underlying synchronized oscillations and propagating waves in a model of ferret
            thalamic slices. J. Neurophysiol. 76, 2049–2070.*

            :param Vm: membrane potential (mV)
            :param C: proportion of Ih channels in closed state (-)
            :param O: proportion of Ih channels in open state (-)
            :param P0: proportion of Ih channels regulating factor in unbound state (-)
            :return: derivative of proportion w.r.t. time (s-1)
        '''

        return - self.derC(C, O, Vm) - self.k3 * O * (1 - P0) + self.k4 * (1 - O - C)


    def derP0(self, P0, C_Ca):
        ''' Compute the evolution of the proportion of Ih channels regulating factor
            in unbound state.

            Kinetics scheme of Calcium dependent activation derived from:
            *Destexhe, A., Bal, T., McCormick, D.A., and Sejnowski, T.J. (1996). Ionic mechanisms
            underlying synchronized oscillations and propagating waves in a model of ferret
            thalamic slices. J. Neurophysiol. 76, 2049–2070.*

            :param Vm: membrane potential (mV)
            :param P0: proportion of Ih channels regulating factor in unbound state (-)
            :param C_Ca: Calcium concentration in effective submembranal space (M)
            :return: derivative of proportion w.r.t. time (s-1)
        '''

        return self.k2 * (1 - P0) - self.k1 * P0 * C_Ca**self.nCa


    def derC_Ca(self, C_Ca, ICa):
        ''' Compute the evolution of the Calcium concentration in submembranal space.

            Model of Ca2+ buffering and contribution from iCa derived from:
            *McCormick, D.A., and Huguenard, J.R. (1992). A model of the electrophysiological
            properties of thalamocortical relay neurons. J. Neurophysiol. 68, 1384–1400.*


            :param Vm: membrane potential (mV)
            :param C_Ca: Calcium concentration in submembranal space (M)
            :param ICa: inward Calcium current filling up the submembranal space with Ca2+ (mA/m2)
            :return: derivative of Calcium concentration in submembranal space w.r.t. time (s-1)
        '''

        return (self.CCa_min - C_Ca) / self.tau_Ca_removal - self.iT_2_CCa * ICa


    def currKL(self, Vm):
        ''' Compute the voltage-dependent leak Potassium current per unit area.

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        return self.GKL * (Vm - self.VK)


    def currH(self, O, C, Vm):
        ''' Compute the outward mixed cationic current per unit area.

            :param O: proportion of the channels in open form
            :param OL: proportion of the channels in locked-open form
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''

        OL = 1 - O - C
        return self.GhMax * (O + 2 * OL) * (Vm - self.Vh)


    def currNet(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        m, h, n, s, u, O, C, _, _ = states
        return (self.currNa(m, h, Vm) + self.currK(n, Vm)
                + self.currCa(s, u, Vm)
                + self.currKL(Vm)
                + self.currH(O, C, Vm)
                + self.currL(Vm))  # mA/m2


    def steadyStates(self, Vm):
        ''' Concrete implementation of the abstract API method. '''

        # Call parent method to compute Sodium, Potassium and Calcium channels gates steady-states
        NaKCa_eqstates = super().steadyStates(Vm)

        # Compute steady-state Calcium current
        seq = NaKCa_eqstates[3]
        ueq = NaKCa_eqstates[4]
        iTeq = self.currCa(seq, ueq, Vm)

        # Compute steady-state variables for the kinetics system of Ih
        CCa_eq = self.CCa_min - self.tau_Ca_removal * self.iT_2_CCa * iTeq
        BA = self.betao(Vm) / self.alphao(Vm)
        P0_eq = self.k2 / (self.k2 + self.k1 * CCa_eq**self.nCa)
        O_eq = self.k4 / (self.k3 * (1 - P0_eq) + self.k4 * (1 + BA))
        C_eq = BA * O_eq

        kin_eqstates = np.array([O_eq, C_eq, P0_eq, CCa_eq])

        # Merge all steady-states and return
        return np.concatenate((NaKCa_eqstates, kin_eqstates))


    def derStates(self, Vm, states):
        ''' Concrete implementation of the abstract API method. '''

        m, h, n, s, u, O, C, P0, C_Ca = states

        NaKCa_states = [m, h, n, s, u]
        NaKCa_derstates = super().derStates(Vm, NaKCa_states)

        dO_dt = self.derO(C, O, P0, Vm)
        dC_dt = self.derC(C, O, Vm)
        dP0_dt = self.derP0(P0, C_Ca)
        ICa = self.currCa(s, u, Vm)
        dCCa_dt = self.derC_Ca(C_Ca, ICa)

        return NaKCa_derstates + [dO_dt, dC_dt, dP0_dt, dCCa_dt]


    def getEffRates(self, Vm):
        ''' Concrete implementation of the abstract API method. '''

        # Compute effective coefficients for Sodium, Potassium and Calcium conductances
        NaKCa_effrates = super().getEffRates(Vm)

        # Compute effective coefficients for Ih conductance
        ao_avg = np.mean(self.alphao(Vm))
        bo_avg = np.mean(self.betao(Vm))
        iH_effrates = np.array([ao_avg, bo_avg])

        # Return array of coefficients
        return np.concatenate((NaKCa_effrates, iH_effrates))


    def derStatesEff(self, Qm, states, interp_data):
        ''' Concrete implementation of the abstract API method. '''

        rates = np.array([np.interp(Qm, interp_data['Q'], interp_data[rn])
                          for rn in self.coeff_names])
        Vmeff = np.interp(Qm, interp_data['Q'], interp_data['V'])

        # Unpack states
        m, h, n, s, u, O, C, P0, C_Ca = states

        # INa, IK, ICa effective states derivatives
        dmdt = rates[0] * (1 - m) - rates[1] * m
        dhdt = rates[2] * (1 - h) - rates[3] * h
        dndt = rates[4] * (1 - n) - rates[5] * n
        dsdt = rates[6] * (1 - s) - rates[7] * s
        dudt = rates[8] * (1 - u) - rates[9] * u

        # Ih effective states derivatives
        dC_dt = rates[11] * O - rates[10] * C
        dO_dt = - dC_dt - self.k3 * O * (1 - P0) + self.k4 * (1 - O - C)
        dP0_dt = self.derP0(P0, C_Ca)
        ICa_eff = self.currCa(s, u, Vmeff)
        dCCa_dt = self.derC_Ca(C_Ca, ICa_eff)

        # Merge derivatives and return
        return [dmdt, dhdt, dndt, dsdt, dudt, dO_dt, dC_dt, dP0_dt, dCCa_dt]
