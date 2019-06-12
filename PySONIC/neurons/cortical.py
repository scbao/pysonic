# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-07-31 15:19:51
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-12 23:04:46

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

    # ------------------------------ States derivatives ------------------------------

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

    def derStates(self, Vm, states):
        m, h, n, p = states
        return {
            'm': self.derM(Vm, m),
            'h': self.derH(Vm, h),
            'n': self.derN(Vm, n),
            'p': self.derP(Vm, p)
        }

    def derEffStates(self, Qm, states, lkp):
        rates = self.interpEffRates(Qm, lkp)
        m, h, n, p = states
        return {
            'm': rates['alpham'] * (1 - m) - rates['betam'] * m,
            'h': rates['alphah'] * (1 - h) - rates['betah'] * h,
            'n': rates['alphan'] * (1 - n) - rates['betan'] * n,
            'p': rates['alphap'] * (1 - p) - rates['betap'] * p
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

    def iM(self, p, Vm):
        ''' slow non-inactivating Potassium current

            :param p: open-probability of p-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gMbar * p * (Vm - self.EK)

    def iLeak(self, Vm):
        ''' non-specific leakage current

            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gLeak * (Vm - self.ELeak)

    def currents(self, Vm, states):
        m, h, n, p = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iKd': self.iKd(n, Vm),
            'iM': self.iM(p, Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2

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

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n', 'p')


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

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n', 'p')


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

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n', 'p', 's', 'u')

    # ------------------------------ Gating states kinetics ------------------------------

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

    def derStates(self, Vm, states):
        # Unpack input states
        *NaK_states, s, u = states

        # Call parent method to compute Sodium and Potassium channels states derivatives
        dstates = super().derStates(Vm, NaK_states)

        # Compute Calcium channels states derivatives
        dstates['s'] = self.derS(Vm, s)
        dstates['u'] = self.derU(Vm, u)

        return dstates

    def derEffStates(self, Qm, states, lkp):
        # Unpack input states
        *NaK_states, s, u = states

        # Call parent method to compute Sodium and Potassium channels states derivatives
        dstates = super().derEffStates(Qm, NaK_states, lkp)

        # Compute Calcium channels states derivatives
        Ca_rates = self.interpEffRates(Qm, lkp, keys=self.getRatesNames(['s', 'u']))
        dstates['s'] = Ca_rates['alphas'] * (1 - s) - Ca_rates['betas'] * s
        dstates['u'] = Ca_rates['alphau'] * (1 - u) - Ca_rates['betau'] * u

        # Merge all states derivatives and return
        return dstates

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
        # Voltage-gated steady-states
        sstates = super().steadyStates(Vm)
        sstates['s'] = self.sinf(Vm)
        sstates['u'] = self.uinf(Vm)
        return sstates

    # ------------------------------ Membrane currents ------------------------------

    def iCaT(self, s, u, Vm):
        ''' low-threshold (T-type) Calcium current

            :param s: open-probability of s-gate (-)
            :param u: open-probability of u-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gCaTbar * s**2 * u * (Vm - self.ECa)

    def currents(self, Vm, states):
        m, h, n, p, s, u = states
        currents = super().currents(Vm, [m, h, n, p])
        currents['iCaT'] = self.iCaT(s, u, Vm)  # mA/m2
        return currents

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        # Call parent method to compute Sodium and Potassium effective rate constants
        effrates = super().computeEffRates(Vm)

        # Compute Calcium effective rate constants
        effrates['alphas'] = np.mean(self.sinf(Vm) / self.taus(Vm))
        effrates['betas'] = np.mean((1 - self.sinf(Vm)) / self.taus(Vm))
        effrates['alphau'] = np.mean(self.uinf(Vm) / self.tauu(Vm))
        effrates['betau'] = np.mean((1 - self.uinf(Vm)) / self.tauu(Vm))

        return effrates


class CorticalIB(Cortical):
    ''' Cortical intrinsically bursting neuron

        References:
        *Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., Bal, T., Frégnac,
        Y., Markram, H., and Destexhe, A. (2008). Minimal Hodgkin-Huxley type models for
        different classes of cortical and thalamic neurons. Biol Cybern 99, 427–441.*

        *Reuveni, I., Friedman, A., Amitai, Y., and Gutnick, M.J. (1993). Stepwise repolarization
        from Ca2+ plateaus in neocortical pyramidal cells: evidence for nonhomogeneous distribution
        of HVA Ca2+ channels in dendrites. J. Neurosci. 13, 4609–4621.*
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

    # ------------------------------ States names (ordered) ------------------------------
    states = ('m', 'h', 'n', 'p', 'q', 'r')

    # ------------------------------ Gating states kinetics ------------------------------

    def alphaq(self, Vm):
        ''' Voltage-dependent activation rate of q-gate

            :param Vm: membrane potential (mV)
            :return: rate (s-1)
        '''
        alpha = 0.055 * self.vtrap(-(Vm + 27), 3.8)  # ms-1
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

    # ------------------------------ States derivatives ------------------------------

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

    def derStates(self, Vm, states):
        # Unpack input states
        *NaK_states, q, r = states

        # Call parent method to compute Sodium and Potassium channels states derivatives
        dstates = super().derStates(Vm, NaK_states)

        # Compute L-type Calcium channels states derivatives
        dstates['q'] = self.derQ(Vm, q)
        dstates['r'] = self.derR(Vm, r)

        # Merge all states derivatives and return
        return dstates

    def derEffStates(self, Qm, states, lkp):

        # Unpack input states
        *NaK_states, q, r = states

        # Call parent method to compute Sodium and Potassium channels states derivatives
        dstates = super().derEffStates(Qm, NaK_states, lkp)

        # Compute Calcium channels states derivatives
        Ca_rates = self.interpEffRates(Qm, lkp, keys=self.getRatesNames(['q', 'r']))
        dstates['q'] = Ca_rates['alphaq'] * (1 - q) - Ca_rates['betaq'] * q
        dstates['r'] = Ca_rates['alphar'] * (1 - r) - Ca_rates['betar'] * r

        # Merge all states derivatives and return
        return dstates

    # ------------------------------ Steady states ------------------------------

    def steadyStates(self, Vm):
        # Voltage-gated steady-states
        sstates = super().steadyStates(Vm)
        sstates['q'] = self.alphaq(Vm) / (self.alphaq(Vm) + self.betaq(Vm))
        sstates['r'] = self.alphar(Vm) / (self.alphar(Vm) + self.betar(Vm))
        return sstates

    # ------------------------------ Membrane currents ------------------------------

    def iCaL(self, q, r, Vm):
        ''' high-threshold (L-type) Calcium current

            :param q: open-probability of q-gate (-)
            :param r: open-probability of r-gate (-)
            :param Vm: membrane potential (mV)
            :return: current per unit area (mA/m2)
        '''
        return self.gCaLbar * q**2 * r * (Vm - self.ECa)

    def currents(self, Vm, states):
        m, h, n, p, q, r = states
        return {
            'iNa': self.iNa(m, h, Vm),
            'iKd': self.iKd(n, Vm),
            'iM': self.iM(p, Vm),
            'iCaL': self.iCaL(q, r, Vm),
            'iLeak': self.iLeak(Vm)
        }  # mA/m2

    # ------------------------------ Other methods ------------------------------

    def computeEffRates(self, Vm):
        # Call parent method to compute Sodium and Potassium effective rate constants
        effrates = super().computeEffRates(Vm)

        # Compute Calcium effective rate constants
        effrates['alphaq'] = np.mean(self.alphaq(Vm))
        effrates['betaq'] = np.mean(self.betaq(Vm))
        effrates['alphar'] = np.mean(self.alphar(Vm))
        effrates['betar'] = np.mean(self.betar(Vm))

        return effrates
