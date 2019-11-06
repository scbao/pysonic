# -*- coding: utf-8 -*-
# @Author: Mariia Popova
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-10-03 15:58:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-06 10:22:31

import numpy as np
from ..core import PointNeuron
from ..constants import CELSIUS_2_KELVIN, FARADAY, Rg, Z_Ca
from ..utils import findModifiedEq


class Sundt(PointNeuron):
    ''' Unmyelinated C-fiber model.

        Reference:
        *Sundt D., Gamper N., Jaffe D. B., Spike propagation through the dorsal
        root ganglia in an unmyelinated sensory neuron: a modeling study.
        Journal of Neurophysiology (2015)*
    '''

    # Neuron name
    name = 'sundt'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2  # Membrane capacitance (F/m2)
    Vm0 = -60.  # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 55.0  # Sodium
    EK = -90.0  # Potassium

    # Maximal channel conductances (S/m2)
    gNabar = 400.0  # Sodium
    gKdbar = 400.0  # Delayed-rectifier Potassium
    gMbar = 4.0     # Slow non-inactivating Potassium
    gCaLbar = 30    # High-threshold Calcium (???)
    gKCabar = 2.0   # Calcium dependent Potassium (???)
    gLeak = 1e0     # Non-specific leakage

    # Na+ current parameters
    deltaVm = 6.0  # Voltage offset to shift the rate constants  (6 mV in Sundt 2015)

    # Kd current parameters (Borg Graham 1987 for the formalism, Migliore 1995 for the values)
    alphan0 = 0.03    # ms-1
    alphal0 = 0.001   # ms-1
    betan0 = alphan0  # ms-1
    betal0 = alphal0  # ms-1
    Vhalfn = -32      # Membrane voltage at which alphan = alphan0 and betan = betan0 (mV)
    Vhalfl = -61      # Membrane voltage at which alphal = alphal0 and betal = betal0 (mV)
    zn = 5            # Effective valence of the n-gating particle
    zl = -2           # Effective valence of the l-gating particle
    gamman = 0.4      # Normalized position of the n-transition state within the membrane
    gammal = 1        # Normalized position of the l-transition state within the membrane

    # iM parameters
    taupMax = 1.0  # Max. adaptation decay of slow non-inactivating Potassium current (s)

    # Ca2+ parameters
    Cao = 2e-3        # Extracellular Calcium concentration (M)
    Cai0 = 70e-9      # Intracellular Calcium concentration at rest (M) (Aradi 1999)
    deff = 200e-9     # effective depth beneath membrane for intracellular [Ca2+] calculation (m)
    taur_Cai = 20e-3  # decay time constant for intracellular Ca2+ dissolution (s)

    # iKCa parameters
    Ca_factor = 1e6  # conversion factor for q-gate Calcium sensitivity (expressed in uM)
    Ca_power = 3     # power exponent for q-gate Calcium sensitivity (-)

    # Additional parameters
    celsius = 35.0         # Temperature (Celsius)
    celsius_Traub = 30.0   # Temperature in Traub 1991 (Celsius)
    celsius_Yamada = 23.5  # Temperature in Yamada 1989 (Celsius)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKd activation gate',
        'l': 'iKd inactivation gate',
        'p': 'iM gate',
        'c': 'iCaL gate',
        'q': 'iKCa Calcium dependent gate',
        'Cai': 'Calcium intracellular concentration (M)'
    }

    def __new__(cls):
        cls.q10_Traub = 3**((cls.celsius - cls.celsius_Traub) / 10)
        cls.q10_Yamada = 3**((cls.celsius - cls.celsius_Yamada) / 10)
        cls.T = cls.celsius + CELSIUS_2_KELVIN
        cls.current_to_molar_rate_Ca = cls.currentToConcentrationRate(Z_Ca, cls.deff)
        cls.Vref = Rg * cls.T / FARADAY * 1e3  # reference voltagte for iKd rate constants (mV)

        # Compute total current at resting potential, without iLeak
        sstates = {k: cls.steadyStates()[k](cls.Vm0) for k in cls.statesNames()}
        i_dict = cls.currents()
        del i_dict['iLeak']
        iNet = sum([cfunc(cls.Vm0, sstates) for cfunc in i_dict.values()])  # mA/m2

        # Compute Eleak such that iLeak cancels out the net current at resting potential
        cls.ELeak = cls.Vm0 + iNet / cls.gLeak  # mV
        # print(f'Eleak = {cls.ELeak:.2f} mV')

        return super(Sundt, cls).__new__(cls)


    @classmethod
    def getPltScheme(cls):
        pltscheme = super().getPltScheme()
        pltscheme['[Ca^{2+}]_i'] = ['Cai']
        return pltscheme

    @classmethod
    def getPltVars(cls, wrapleft='df["', wrapright='"]'):
        return {**super().getPltVars(wrapleft, wrapright), **{
            'Cai': {
                'desc': 'sumbmembrane Ca2+ concentration',
                'label': '[Ca^{2+}]_i',
                'unit': 'uM',
                'factor': 1e6
            }
        }}

    # ------------------------------ Gating states kinetics ------------------------------

    # iNa kinetics: adapted from Traub 1991, with 2 notable changes:
    # - Q10 correction to account for temperature adaptation from 30 to 35 degrees
    # - 6 mV voltage offset in the activation and inactivation rates to shift iNa voltage dependence
    #   approximately midway between values reported for Nav1.7 and Nav1.8 currents.

    @classmethod
    def alpham(cls, Vm):
        Vm += cls.deltaVm
        return cls.q10_Traub * 0.32 * cls.vtrap((13.1 - Vm), 4) * 1e3  # s-1

    @classmethod
    def betam(cls, Vm):
        Vm += cls.deltaVm
        return cls.q10_Traub * 0.28 * cls.vtrap((Vm - 40.1), 5) * 1e3  # s-1

    @classmethod
    def alphah(cls, Vm):
        Vm += cls.deltaVm
        return cls.q10_Traub * 0.128 * np.exp((17.0 - Vm) / 18) * 1e3  # s-1

    @classmethod
    def betah(cls, Vm):
        Vm += cls.deltaVm
        return cls.q10_Traub * 4 / (1 + np.exp((40.0 - Vm) / 5)) * 1e3 # s-1

    # iKd kinetics: using Migliore 1995 values, with Borg-Graham 1991 formalism

    @classmethod
    def alphan(cls, Vm):
        return cls.alphan0 * np.exp(cls.zn * cls.gamman * (Vm - cls.Vhalfn) / cls.Vref) * 1e3  # s-1

    @classmethod
    def betan(cls, Vm):
        return cls.betan0 * np.exp(-cls.zn * (1 - cls.gamman) * (Vm - cls.Vhalfn) / cls.Vref) * 1e3  # s-1

    @classmethod
    def alphal(cls, Vm):
        return cls.alphal0 * np.exp(cls.zl * cls.gammal * (Vm - cls.Vhalfl) / cls.Vref) * 1e3  # s-1

    @classmethod
    def betal(cls, Vm):
        return cls.betal0 * np.exp(-cls.zl * (1 - cls.gammal) * (Vm - cls.Vhalfl) / cls.Vref) * 1e3  # s-1

    # iM kinetics: taken from Yamada 1989, with notable changes:
    # - Q10 correction to account for temperature adaptation from 23.5 to 35 degrees
    # - not sure about tau_p formulation (3.3 factor multiplying first-only or both exponential terms ???)

    @staticmethod
    def pinf(Vm):
        return 1.0 / (1 + np.exp(-(Vm + 35) / 10))

    @classmethod
    def taup(cls, Vm):
        tau = cls.taupMax / (3.3 * (np.exp((Vm + 35) / 20) + np.exp(-(Vm + 35) / 20)))  # s
        return tau * cls.q10_Yamada

    # iCaL kinetics: from Migliore 1995 that itself refers to Jaffe 1994.

    @classmethod
    def alphac(cls, Vm):
        return 15.69 * cls.vtrap((81.5 - Vm), 10.) * 1e3  # s-1

    @classmethod
    def betac(cls, Vm):
        return 0.29 * np.exp(-Vm / 10.86) * 1e3  # s-1

    # iKCa kinetics: from Aradi 1999, which uses equations from Yuen 1991 with a few modifications:
    # - 12 mV (???) shift in activation curve
    # - log10 instead of log for Ca2+ sensitivity
    # - global dampening factor of 1.67 applied on both rates
    # Sundt 2015 applies an extra modification:
    # - higher Calcium sensitivity (third power of Ca concentration)
    # Also, there is an error in the alphaq denominator in the paper: using -4 instead of -4.5

    @classmethod
    def alphaq(cls, Cai):
        return 0.00246 / np.exp((12 * np.log10((Cai * cls.Ca_factor)**cls.Ca_power) + 28.48) / -4.5) * 1e3  # s-1

    @classmethod
    def betaq(cls, Cai):
        return 0.006 / np.exp((12 * np.log10((Cai * cls.Ca_factor)**cls.Ca_power) + 60.4) / 35) * 1e3  # s-1


    # ------------------------------ States derivatives ------------------------------

    # Ca2+ dynamics: using accumulation-dissolution formalism as in Aradi, with
    # a longer Ca2+ intracellular dissolution time constant (20 ms vs. 9 ms)

    @classmethod
    def derCai(cls, c, Cai, Vm):
        return -cls.current_to_molar_rate_Ca * cls.iCaL(c, Cai, Vm) - (Cai - cls.Cai0) / cls.taur_Cai  # M/s

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n'],
            'l': lambda Vm, x: cls.alphal(Vm) * (1 - x['l']) - cls.betal(Vm) * x['l'],
            'p': lambda Vm, x: (cls.pinf(Vm) - x['p']) / cls.taup(Vm),
            'c': lambda Vm, x: cls.alphac(Vm) * (1 - x['c']) - cls.betac(Vm) * x['c'],
            'q': lambda Vm, x: cls.alphaq(x['Cai']) * (1 - x['q']) - cls.betaq(x['Cai']) * x['q'],
            'Cai': lambda Vm, x: cls.derCai(x['c'], x['Cai'], Vm)
        }

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def qinf(cls, Cai):
        return cls.alphaq(Cai) / (cls.alphaq(Cai) + cls.betaq(Cai))

    @classmethod
    def Caiinf(cls, c, Vm):
        return findModifiedEq(
            cls.Cai0,
            lambda Cai, c, Vm: cls.derCai(c, Cai, Vm),
            c, Vm
        )

    @classmethod
    def steadyStates(cls):
        lambda_dict = {
            'm': lambda Vm: cls.alpham(Vm) / (cls.alpham(Vm) + cls.betam(Vm)),
            'h': lambda Vm: cls.alphah(Vm) / (cls.alphah(Vm) + cls.betah(Vm)),
            'n': lambda Vm: cls.alphan(Vm) / (cls.alphan(Vm) + cls.betan(Vm)),
            'l': lambda Vm: cls.alphal(Vm) / (cls.alphal(Vm) + cls.betal(Vm)),
            'p': lambda Vm: cls.pinf(Vm),
            'c': lambda Vm: cls.alphac(Vm) / (cls.alphac(Vm) + cls.betac(Vm)),
        }
        lambda_dict['Cai'] = lambda Vm: cls.Caiinf(lambda_dict['c'](Vm), Vm)
        lambda_dict['q'] = lambda Vm: cls.qinf(lambda_dict['Cai'](Vm))
        return lambda_dict

    # ------------------------------ Membrane currents ------------------------------

    # Sodium current: inconsistency with 1991 ref: m2h vs. m3h

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current.

            Gating formalism from Migliore 1995, using 3rd power for m in order to
            reproduce thinner AP waveform (half-width of ca. 1 ms)
        '''
        return cls.gNabar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKd(cls, n, l, Vm):
        ''' delayed-rectifier Potassium current '''
        return cls.gKdbar * n**3 * l * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iM(cls, p, Vm):
        ''' slow non-inactivating Potassium current '''
        return cls.gMbar * p * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iCaL(cls, c, Cai, Vm):
        ''' Calcium current '''
        ECa = cls.nernst(Z_Ca, Cai, cls.Cao, cls.T)  # mV
        return cls.gCaLbar * c**2 * (Vm - ECa)  # mA/m2

    @classmethod
    def iKCa(cls, q, Vm):
        ''' Calcium-dependent Potassium current '''
        return cls.gKCabar * q**2 * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {
            'iNa': lambda Vm, x: cls.iNa(x['m'], x['h'], Vm),
            'iKd': lambda Vm, x: cls.iKd(x['n'], x['l'], Vm),
            'iM': lambda Vm, x: cls.iM(x['p'], Vm),
            'iCaL': lambda Vm, x: cls.iCaL(x['c'], x['Cai'], Vm),
            'iKCa': lambda Vm, x: cls.iKCa(x['q'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }

    def chooseTimeStep(self):
        ''' neuron-specific time step for fast dynamics. '''
        return super().chooseTimeStep() * 1e-2