# -*- coding: utf-8 -*-
# @Author: Mariia Popova
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-10-03 15:58:38
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-05 18:12:25

import numpy as np
from ..core import PointNeuron
from ..constants import CELSIUS_2_KELVIN, FARADAY, Rg, Z_Ca
from ..utils import findModifiedEq


class Sundt(PointNeuron):
    ''' Sundt neuron only sodium and delayed-rectifier potassium currents

        Reference:
        *Sundt D., Gamper N., Jaffe D. B., Spike propagation through the dorsal
        root ganglia in an unmyelinated sensory neuron: a modeling study.
        Journal of Neurophysiology (2015)*
    '''

    # Neuron name
    name = 'sundt'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2   # Membrane capacitance (F/m2)
    Vm0 = -60.   # Membrane potential (mV)

    # Reversal potentials (mV)
    ENa = 55.0     # Sodium
    EK = -90.0     # Potassium

    # Maximal channel conductances (S/m2)
    gNabar = 400.0    # Sodium
    gKdbar = 400.0    # Delayed-rectifier Potassium
    gKmbar = 4.0      # KCNQ Potassium
    gCaLbar = 30      # Calcium ????
    gKCabar = 2.0     # Calcium dependent Potassium ????
    gLeak = 1e0       # Non-specific leakage

    # Additional parameters
    Cao = 2e-3             # Extracellular Calcium concentration (M)
    Cai0 = 70e-9           # Intracellular Calcium concentration at rest (M) (Aradi 1999)
    celsius = 35.0         # Temperature (Celsius)
    celsius_Traub = 30.0   # Temperature in Traub 1991 (Celsius)
    celsius_Yamada = 23.5  # Temperature in Yamada 1989 (Celsius)

    # Na+ current parameters
    deltaVm = 6.0    # Voltage offset to shift the rate constants  (6 mV in Sundt 2015)

    # K+ current parameters (Borg Graham 1987 for the formalism, Migliore 1995 for the values)
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

    # Ca2+ parameters
    Ca_factor = 1e6   # conversion factor for q-gate Calcium sensitivity (expressed in uM)
    Ca_power = 3      # power exponent for q-gate Calcium sensitivity (-)
    deff = 200e-9     # effective depth beneath membrane for intracellular [Ca2+] calculation (m)
    taur_Cai = 20e-3  # decay time constant for intracellular Ca2+ dissolution (s)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
        'm': 'iNa activation gate',
        'h': 'iNa inactivation gate',
        'n': 'iKdr gate',
        'l': 'iKdr Borg-Graham formalism gate',
        'mkm': 'iKm gate',
        'c': 'iCa gate',
        'q': 'iK Calcium dependent gate',
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
        # print(f'iNet = {iNet:.2f} mA/m2')

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

    # Sodium kinetics: adapted from Traub 1991, with a q10 = sqrt(3) to account for temperature
    # adaptation from 30 to 35 degrees, and a voltage offset of DV = +6 mV shifting the activation
    # and inactivation rates profiles.

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

    # Potassium kinetics: using Migliore 1995 values, with Borg-Graham 1991 formalism

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

    # KCNQ Potassium kinetics: taken from Yamada 1989 (cannot find source...), with
    # Q10 adaptation from 23.5 to 35 degrees.

    @staticmethod
    def mkminf(Vm):
        return 1.0 / (1 + np.exp(-(Vm + 35) / 10))

    @classmethod
    def taumkm(cls, Vm):
        return 1e-3 / (3.3 * (np.exp((Vm + 35) / 20) + np.exp(-(Vm + 35) / 20)) / cls.q10_Yamada)  # s

    # L-type Calcium kinetics: from Migliore 1995 that itself refers to Jaffe 1994.

    @classmethod
    def alphac(cls, Vm):
        return 15.69 * cls.vtrap((81.5 - Vm), 10.) * 1e3  # s-1

    @classmethod
    def betac(cls, Vm):
        return 0.29 * np.exp(-Vm / 10.86) * 1e3  # s-1

    # Calcium-dependent Potassium kinetics: from Aradi 1999, correcting error in alphaq denominator
    # (4.5 vs 4).
    # - 3 (vs. 1) in Cai exponent

    @classmethod
    def alphaq(cls, Cai):
        return 0.00246 / np.exp((12 * np.log10(np.power(Cai * cls.Ca_factor, cls.Ca_power)) + 28.48) / -4.5) * 1e3  # s-1

    @classmethod
    def betaq(cls, Cai):
        return 0.006 / np.exp((12 * np.log10(np.power(Cai * cls.Ca_factor, cls.Ca_power)) + 60.4) / 35) * 1e3  # s-1


    # ------------------------------ States derivatives ------------------------------

    # Ca2+ dynamics: discrepancy in dissolution rate between Sundt (20 ms) and Aradi ref. (9 ms)

    @classmethod
    def derCai(cls, c, Cai, Vm):
        return - cls.current_to_molar_rate_Ca * cls.iCaL(c, Cai, Vm) - (Cai - cls.Cai0) / cls.taur_Cai  # M/s

    @classmethod
    def ECa(cls, Cai):
        ''' Calcium reversal potential '''
        return 1e3 * np.log(cls.Cao / Cai) * cls.T * Rg / (Z_Ca * FARADAY)  # mV

    @classmethod
    def derStates(cls):
        return {
            'm': lambda Vm, x: cls.alpham(Vm) * (1 - x['m']) - cls.betam(Vm) * x['m'],
            'h': lambda Vm, x: cls.alphah(Vm) * (1 - x['h']) - cls.betah(Vm) * x['h'],
            'n': lambda Vm, x: cls.alphan(Vm) * (1 - x['n']) - cls.betan(Vm) * x['n'],
            'l': lambda Vm, x: cls.alphal(Vm) * (1 - x['l']) - cls.betal(Vm) * x['l'],
            'mkm': lambda Vm, x: (cls.mkminf(Vm) - x['mkm']) / cls.taumkm(Vm),
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
            'mkm': lambda Vm: cls.mkminf(Vm),
            'c': lambda Vm: cls.alphac(Vm) / (cls.alphac(Vm) + cls.betac(Vm)),
        }
        lambda_dict['Cai'] = lambda Vm: cls.Caiinf(lambda_dict['c'](Vm), Vm)
        lambda_dict['q'] = lambda Vm: cls.qinf(lambda_dict['Cai'](Vm))
        return lambda_dict

    # ------------------------------ Membrane currents ------------------------------

    # Sodium current: inconsistency with 1991 ref: m2h vs. m3h

    @classmethod
    def iNa(cls, m, h, Vm):
        ''' Sodium current '''
        return cls.gNabar * m**3 * h * (Vm - cls.ENa)  # mA/m2

    @classmethod
    def iKd(cls, n, l, Vm):
        ''' delayed-rectifier Potassium current '''
        return cls.gKdbar * n**3 * l * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iKm(cls, mkm, Vm):
        ''' slowly activating Potassium current '''
        return cls.gKmbar * mkm * (Vm - cls.EK)  # mA/m2

    @classmethod
    def iCaL(cls, c, Cai, Vm):
        ''' Calcium current '''
        return cls.gCaLbar * c**2 * (Vm - cls.ECa(Cai))  # mA/m2

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
            'iKm': lambda Vm, x: cls.iKm(x['mkm'], Vm),
            'iCaL': lambda Vm, x: cls.iCaL(x['c'], x['Cai'], Vm),
            'iKCa': lambda Vm, x: cls.iKCa(x['q'], Vm),
            'iLeak': lambda Vm, _: cls.iLeak(Vm)
        }

    def chooseTimeStep(self):
        return super().chooseTimeStep() * 1e-2