# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-07-07 16:56:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-07-10 12:46:14

from ..core import PointNeuron, addSonicFeatures


@addSonicFeatures
class PassiveNeuron(PointNeuron):
    ''' Generic point-neuron model with only a passive current. '''

    states = {}

    def __new__(cls, Cm0, gLeak, ELeak):
        ''' Initialization.

            :param Cm0: membrane capacitance (F/m2)
            :param gLeak: leakage conductance (S/m2)
            :param ELeak: leakage revwersal potential (mV)
        '''
        cls.Cm0 = Cm0
        cls.gLeak = gLeak
        cls.ELeak = ELeak
        return super(PassiveNeuron, cls).__new__(cls)

    @property
    def name(self):
        return f'pas_tau_{self.tau_pas * 1e3:.2e}ms'

    @property
    def Cm0(self):
        return self._Cm0

    @Cm0.setter
    def Cm0(self, value):
        self._Cm0 = value

    @property
    def Vm0(self):
        return self.ELeak

    @classmethod
    def derStates(cls):
        return {}

    @classmethod
    def steadyStates(cls):
        return {}

    @classmethod
    def iLeak(cls, Vm):
        ''' non-specific leakage current '''
        return cls.gLeak * (Vm - cls.ELeak)  # mA/m2

    @classmethod
    def currents(cls):
        return {'iLeak': lambda Vm, _: cls.iLeak(Vm)}
