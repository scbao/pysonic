# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-14 18:37:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-29 20:11:25

''' Test the basic functionalities of the package. '''

from PySONIC.core import BilayerSonophore, NeuronalBilayerSonophore
from PySONIC.utils import logger
from PySONIC.neurons import getPointNeuron, getNeuronsDict
from PySONIC.test import TestBase


class TestSims(TestBase):

    def test_MECH(self, is_profiled=False):
        logger.info('Test: running MECH simulation')
        a = 32e-9       # m
        Qm0 = -80e-5    # membrane resting charge density (C/m2)
        Cm0 = 1e-2      # membrane resting capacitance (F/m2)
        Fdrive = 350e3  # Hz
        Adrive = 100e3  # Pa
        Qm = 50e-5      # C/m2
        bls = BilayerSonophore(a, Cm0, Qm0)
        self.execute('bls.simulate(Fdrive, Adrive, Qm)', globals(), locals(), is_profiled)

    def test_ESTIM(self, is_profiled=False):
        logger.info('Test: running ESTIM simulation')
        Astim = 10.0     # mA/m2
        tstim = 100e-3   # s
        toffset = 50e-3  # s
        pneuron = getPointNeuron('RS')
        self.execute('pneuron.simulate(Astim, tstim, toffset)', globals(), locals(), is_profiled)

    def test_ASTIM_sonic(self, is_profiled=False):
        logger.info('Test: ASTIM sonic simulation')
        a = 32e-9        # m
        Fdrive = 500e3   # Hz
        Adrive = 100e3   # Pa
        tstim = 50e-3    # s
        toffset = 10e-3  # s
        pneuron = getPointNeuron('RS')
        nbls = NeuronalBilayerSonophore(a, pneuron)

        # test error 1: sonophore radius outside of lookup range
        try:
            nbls = NeuronalBilayerSonophore(100e-9, pneuron)
            nbls.simulate(Fdrive, Adrive, tstim, toffset, method='sonic')
        except ValueError:
            logger.debug('Out of range radius: OK')

        # test error 2: frequency outside of lookups range
        try:
            nbls = NeuronalBilayerSonophore(a, pneuron)
            nbls.simulate(10e3, Adrive, tstim, toffset, method='sonic')
        except ValueError:
            logger.debug('Out of range frequency: OK')

        # test error 3: amplitude outside of lookups range
        try:
            nbls = NeuronalBilayerSonophore(a, pneuron)
            nbls.simulate(Fdrive, 1e6, tstim, toffset, method='sonic')
        except ValueError:
            logger.debug('Out of range amplitude: OK')

        # Run simulation on all neurons
        for name, neuron_class in getNeuronsDict().items():
            if name not in ('template', 'LeechP', 'LeechT', 'LeechR', 'sweeney'):
                pneuron = neuron_class()
                nbls = NeuronalBilayerSonophore(a, pneuron)
                self.execute("nbls.simulate(Fdrive, Adrive, tstim, toffset, method='sonic')",
                             globals(), locals(), is_profiled)

    def test_ASTIM_full(self, is_profiled=False):
        logger.info('Test: running ASTIM detailed simulation')
        a = 32e-9       # m
        Fdrive = 500e3  # Hz
        Adrive = 100e3  # Pa
        tstim = 1e-6    # s
        toffset = 1e-6  # s
        pneuron = getPointNeuron('RS')
        nbls = NeuronalBilayerSonophore(a, pneuron)
        self.execute("nbls.simulate(Fdrive, Adrive, tstim, toffset, method='full')",
                     globals(), locals(), is_profiled)

    def test_ASTIM_hybrid(self, is_profiled=False):
        logger.info('Test: running ASTIM hybrid simulation')
        a = 32e-9         # m
        Fdrive = 350e3    # Hz
        Adrive = 100e3    # Pa
        tstim = 0.6e-3    # s
        toffset = 0.1e-3  # s
        pneuron = getPointNeuron('RS')
        nbls = NeuronalBilayerSonophore(a, pneuron)
        self.execute("nbls.simulate(Fdrive, Adrive, tstim, toffset, method='hybrid')",
                     globals(), locals(), is_profiled)


if __name__ == '__main__':
    tester = TestSims()
    tester.main()
