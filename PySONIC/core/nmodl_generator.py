# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2019-03-18 21:17:03
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-18 21:18:02


import inspect
import re
from time import gmtime, strftime


def escaped_pow(x):
    return ' * '.join([x.group(1)] * int(x.group(2)))


class NmodlGenerator:

    tabreturn = '\n   '
    NEURON_protected_vars = ['O', 'C']

    def __init__(self, neuron):
        self.neuron = neuron
        self.translated_states = [self.translateState(s) for s in self.neuron.states]

    def print(self, outfile):
        all_blocks = [
            self.title(),
            self.description(),
            self.constants(),
            self.tscale(),
            self.neuron_block(),
            self.parameter_block(),
            self.state_block(),
            self.assigned_block(),
            self.function_tables(),
            self.initial_block(),
            self.breakpoint_block(),
            self.derivative_block()
        ]
        with open(outfile, "w") as fh:
            fh.write('\n\n'.join(all_blocks))

    def translateState(self, state):
        return '{}{}'.format(state, '1' if state in self.NEURON_protected_vars else '')

    def title(self):
        return 'TITLE {} membrane mechanism'.format(self.neuron.name)

    def description(self):
        return '\n'.join([
            'COMMENT',
            self.neuron.getDesc(),
            '',
            '@Author: Theo Lemaire, EPFL',
            '@Date: {}'.format(strftime("%Y-%m-%d", gmtime())),
            '@Email: theo.lemaire@epfl.ch',
            'ENDCOMMENT'
        ])

    def constants(self):
        block = [
            'FARADAY = 96494     (coul)     : moles do not appear in units',
            'R = 8.31342         (J/mol/K)  : Universal gas constant'
        ]
        return 'CONSTANT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def tscale(self):
        return 'INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}'

    def neuron_block(self):
        block = [
            'SUFFIX {}'.format(self.neuron.name),
            '',
            ': Constituting currents',
            *['NONSPECIFIC_CURRENT {}'.format(i) for i in self.neuron.getCurrentsNames()],
            '',
            ': RANGE variables',
            'RANGE Adrive, Vmeff : section specific',
            'RANGE stimon : common to all sections (but set as RANGE to be accessible from caller)'
        ]
        return 'NEURON {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def parameter_block(self):
        block = [
            ': Parameters set by python/hoc caller',
            'stimon : Stimulation state',
            'Adrive (kPa) : Stimulation amplitude',
            '',
            ': Membrane properties',
            'cm = {} (uF/cm2)'.format(self.neuron.Cm0 * 1e2)
        ]

        # Reversal potentials
        possibles_E = list(set(['Na', 'K', 'Ca'] + [i[1:] for i in self.neuron.getCurrentsNames()]))
        for x in possibles_E:
            nernst_pot = 'E{}'.format(x)
            if hasattr(self.neuron, nernst_pot):
                block.append('{} = {} (mV)'.format(
                    nernst_pot, getattr(self.neuron, nernst_pot)))

        # Conductances / permeabilities
        for i in self.neuron.getCurrentsNames():
            suffix = '{}{}'.format(i[1:], '' if 'Leak' in i else 'bar')
            factors = {'g': 1e-4, 'p': 1e2}
            units = {'g': 'S/cm2', 'p': 'cm/s'}
            for prefix in ['g', 'p']:
                attr = '{}{}'.format(prefix, suffix)
                if hasattr(self.neuron, attr):
                    val = getattr(self.neuron, attr) * factors[prefix]
                    block.append('{} = {} ({})'.format(attr, val, units[prefix]))

        return 'PARAMETER {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def state_block(self):
        block = [': Standard gating states', *self.translated_states]
        return 'STATE {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def assigned_block(self):
        block = [
            ': Variables computed during the simulation and whose value can be retrieved',
            'Vmeff (mV)',
            'v (mV)',
            *['{} (mA/cm2)'.format(i) for i in self.neuron.getCurrentsNames()]
        ]
        return 'ASSIGNED {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def function_tables(self):
        block = [
            ': Function tables to interpolate effective variables',
            'FUNCTION_TABLE V(A(kPa), Q(nC/cm2)) (mV)',
            *['FUNCTION_TABLE {}(A(kPa), Q(nC/cm2)) (mV)'.format(r) for r in self.neuron.rates]
        ]
        return '\n'.join(block)

    def initial_block(self):
        block = [
            ': Set initial states values'
        ]
        for g in self.neuron.getGates():
            block.append('{0} = alpha{1}(0, v) / (alpha{1}(0, v) + beta{1}(0, v))'.format(
                self.translateState(g), g.lower()))

        return 'INITIAL {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def breakpoint_block(self):
        block = [
            ': Integrate states',
            'SOLVE states METHOD cnexp',
            '',
            ': Compute effective membrane potential',
            'Vmeff = V(Adrive * stimon, v)',
            '',
            ': Compute ionic currents'
        ]
        for i in self.neuron.getCurrentsNames():
            func_exp = inspect.getsource(getattr(self.neuron, i)).splitlines()[-1]
            func_exp = func_exp[func_exp.find('return') + 7:]
            func_exp = func_exp.replace('self.', '').replace('Vm', 'Vmeff')
            func_exp = re.sub(r'([A-Za-z][A-Za-z0-9]*)\*\*([0-9])', escaped_pow, func_exp)
            block.append('{} = {}'.format(i, func_exp))

        return 'BREAKPOINT {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))

    def derivative_block(self):
        block = [': Gating states derivatives']
        for g in self.neuron.getGates():
            block.append(
                '{0}\' = alpha{1}{2} * (1 - {0}) - beta{1}{2} * {0}'.format(
                    self.translateState(g), g.lower(), '(Adrive * stimon, v)')
            )

        return 'DERIVATIVE states {{{}{}\n}}'.format(self.tabreturn, self.tabreturn.join(block))