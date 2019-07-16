# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-29 11:26:27
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-16 18:55:02

from time import gmtime, strftime
import re
import inspect

from ..constants import *


class Translator:
    '''Generic Translator interface. '''

    # Generic regexp patterns
    variable_pattern = r'[a-zA-Z_][a-zA-Z0-9_]*'
    integer_pattern = r'[0-9]+'
    float_pattern = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
    strict_func_pattern = r'({})\('.format(variable_pattern)
    loose_func_pattern = r'({}\.)?{}'.format(variable_pattern, strict_func_pattern)
    class_attribute_pattern = r'cls\.({})'.format(variable_pattern)
    class_method_pattern = r'cls\.{}'.format(strict_func_pattern)
    dict_accessor_pattern = r'({})\[(\'|")([A-Za-z0-9_]+)(\'|")\]'.format(variable_pattern)

    # Variable possible bounders
    arithmetic_operators = ['\+', '-', '/', '\*', '\^']
    surrounders = ['\s', ',', '\)']
    variable_followers = arithmetic_operators + surrounders
    variable_preceders = variable_followers + ['\(']
    preceded_variable_pattern = r'({})({})^\('.format(
        '|'.join([x for x in variable_preceders]), variable_pattern)
    followed_variable_pattern = r'({})({})'.format(
        variable_pattern, '|'.join([x for x in variable_followers]))

    lambda_pattern = r'lambda ([a-zA-Z0-9_,\s]*): (.+)'
    func_pattern = r'([a-z_A-Z]*).([a-zA-Z_][a-z_A-Z0-9]*)\(([^\)]*)\)'

    def __init__(self, verbose=False):
        self.verbose = verbose

    @classmethod
    def getClassAttributes(cls, s):
        ''' Find class attributes in expression. '''
        class_attr_matches = re.findall(cls.class_attribute_pattern, s)
        class_method_matches = re.findall(cls.class_method_pattern, s)
        class_attrs = []
        for candidate in class_attr_matches:
            if candidate not in class_method_matches:
                class_attrs.append(candidate)
        return class_attrs

    @staticmethod
    def removeClassReferences(s):
        return s.replace('self.', '').replace('cls.', '')

    @staticmethod
    def removeLineComments(s):
        return s.split('#', 1)[0].strip()

    @staticmethod
    def removeStartingUnderscores(s):
        return re.sub(r'^[_]+(.+)', lambda x: x.group(1), s)

    @classmethod
    def getLambdaSource(cls, dict_lambda):
        ''' Get the source code of a lambda function. '''
        # Get lambda source code
        lambda_source = inspect.getsource(dict_lambda)
        if lambda_source.count(':') == 2:
            sep_character = ':'
        else:
            sep_character = '='
        lambda_source = lambda_source.split(sep_character, 1)[-1]

        # Clean up source from
        lambda_source = re.sub(' +', ' ', lambda_source.replace('\n', '')).strip().replace('( ', '(')
        lambda_source = cls.removeLineComments(lambda_source)
        if lambda_source[-1] == ',':
            lambda_source = lambda_source[:-1]

        # Match lambda pattern in cleaned-up source, and return match groups
        m = re.match(cls.lambda_pattern, lambda_source)
        if m is None:
            raise ValueError('source does not match lambda pattern: \n {}'.format(
                lambda_source))
        return m.groups()

    @staticmethod
    def getIndent(level):
        ''' Return print indent corresponding to parsing level. '''
        return ''.join(['   '] * level)

    @staticmethod
    def getDocstring(func):
        ''' Get formatted function docstring. '''
        fdoc = inspect.getdoc(func)
        if fdoc is not None:
            fdoc = fdoc.replace('\n', ' ').strip()
        return fdoc

    @staticmethod
    def getFuncSignatureArgs(func):
        return list(inspect.signature(func).parameters.keys())

    @classmethod
    def getFuncCallsOld(cls, s):
        ''' Find function calls in expression. '''
        # TODO: improve function parsing to ensure matching parentheses
        return [m for m in re.finditer(cls.func_pattern, s)]

    @classmethod
    def getFuncCalls(cls, s):
        ''' Return a list of match objects for each function call in expression. '''
        return [m for m in re.finditer(cls.loose_func_pattern, s)]

    @staticmethod
    def getClosure(s, push_char='(', pop_char=')'):
        ''' Get the closure of a given opening character, i.e. all the substring between
            the opening character and its matching closing character. '''
        closure = ''
        balance = 1
        for c in s:
            if c == push_char:
                balance += 1
            elif c == pop_char:
                balance -= 1
            if balance == 0:
                break
            closure += c
        if balance > 0:
            raise ValueError('closure not found')
        return closure

    def parseFuncFields(self, m, expr, level=0):
        ''' Parse a function call with all its relevant fields: name, arguments, and prefix. '''
        fprefix, fname = m.groups()
        fcall = fname
        if fprefix:
            fcall = '{}{}'.format(fprefix, fname)
        else:
            fprefix = ''
        fclosure = self.getClosure(expr[m.end():])
        fclosure = self.translateExpr(fclosure, level=level + 1)
        fcall = f'{fcall}({fclosure})'
        fargs = [x.strip() for x in fclosure.split(',')]
        i = 0
        while i < len(fargs):
            j = fargs[i].find('(')
            if j == -1:
                i += 1
            else:
                try:
                    self.getClosure(fargs[i][j + 1:])
                    i += 1
                except ValueError:
                    fargs[i:i + 2] = [', '.join(fargs[i:i + 2])]

        fargs = list(filter(None, fargs))
        if len(fargs) == 0:
            raise ValueError(f'calling no-arguments function "{fname}"')

        return fcall, fname, fargs, fprefix

    @staticmethod
    def getFuncSource(func):
        ''' Get function source code lines. '''
        func_lines = inspect.getsource(func).split("'''", 2)[-1].splitlines()
        code_lines = []
        for line in func_lines:
            stripped_line = line.strip()
            if len(stripped_line) > 0:
                if not any(stripped_line.startswith(x) for x in ['@', 'def']):
                    code_lines.append(stripped_line)
        return code_lines

    @staticmethod
    def defineConstLambda(const):
        ''' Define a lambda function that returns a constant. '''
        return lambda _: const

    @staticmethod
    def getFuncArgs(m):
        ''' Determine function arguments. '''
        fprefix, fname, fargs = m.groups()
        fcall = '{}({})'.format(fname, fargs)
        if fprefix:
            fcall = '{}.{}'.format(fprefix, fcall)
        fargs = fargs.split(',')
        fargs = [x.strip() for x in fargs]
        return fcall, fname, fargs

    @classmethod
    def removeEnclosingBrackets(cls, s):
        ''' Remove unnecessary enclosing brackets in expression. '''
        if s.startswith('(') and s.endswith(')'):
            closure = cls.getClosure(s[1:])
            if closure == s[1:-1]:
                s = closure
        return s

    def parseLambdaDict(self, lambda_dict, translate_func):
        ''' Parse lambda-dictionary function. '''
        translated_lambda_str_dict = {}

        # For each key and lambda function
        for k, func in lambda_dict.items():
            # Get lambda function source code
            func_args, func_exp = self.getLambdaSource(func)

            # Translate expression
            self.current_key = k
            translated_func_exp = translate_func(func_exp)

            # Remove unnecessary enclosing brackets
            translated_func_exp = self.removeEnclosingBrackets(translated_func_exp)

            # Assign translated expression
            translated_lambda_str_dict[k] = translated_func_exp

        return translated_lambda_str_dict


class PointNeuronTranslator(Translator):
    ''' Generic PointNeuron translator interface. '''

    # Gating patterns
    xinf_pattern = re.compile('^({})inf$'.format(Translator.variable_pattern))
    taux_pattern = re.compile('^tau({})$'.format(Translator.variable_pattern))
    alphax_pattern = re.compile('^alpha({})$'.format(Translator.variable_pattern))
    betax_pattern = re.compile('^beta({})$'.format(Translator.variable_pattern))

    # Neuron-specific regexp patterns
    conductance_pattern = re.compile('(g)([A-Za-z0-9_]*)(Leak|bar)')
    permeability_pattern = re.compile('(p)([A-Za-z0-9_]*)(bar)')
    reversal_potential_pattern = re.compile('(E)([A-Za-z0-9_]+)')
    time_constant_pattern = re.compile('(tau)([A-Za-z0-9_]+)')
    rate_constant_pattern = re.compile('(k)([0-9_]+)')
    ion_concentration_pattern = re.compile('(Ca|Na|K)(i|o)([A-Za-z0-9_]*)')
    current_to_molar_rate_pattern = re.compile('(current_to_molar_rate)([A-Za-z0-9_]+)')
    temperature_pattern = re.compile('^(T)$')

    def __init__(self, pclass, verbose=False):
        super().__init__(verbose=verbose)
        self.pclass = pclass

    def isEffectiveVariable(self, fname, fargs):
        ''' Determine if function is an effective variable. '''

        # Is function sole argument Vm ?
        is_single_arg_Vm_func = len(fargs) == 1 and fargs[0] == 'Vm'

        # Is function a current of a neuron class?
        is_current_func = fname in self.pclass.currents().keys()

        return is_single_arg_Vm_func and not is_current_func


class SonicTranslator(PointNeuronTranslator):
    ''' Translate PointNeuron standard methods into methods adapted for SONIC simulations'''
    lambda_dict_key = 'lambda_dict'
    lambda_dict_accessor_pattern = r'({})\[(\'|")([A-Za-z0-9_]+)(\'|")\]'.format(lambda_dict_key)
    lambda_dict_call_pattern = r'{}\(({})\)'.format(
        lambda_dict_accessor_pattern, Translator.variable_pattern)

    def __init__(self, pclass, verbose=False):
        super().__init__(pclass, verbose=verbose)
        self.eff_rates, self.eff_rates_str = {}, {}
        self.alphax_list, self.betax_list, self.taux_list, self.xinf_list = [], [], [], []

    def parseLambdaDict(self, lambda_dict, translate_func):
        # Translate lambda dict
        translated_dict = super().parseLambdaDict(lambda_dict, translate_func)

        # Correct inter-dependencies and references to Vm
        for k, v in translated_dict.items():
            v = re.sub(self.lambda_dict_call_pattern, lambda x: self.translateLambdaDictcall(x), v)
            v = v.replace('Vm', "lkp['V']")
            translated_dict[k] = v
        return translated_dict

    @classmethod
    def translateLambdaDictcall(cls, x):
        return f'{x.group(1)}[\'{x.group(3)}\'](lkp)'


    def addToEffRates(self, expr):
        ''' add effective rate(s) corresponding to function expression '''

        err_str = 'gating states must be defined via the alphaX-betaX or Xinf-tauX paradigm'

        # If expression matches alpha or beta rate -> return corresponding
        # effective rate function
        for p, l in zip([self.alphax_pattern, self.betax_pattern], [self.alphax_list, self.betax_list]):
            if p.match(expr):
                try:
                    self.eff_rates[expr] = getattr(self.pclass, expr)
                    self.eff_rates_str[expr] = 'self.{}'.format(expr)
                    l.append(expr)
                except AttributeError:
                    raise ValueError(err_str)

        # If expression matches xinf or taux -> add corresponding alpha and beta
        # effective rates functions
        else:
            for p, l in zip([self.taux_pattern, self.xinf_pattern], [self.taux_list, self.xinf_list]):
                m = p.match(expr)
                if m:
                    k = m.group(1)
                    alphax_str, betax_str = ['{}{}'.format(p, k) for p in ['alpha', 'beta']]
                    xinf_str, taux_str = ['{}inf'.format(k), 'tau{}'.format(k)]
                    try:
                        xinf, taux = [getattr(self.pclass, s) for s in [xinf_str, taux_str]]
                        # If taux is a constant, define a lambda function that returns it
                        if not callable(taux):
                            taux = self.defineConstLambda(taux)
                        self.eff_rates.update({
                            alphax_str: lambda Vm: xinf(Vm) / taux(Vm),
                            betax_str: lambda Vm: (1 - xinf(Vm)) / taux(Vm)
                        })
                        self.eff_rates_str.update({
                            alphax_str: 'lambda Vm: cls.{}(Vm) / cls.{}(Vm)'.format(
                                xinf_str, taux_str),
                            betax_str: 'lambda Vm: (1 - cls.{}(Vm)) / cls.{}(Vm)'.format(
                                xinf_str, taux_str)
                        })
                        l.append(expr)
                    except AttributeError:
                        raise ValueError(err_str)

    def createLookupLambda(self, expr):
        ''' Create a lookup lambda function from an expression. '''
        f = eval('lambda cls, lkp, x: {}'.format(expr))
        return lambda *args: f(self.pclass, *args)

    def translateExpr(self, expr, level=0):
        # Get all function calls in the expression
        matches = self.getFuncCalls(expr)
        f_list = [self.parseFuncFields(m, expr, level=level + 1) for m in matches]

        # For each function call
        for (fcall, fname, fargs, fprefix) in f_list:

            # If effective variable -> replace by dict lookup
            if self.isEffectiveVariable(fname, fargs):
                self.addToEffRates(fname)
                new_fcall = "lkp['{}']".format(fname)
                expr = expr.replace(fcall, new_fcall)

        # Return modified expression
        return expr

    def parseDerStates(self):
        ''' Parse neuron's derStates method to construct adapted derEffStates and effRates
            methods used for SONIC simulations. '''

        # Get dictionary of translated lambda functions expressions for derivative states
        eff_dstates_str = self.parseLambdaDict(self.pclass.derStates(), self.translateExpr)
        if self.verbose:
            print('---------- derEffStates ----------')
            for k, v in eff_dstates_str.items():
                print("    {} : lambda lkp, x: {}".format(k, v))
            print('')
            print('---------- effRates ----------')
            for k, v in self.eff_rates_str.items():
                print("    {} : {}".format(k, v))
            print('')

        # Return dictionary of evaluated functions
        return {k: self.createLookupLambda(v) for k, v in eff_dstates_str.items()}

    def parseSteadyStates(self):
        ''' Parse neuron's steadyStates method to construct an adapted quasiSteadyStates
            method used for SONIC QSS simulations. '''

        # Get dictionary of translated lambda functions expressions steady states
        qsstates_str = self.parseLambdaDict(self.pclass.steadyStates(), self.translateExpr)
        if self.verbose:
            print('---------- quasiSteadyStates ----------')
            for k, v in qsstates_str.items():
                print("    {} : lambda lkp: {}".format(k, v))
            print('')
        # Return dictionary of evaluated functions
        return {k: self.createLookupLambda(v) for k, v in qsstates_str.items()}
