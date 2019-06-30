# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-29 11:26:27
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-30 13:01:32

from time import gmtime, strftime
import re
import inspect
import pprint

from ..constants import *


class Translator:
    ''' Translate PointNeuron standard methods into methods
    adapted for SONIC simulations'''

    lambda_pattern = 'lambda ([a-z_A-Z,0-9\s]*): (.+)'
    func_pattern = '([a-z_A-Z]*).([a-z_A-Z][a-z_A-Z0-9]*)\(([^\)]*)\)'
    class_attribute_pattern = '(cls).([a-z_A-Z_][a-z_A-Z0-9_]*)'

    def __init__(self, verbose=False):
        self.verbose = verbose

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

        # Clean up source from line break, extra spaces and final comma
        lambda_source = re.sub(' +', ' ', lambda_source.replace('\n', ' ')).strip()
        if lambda_source[-1] == ',':
            lambda_source = lambda_source[:-1]

        # Match lambda pattern in cleaned-up source, and return match groups
        m = re.match(cls.lambda_pattern, lambda_source)
        if m is None:
            raise ValueError('source does not match lambda pattern: \n {}'.format(
                lambda_source))
        return m.groups()

    @classmethod
    def getClassAttributeCalls(cls, s):
        ''' Find attribute calls in expression. '''
        return re.finditer(cls.class_attribute_pattern, s)

    @classmethod
    def getFuncCalls(cls, s):
        ''' Find function calls in expression. '''
        return re.finditer(cls.func_pattern, s)

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
        return fcall, fname, fargs

    def parseLambdaDict(self, lambda_dict, translate_func):
        ''' Parse pclassect function. '''
        translated_lambda_str_dict = {}

        # For each key and lambda function
        for k, dfunc in lambda_dict.items():
            # Get lambda function source code
            dfunc_args, dfunc_exp = self.getLambdaSource(dfunc)

            # Assign translated expression
            translated_lambda_str_dict[k] = translate_func(dfunc_exp)

        return translated_lambda_str_dict


class PointNeuronTranslator(Translator):

    suffix_pattern = '[A-Za-z0-9_]+'
    # Define patterns
    xinf_pattern = re.compile('^({})inf$'.format(suffix_pattern))
    taux_pattern = re.compile('^tau({})$'.format(suffix_pattern))
    alphax_pattern = re.compile('^alpha({})$'.format(suffix_pattern))
    betax_pattern = re.compile('^beta({})$'.format(suffix_pattern))

    def __init__(self, pclass, verbose=False):
        super().__init__(verbose=verbose)
        self.pclass = pclass


class SonicTranslator(PointNeuronTranslator):

    def __init__(self, pclass, verbose=False):
        super().__init__(pclass, verbose=verbose)
        self.eff_rates, self.eff_rates_str = {}, {}

    def addToEffRates(self, expr):
        ''' add effective rate(s) corresponding to function expression '''

        err_str = 'gating states must be defined via the alphaX-betaX or Xinf-tauX paradigm'

        # If expression matches alpha or beta rate -> return corresponding
        # effective rate function
        for pattern in [self.alphax_pattern, self.betax_pattern]:
            if pattern.match(expr):
                try:
                    self.eff_rates[expr] = getattr(self.pclass, expr)
                    self.eff_rates_str[expr] = 'self.{}'.format(expr)
                except AttributeError:
                    raise ValueError(err_str)

        # If expression matches xinf or taux -> add corresponding alpha and beta
        # effective rates functions
        else:
            for pattern in [self.taux_pattern, self.xinf_pattern]:
                m = pattern.match(expr)
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
                    except AttributeError:
                        raise ValueError(err_str)

    def createDerEffStateLambda(self, expr):
        ''' Create a lambda function that computes an effective state derivative '''
        f = eval('lambda cls, lkp, x: {}'.format(expr))
        return lambda *args: f(self.pclass, *args)

    def translateExpr(self, expr):
        # For each internal function call in the expression
        matches = self.getFuncCalls(expr)
        for m in matches:
            # Determine function arguments
            fcall, fname, fargs = self.getFuncArgs(m)

            # Translate function call and replace it in the expression
            if len(fargs) == 1 and fargs[0] == 'Vm':
                # If sole argument is Vm replace function call by lookup retrieval
                self.addToEffRates(fname)
                new_fcall = "lkp['{}']".format(fname)
            else:
                # Otherwise, do not replace anything
                new_fcall = fcall

            expr = expr.replace(fcall, new_fcall)

        return expr

    def parseDerStates(self):
        ''' Parse neuron's derStates method to construct adapted derEffStates and effRates
            methods used for SONIC simulations. '''

        # Get dictionary of translated lambda functions expressions for derivative states
        eff_dstates_str = self.parseLambdaDict(self.pclass.derStates(), self.translateExpr)
        eff_dstates_str = {k: v.replace('Vm', "lkp['V']") for k, v in eff_dstates_str.items()}
        if self.verbose:
            print('---------- derEffStates ----------')
            pprint.PrettyPrinter(indent=4).pprint({
                k: 'lambda lkp, x: {}'.format(v) for k, v in eff_dstates_str.items()})
            print('---------- effRates ----------')
            pprint.PrettyPrinter(indent=4).pprint(self.eff_rates_str)

        # Return dictionary of evaluated functions
        return {k: self.createDerEffStateLambda(v) for k, v in eff_dstates_str.items()}
