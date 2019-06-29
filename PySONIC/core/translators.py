# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-29 11:26:27
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-29 19:56:19

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

    def __init__(self, verbose=False):
        self.verbose = verbose

    @classmethod
    def getLambdaSource(cls, dict_lambda):
        ''' Get the source code of a lambda function. '''
        # Get lambda source code
        lambda_source = inspect.getsource(dict_lambda).split(':', 1)[-1]

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
    def getFuncCalls(cls, s):
        ''' Find function calls in expression. '''
        return re.finditer(cls.func_pattern, s)

    @staticmethod
    def defineConstLambda(const):
        ''' Define a lambda function that returns a constant. '''
        return lambda _: const

    def getFuncArgs(self, m):
        ''' Determine function arguments. '''
        fprefix, fname, fargs = m.groups()
        fcall = '{}({})'.format(fname, fargs)
        if fprefix:
            fcall = '{}.{}'.format(fprefix, fcall)
        fargs = fargs.split(',')
        return fcall, fname, fargs

    def parseLambdaDict(self, lambda_dict, func_call_replace):
        ''' Parse pclassect function. '''

        translated_lambda_str_dict = {}

        # For each key and lambda function
        for k, dfunc in lambda_dict.items():
            # Get lambda function source code
            dfunc_args, dfunc_exp = self.getLambdaSource(dfunc)

            # For each internal function call in the lambda expression
            matches = self.getFuncCalls(dfunc_exp)
            for m in matches:
                # Determine function arguments
                fcall, fname, fargs = self.getFuncArgs(m)

                # Translate function call and replace it in lambda expression
                new_fcall = func_call_replace(fcall, fname, fargs)
                dfunc_exp = dfunc_exp.replace(fcall, new_fcall)

            # Assign translated expression
            translated_lambda_str_dict[k] = dfunc_exp

        return translated_lambda_str_dict


class PointNeuronTranslator(Translator):

    suffix_pattern = '[A-Za-z0-9_]+'

    def __init__(self, pclass, verbose=False):
        super().__init__(verbose=verbose)
        self.pclass = pclass

        # Define patterns
        self.xinf_pattern = re.compile('^({})inf$'.format(self.suffix_pattern))
        self.taux_pattern = re.compile('^tau({})$'.format(self.suffix_pattern))
        self.alphax_pattern = re.compile('^alpha({})$'.format(self.suffix_pattern))
        self.betax_pattern = re.compile('^beta({})$'.format(self.suffix_pattern))

        # Initialize effective rates dictionaries
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

    def translateVmFunc(self, fcall, fname, fargs):
        if len(fargs) == 1 and fargs[0] == 'Vm':
            # If sole argument is Vm replace function call by lookup retrieval
            self.addToEffRates(fname)
            return "lkp['{}']".format(fname)
        else:
            # Otherwise, do not replace anything
            return fcall

    def parseDerStates(self):
        ''' Parse neuron's derStates method to construct adapted derEffStates and effRates
            methods used for SONIC simulations. '''

        # Get dictionary of translated lambda functions expressions for derivative states
        func = self.pclass.derStates
        d = func()
        eff_dstates_str = self.parseLambdaDict(d, self.translateVmFunc)
        eff_dstates_str = {k: v.replace('Vm', "lkp['V']") for k, v in eff_dstates_str.items()}
        if self.verbose:
            print('---------- derEffStates ----------')
            pprint.PrettyPrinter(indent=4).pprint({
                k: 'lambda lkp, x: {}'.format(v) for k, v in eff_dstates_str.items()})
            print('---------- effRates ----------')
            pprint.PrettyPrinter(indent=4).pprint(self.eff_rates_str)

        # Return dictionary of evaluated functions
        return {k: self.createDerEffStateLambda(v) for k, v in eff_dstates_str.items()}


class NmodlTranslator(Translator):

    tabreturn = '\n   '
    NEURON_protected_vars = ['O', 'C']

    def __init__(self, pclass):
        self.pclass = pclass
        super().__init__(verbose=True)

    def parseCurrents(self):
        ''' Parse neuron's currents method to construct adapted BREAKPOINT block
            in MOD files. '''

        currents_str = []

        # For each current
        for k, dfunc in self.pclass.currents().items():
            # Get current function source code
            cfunc_args, cfunc_exp = self.getLambdaSource(dfunc)

            # For each internal function call in the derivative function expression
            matches = self.getFuncCalls(dfunc_exp)
            for m in matches:
                # Determine function arguments
                fcall, fname, fargs = self.getFuncArgs(m)

                # If sole argument is Vm
                if len(fargs) == 1 and fargs[0] == 'Vm':
                    # Replace function call by lookup retrieval in expression
                    dfunc_exp = dfunc_exp.replace(fcall, "lkp['{}']".format(fname))

                    # Add the corresponding effective rate function(s) to the dictionnary
                    d, dstr = self.createEffRates(fname)
                    eff_rates.update(d)
                    eff_rates_str.update(dstr)

            # Replace Vm by lkp['V'] in expression
            dfunc_exp = dfunc_exp.replace('Vm', "lkp['V']")

            # Create the modified lambda expression and evaluate it
            eff_dstates[k], eff_dstates_str[k] = self.createDerEffStateLambda(dfunc_exp)

        if self.verbose:
            print('---------- derEffStates ----------')
            pprint.PrettyPrinter(indent=4).pprint(eff_dstates_str)
            print('---------- effRates ----------')
            pprint.PrettyPrinter(indent=4).pprint(eff_rates_str)

        # Define methods that return dictionaries of effective states derivatives
        # and effective rates functions
        def derEffStates():
            return eff_dstates

        def effRates():
            return eff_rates

        return derEffStates, effRates
