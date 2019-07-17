# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-06 13:36:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-17 15:29:10

from types import MethodType
import inspect
import sys

from ..core.translators import SonicTranslator
from .template import *
from .cortical import *
from .thalamic import *
from .leech import *
from .stn import *
from .fh import *


def getNeuronsDict():
    ''' Construct a dictionary of all the implemented point neuron classes, indexed by name. '''
    current_module = sys.modules[__name__]
    neurons_dict = {}
    for _, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and hasattr(obj, 'name') and isinstance(obj.name, str):
            neurons_dict[obj.name] = obj
    return neurons_dict


def getPointNeuron(name):
    ''' Return a point-neuron instance corresponding to a given name. '''
    neuron_classes = getNeuronsDict()
    try:
        return neuron_classes[name]()
    except KeyError:
        raise ValueError('"{}" neuron not found. Implemented neurons are: {}'.format(
            name, ', '.join(list(neuron_classes.keys()))))


def createClassMethod(func):
    return lambda self: func


for pname, pclass in getNeuronsDict().items():
    translator = SonicTranslator(pclass, verbose=False)
    eff_dstates = translator.parseDerStates()
    pclass.derEffStates = MethodType(createClassMethod(eff_dstates), pclass)
    pclass.effRates = MethodType(createClassMethod(translator.eff_rates), pclass)
    pclass.rates = list(translator.eff_rates.keys())
    pclass.alphax_list = set(translator.alphax_list)
    pclass.betax_list = set(translator.betax_list)
    pclass.taux_list = set(translator.taux_list)
    pclass.xinf_list = set(translator.xinf_list)
    qsstates = translator.parseSteadyStates()
    pclass.quasiSteadyStates = MethodType(createClassMethod(qsstates), pclass)
