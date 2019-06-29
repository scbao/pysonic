# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-06 13:36:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-29 17:38:48

import inspect
import sys

from .simulators import *
from .batches import *
from .model import *
from .pneuron import *
from .bls import *
from .translators import *
from .nbls import *
from .lookups import *

from ..neurons import getPointNeuron


def getModelsDict():
    ''' Construct a dictionary of all model classes, indexed by simulation key. '''
    current_module = sys.modules[__name__]
    models_dict = {}
    for _, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and hasattr(obj, 'simkey') and isinstance(obj.simkey, str):
            models_dict[obj.simkey] = obj
    return models_dict


def getModel(meta):
    ''' Return appropriate model object based on a dictionary of meta-information. '''
    if meta['simkey'] == 'MECH':
        model = BilayerSonophore(meta['a'], meta['Cm0'], meta['Qm0'])
    else:
        model = getPointNeuron(meta['neuron'])
        if meta['simkey'] == 'ASTIM':
            model = NeuronalBilayerSonophore(meta['a'], model, meta['Fdrive'])
    return model
