#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-06 13:36:00
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-06 16:00:03

import inspect
import sys

from .simulators import PWSimulator, PeriodicSimulator
from .batches import Batch, createQueue
from .model import Model
from .pneuron import PointNeuron
from .bls import BilayerSonophore, PmCompMethod, LennardJones
from .nbls import NeuronalBilayerSonophore
from .nmodl_generator import NmodlGenerator

from ..neurons import getPointNeuron


def getModelsDict():
    ''' Construct a dictionary of all model classes, indexed by simulation key. '''
    current_module = sys.modules[__name__]
    models_dict = {}
    for _, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and hasattr(obj, 'simkey') and isinstance(obj.simkey, str):
            models_dict[obj.simkey] = obj
    return models_dict


def getModel(key, meta):
    ''' Return appropriate model object based on a sim key and a dictionary of meta-information. '''
    if key == 'MECH':
        model = BilayerSonophore(meta['a'], meta['Cm0'], meta['Qm0'])
    else:
        model = getPointNeuron(meta['neuron'])
        if key == 'ASTIM':
            model = NeuronalBilayerSonophore(meta['a'], model, meta['Fdrive'])
    return model
