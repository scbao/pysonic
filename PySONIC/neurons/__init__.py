#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-06 13:36:00
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-06 21:23:54

import inspect
import sys

from .lookups import *
from .cortical import CorticalRS, CorticalFS, CorticalLTS, CorticalIB
from .thalamic import ThalamicRE, ThalamoCortical
from .leech import LeechTouch, LeechPressure, LeechRetzius
from .stn import OtsukaSTN
from .fh import FrankenhaeuserHuxley


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
