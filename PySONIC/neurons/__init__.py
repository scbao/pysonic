#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-06 13:36:00
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-22 16:49:12

import inspect
import sys

from .cortical import CorticalRS, CorticalFS, CorticalLTS, CorticalIB
from .thalamic import ThalamicRE, ThalamoCortical
from .leech import LeechTouch, LeechPressure, LeechRetzius


def getNeuronsDict():
    current_module = sys.modules[__name__]
    neurons_dict = {}
    for _, obj in inspect.getmembers(current_module):
        if inspect.isclass(obj) and isinstance(obj.name, str):
            neurons_dict[obj.name] = obj
    return neurons_dict
