#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-15 15:59:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-12 19:44:39

''' Plot the profiles of the charge-dependent "effective" HH rates,
    as a function of charge density. '''

from PointNICE.plt import plotEffCoeffs
from PointNICE.neurons import *

neuron = ThalamoCortical()
Fdrive = 350e3
plotEffCoeffs(neuron, Fdrive)
