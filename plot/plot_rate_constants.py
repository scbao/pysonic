# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-09-01 21:08:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-21 16:07:32

from PySONIC.plt import plotRateConstants
from PySONIC.neurons import *


neuron = CorticalRS()
plotRateConstants(neuron)
