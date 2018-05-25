# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-09-01 21:08:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-05-25 15:57:01

from PointNICE.plt import plotRateConstants
from PointNICE.neurons import *


neuron = CorticalRS()
plotRateConstants(neuron)
