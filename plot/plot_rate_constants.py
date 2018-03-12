# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-09-01 21:08:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-12 19:45:10

from PointNICE.plt import plotRateConstants
from PointNICE.neurons import *


neurons = [LeechPressure()]
for neuron in neurons:
    plotRateConstants(neuron)
