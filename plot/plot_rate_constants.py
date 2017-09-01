# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-09-01 21:08:24
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-01 21:29:58

from PointNICE.plt import plotRateConstants
from PointNICE.channels import *


neurons = [CorticalRS()]
for neuron in neurons:
    plotRateConstants(neuron)
