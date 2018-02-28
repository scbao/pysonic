#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-02-27 15:14:54

""" Compare profiles of several specific output variables of NICE simulations. """

import sys
import logging

from PointNICE.utils import logger, OpenFilesDialog
from PointNICE.plt import plotComp

# Set logging level
logger.setLevel(logging.INFO)

# Select data files
pkl_filepaths, _ = OpenFilesDialog('pkl')
if not pkl_filepaths:
    logger.error('No input file')
    sys.exit(1)


# Comparative plot
try:
    yvars = ['Qm']
    # labels = ['classic', 'effective']
    # labels = ['FS neuron', 'LTS neuron', 'RE neuron', 'RS neuron', 'TC neuron']
    plotComp(yvars, pkl_filepaths)
except AssertionError as err:
    logger.error(err)
    sys.exit(1)
