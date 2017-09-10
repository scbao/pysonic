#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-10 18:56:42

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
    labels = ['normal last pulse', 'delayed -1ms', 'delayed +1ms']
    plotComp(yvars, pkl_filepaths[::-1])
except AssertionError as err:
    logger.error(err)
    sys.exit(1)
