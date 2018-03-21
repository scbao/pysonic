#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-21 17:39:11

""" Compare profiles of several specific output variables of NICE simulations. """

import sys
import logging
import colorlover as cl

from PointNICE.utils import logger, OpenFilesDialog, InputError
from PointNICE.plt import plotComp, rescaleColorset

# Set logging level
logger.setLevel(logging.INFO)

# Select data files
pkl_filepaths, _ = OpenFilesDialog('pkl')
if not pkl_filepaths:
    logger.error('No input file')
    sys.exit(1)
nfiles = len(pkl_filepaths)

# Comparative plot
yvars = ['Qm']

try:
    plotComp(yvars, pkl_filepaths)
except InputError as err:
    logger.error(err)
    sys.exit(1)
