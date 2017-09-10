#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-10 17:04:55

""" Compare profiles of several specific output variables of NICE simulations. """

import sys
import logging

from PointNICE.utils import OpenFilesDialog
from PointNICE.plt import plotComp

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.INFO)


# Select data files
pkl_filepaths, _ = OpenFilesDialog('pkl')
if not pkl_filepaths:
    logger.error('No input file')
    sys.exit(1)


# Comparative plot
yvars = ['Qm']
plotComp(yvars, pkl_filepaths[::-1], labels=['normal last pulse', 'delayed -1ms', 'delayed +1ms'])
