#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-03-20 12:19:55
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-10 17:04:50

""" Batch plot profiles of several specific output variables of NICE simulations. """

import sys
import logging

from PointNICE.utils import OpenFilesDialog
from PointNICE.plt import plotBatch

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.INFO)

# Select data files
pkl_filepaths, pkl_dir = OpenFilesDialog('pkl')
if not pkl_filepaths:
    logger.error('No input file')
    sys.exit(1)

# Plot profiles
yvars = {'Q_m': ['Qm'], 'i_{Ca}\ kin.': ['s', 'u', 's2u'], 'I': ['iNa', 'iK', 'iT', 'iL']}
plotBatch(pkl_dir, pkl_filepaths, yvars)
