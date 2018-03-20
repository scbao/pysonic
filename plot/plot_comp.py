#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-20 18:49:20

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
colors = rescaleColorset(cl.to_numeric(cl.scales['12']['qual']['Paired']))
# labels = sum([['', x] for x in ['FS','LTS', 'RE', 'RS', 'TC']], [])
labels = sum([['', x] for x in ['sub-threshold', 'threshold', 'supra-threshold']], [])
# labels = sum([['', x] for x in ['100 Hz', '1 kHz', '10 kHz']], [])
# labels = ['classic', 'effective']
# labels = ['20 kHz', '500 kHz', '2 MHz']
lines = ['--', '-'] * (nfiles // 2)
# patches = [False, True, False, True, False, False]  # * (nfiles // 2)

# colors = rescaleColorset(cl.to_numeric(cl.scales['9']['qual']['Set1']))
# labels = ['100 Hz', '1kHz', '10kHz']
# lines = ['--'] * nfiles
# patches = 'all'


try:
    plotComp(yvars, pkl_filepaths, labels=labels, lines=lines, colors=colors, patches='one')
except InputError as err:
    logger.error(err)
    sys.exit(1)
