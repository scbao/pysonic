#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-03-20 12:19:55
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-23 15:41:25

""" Batch plot profiles of several specific output variables of NICE simulations. """

from PointNICE.utils import OpenFilesDialog
from PointNICE.plt import plotBatch

# List of variables to plot and positions
tag = 'test'
yvars = ['Vm', 'VL']
positions = [0, 0]

# Select data files
pkl_filepaths, pkl_dir = OpenFilesDialog('pkl')
if not pkl_filepaths:
    print('error: no input file')
    quit()


plotBatch(yvars, positions, pkl_dir, pkl_filepaths, tag=tag)
