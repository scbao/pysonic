#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-25 17:09:22

""" Compare profiles of several specific output variables of NICE simulations. """

from PointNICE.utils import OpenFilesDialog
from PointNICE.plt import plotComp

# List of variables to plot
yvars = ['Pac', 'Pmavg', 'Telastic', 'Vm', 'iL', 'iNet']
vars_mech = ['Pac', 'Z', 'ng']

# Select data files
pkl_filepaths, _ = OpenFilesDialog('pkl')
if not pkl_filepaths:
    print('error: no input file')
    quit()

# Comparative plot
plotComp(vars_mech, pkl_filepaths)
