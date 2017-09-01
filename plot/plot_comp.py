#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-01 16:58:41

""" Compare profiles of several specific output variables of NICE simulations. """

from PointNICE.utils import OpenFilesDialog
from PointNICE.plt import plotComp


# Select data files
pkl_filepaths, _ = OpenFilesDialog('pkl')
if not pkl_filepaths:
    print('error: no input file')
    quit()


# Comparative plot
yvars = ['Qm']
plotComp(yvars, pkl_filepaths[::-1], labels=['normal last pulse', 'delayed -1ms', 'delayed +1ms'])
