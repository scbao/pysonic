#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-03-20 12:19:55
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-24 23:52:19

""" Batch plot profiles of several specific output variables of NICE simulations. """

import logging

from PySONIC.utils import logger, OpenFilesDialog
from PySONIC.plt import plotBatch

# Set logging level
logger.setLevel(logging.INFO)

defaults = dict(
    V_m=['Vm'],
    Q_m=['Qm']
)


def main():

    # Select data files
    pkl_filepaths, pkl_dir = OpenFilesDialog('pkl')
    if not pkl_filepaths:
        logger.error('No input file')
        return

    # Plot profiles
    try:
        plotBatch(pkl_dir, pkl_filepaths, title=True, vars_dict=defaults)
    except Exception as err:
        logger.error(err)


if __name__ == '__main__':
    main()
