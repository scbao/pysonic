#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-24 23:52:32

""" Compare profiles of several specific output variables of NICE simulations. """

import logging

from PySONIC.utils import logger, OpenFilesDialog
from PySONIC.plt import plotComp

# Set logging level
logger.setLevel(logging.INFO)


default = 'Qm'


def main():

    # Select data files
    pkl_filepaths, _ = OpenFilesDialog('pkl')
    if not pkl_filepaths:
        logger.error('No input file')
        return

    # Comparative plot
    try:
        plotComp(default, pkl_filepaths)
    except Exception as err:
        logger.error(err)


if __name__ == '__main__':
    main()
