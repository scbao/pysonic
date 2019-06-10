#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-10 19:56:23

''' Plot temporal profiles of specific simulation output variables. '''

import logging
import matplotlib.pyplot as plt

from PySONIC.utils import logger, OpenFilesDialog, selectDirDialog
from PySONIC.plt import ComparativePlot, SchemePlot
from PySONIC.parsers import Parser

# Set logging level
logger.setLevel(logging.INFO)


def main():
    # Parse command line arguments
    parser = Parser()
    parser.addHideOutput()
    parser.addOutputDir(dep_key='save')
    parser.addCompare()
    parser.addSave()
    parser.addSamplingRate()
    args = parser.parse()

    # Select data files
    pkl_filepaths, _ = OpenFilesDialog('pkl')
    if not pkl_filepaths:
        logger.error('No input file')
        return

    # Plot appropriate graph
    if args['compare']:
        if args['plot'] == ['all'] or args['plot'] is None:
            logger.error('Specific variables must be specified for comparative plots')
            quit()
        for pltvar in args['plot']:
            try:
                comp_plot = ComparativePlot(pkl_filepaths, pltvar)
                comp_plot.render()
            except KeyError as e:
                logger.error(e)
                quit()
    else:
        scheme_plot = SchemePlot(pkl_filepaths, pltscheme=parser.parsePltScheme(args))
        scheme_plot.render(
            title=True,
            save=args['save'],
            ask_before_save=not args['save'],
            directory=args['outputdir']
        )
    if not args['hide']:
        plt.show()


if __name__ == '__main__':
    main()
