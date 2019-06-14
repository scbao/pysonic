# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-13 12:41:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-14 10:54:23

''' Plot temporal profiles of specific simulation output variables. '''

import matplotlib.pyplot as plt

from PySONIC.utils import logger
from PySONIC.plt import ComparativePlot, SchemePlot
from PySONIC.parsers import Parser


def main():
    # Parse command line arguments
    parser = Parser()
    parser.addHideOutput()
    parser.addInputFiles()
    parser.addOutputDir(dep_key='save')
    parser.addCompare()
    parser.addSave()
    parser.addFigureExtension()
    parser.addSamplingRate()
    parser.addMarkSpikes()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Plot appropriate graph
    if args['compare']:
        if args['plot'] == ['all'] or args['plot'] is None:
            logger.error('Specific variables must be specified for comparative plots')
            return
        for pltvar in args['plot']:
            try:
                comp_plot = ComparativePlot(args['inputfiles'], pltvar)
                comp_plot.render()
            except KeyError as e:
                logger.error(e)
                return
    else:
        scheme_plot = SchemePlot(args['inputfiles'], pltscheme=args['pltscheme'])
        scheme_plot.render(
            title=True,
            save=args['save'],
            directory=args['outputdir'],
            fig_ext=args['figext'],
            mark_spikes=args['markspikes']
        )
    if not args['hide']:
        plt.show()


if __name__ == '__main__':
    main()
