# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-02-13 12:41:26
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-17 19:59:21

''' Plot phase plane diagram of specific simulation output variables. '''

import matplotlib.pyplot as plt

from PySONIC.utils import logger
from PySONIC.plt import PhaseDiagram
from PySONIC.parsers import PlotParser


def main():
    parser = PlotParser()
    parser.addRelativeTimeBounds()
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Plot phase-plane diagram
    phase_diag = PhaseDiagram(args['inputfiles'], args['plot'][0])
    phase_diag.render(
        trange=args['trange'],
        rel_tbounds=args['rel_tbounds'],
        labels=args['labels'],
        prettify=args['pretty'],
        cmap=args['cmap'],
        cscale=args['cscale']
    )

    if not args['hide']:
        plt.show()


if __name__ == '__main__':
    main()
