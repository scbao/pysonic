# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-03 20:10:53

''' Phase-plane analysis of neuron behavior under quasi-steady state approximation. '''

import os
import matplotlib.pyplot as plt

from PySONIC.utils import logger
from PySONIC.plt import plotQSSvars, plotQSSVarVsAmp, plotEqChargeVsAmp
from PySONIC.parsers import AStimParser


def main():

    # Parse command line arguments
    parser = AStimParser()
    parser.addCmap(default='viridis')
    parser.addSave()
    parser.addInputDir()
    parser.addOutputDir()
    parser.add_argument(
        '--comp', default=False, action='store_true', help='Compare with simulations')
    args = parser.parse()
    logger.setLevel(args['loglevel'])
    args['inputdir'] = parser.parseInputDir(args) if args['comp'] else None
    args['outputdir'] = parser.parseOutputDir(args) if args['save'] else None
    if args['plot'] is None:
        args['plot'] = ['dQdt']

    a, Fdrive, tstim, toffset, PRF = [
        args[k][0] for k in ['radius', 'freq', 'tstim', 'toffset', 'PRF']]

    figs = []
    for i, neuron in enumerate(args['neuron']):
        for DC in args['DC']:
            if args['amp'].size == 1:
                figs.append(
                    plotQSSvars(neuron, a, Fdrive, args['amp'][0]))
            else:
                # Plot evolution of QSS vars vs Q for different amplitudes
                for pvar in args['plot']:
                    figs.append(plotQSSVarVsAmp(
                        neuron, a, Fdrive, pvar, amps=args['amp'], DC=DC,
                        cmap=args['cmap'], zscale=args['Ascale']))

                # Plot equilibrium charge as a function of amplitude
                if 'dQdt' in args['plot']:
                    figs.append(plotEqChargeVsAmp(
                        neuron, a, Fdrive, amps=args['amp'], tstim=tstim, toffset=toffset, PRF=PRF,
                        DC=DC, xscale=args['Ascale'], compdir=args['inputdir'], mpi=args['mpi'],
                        loglevel=args['loglevel']))

    if args['save']:
        for fig in figs:
            s = fig.canvas.get_window_title()
            s = s.replace('(', '- ').replace('/', '_').replace(')', '')
            figname = '{}.png'.format(s)
            fig.savefig(os.path.join(args['outputdir'], figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
