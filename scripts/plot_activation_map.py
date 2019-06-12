# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-26 09:51:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-12 23:08:21

''' Plot (duty-cycle x amplitude) US activation map of a neuron at a given frequency and PRF. '''

import numpy as np
import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.utils import logger, selectDirDialog, Intensity2Pressure
from PySONIC.plt import ActivationMap
from PySONIC.neurons import getPointNeuron


# Default parameters
defaults = dict(
    neuron='RS',
    radius=32,  # nm
    freq=500,  # kHz
    duration=1000,  # ms
    PRF=100,  # Hz
    amps=np.logspace(np.log10(10), np.log10(600), num=30),  # kPa
    DCs=np.arange(1, 101),  # %
    Ascale='log',
    FRscale='log',
    FRbounds=(1e0, 1e3),  # Hz
    tmax=240,  # ms
    Vbounds=(-150, 50),  # mV
)


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-i', '--inputdir', type=str, default=None, help='Input directory')

    ap.add_argument('-r', '--threshold', default=False, action='store_true',
                    help='Show threshold amplitudes')

    ap.add_argument('--interactive', default=False, action='store_true',
                    help='Show traces on click')

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-a', '--radius', type=float, default=defaults['radius'],
                    help='Sonophore radius (nm)')
    ap.add_argument('-f', '--freq', type=float, default=defaults['freq'],
                    help='US frequency (kHz)')
    ap.add_argument('-d', '--duration', type=float, default=defaults['duration'],
                    help='Stimulus duration (ms)')
    ap.add_argument('-A', '--amps', nargs='+', type=float, help='Acoustic pressure amplitude (kPa)')
    ap.add_argument('-I', '--intensities', nargs='+', type=float, help='Acoustic intensity (W/cm2)')
    ap.add_argument('--PRF', type=float, default=defaults['PRF'], help='PRF (Hz)')
    ap.add_argument('--DC', nargs='+', type=float, help='Duty cycle (%%)')

    # Plot options
    ap.add_argument('--Ascale', type=str, default=defaults['Ascale'],
                    help='y-axis scale ("log" or "lin")')
    ap.add_argument('--FRscale', type=str, default=defaults['FRscale'],
                    help='map color scale ("log" or "lin")')
    ap.add_argument('--FRbounds', type=float, nargs='+', default=defaults['FRbounds'],
                    help='Lower and upper bounds for firing rate (Hz)')
    ap.add_argument('--tmax', type=float, default=defaults['tmax'],
                    help='Max time value for callback graphs (ms)')
    ap.add_argument('--Vbounds', type=float, nargs='+', default=defaults['Vbounds'],
                    help='Y-axis extent for callback graphs (mV)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}

    # Runtime options
    loglevel = logging.DEBUG if args['verbose'] is True else logging.INFO
    logger.setLevel(loglevel)
    inputdir = args['inputdir'] if 'inputdir' in args else selectDirDialog()
    if inputdir == '':
        logger.error('Operation cancelled')
        return

    # Parameters
    pneuron = getPointNeuron(args['neuron'])
    a = args['radius'] * 1e-9  # m
    Fdrive = args['freq'] * 1e3  # Hz
    tstim = args['duration'] * 1e-3  # s
    PRF = args['PRF']  # Hz
    DCs = np.array(args.get('DCs', defaults['DCs'])) * 1e-2  # (-)
    if 'amps' in args:
        amps = np.array(args['amps']) * 1e3  # Pa
    elif 'intensities' in args:
        amps = Intensity2Pressure(np.array(args['intensities']) * 1e4)  # Pa
    else:
        amps = np.array(defaults['amps']) * 1e3  # Pa

    # Plot options
    for item in ['Ascale', 'FRscale']:
        assert args[item] in ('lin', 'log'), 'Unknown {}'.format(item)

    # Plot activation map
    actmap = ActivationMap(inputdir, pneuron, a, Fdrive, tstim, PRF, amps, DCs)
    actmap.render(
        Ascale=args['Ascale'],
        FRscale=args['FRscale'],
        FRbounds=args['FRbounds'],
        interactive=args['interactive'],
        Vbounds=args['Vbounds'],
        tmax=args['tmax'],
        thresholds=args['threshold']
    )

    plt.show()


if __name__ == '__main__':
    main()
