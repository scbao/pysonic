
import logging
import numpy as np
from argparse import ArgumentParser

from .utils import Intensity2Pressure, selectDirDialog
from .neurons import getNeuronsDict, CorticalRS


class GenericParser(ArgumentParser):
    ''' Generic parser interface. '''

    dist_str = '[scale min max n]'

    def __init__(self):
        super().__init__()
        self.defaults = {}
        self.allowed = {}
        self.factors = {}
        self.addPlot()
        self.addMPI()
        self.addVerbose()
        self.addOutputDir()

    def getDistribution(self, xmin, xmax, nx, scale='lin'):
        if scale == 'log':
            xmin, xmax = np.log10(xmin), np.log10(xmax)
        return {'lin': np.linspace, 'log': np.logspace}[scale](xmin, xmax, nx)

    def getDistFromList(self, xlist):
        if not isinstance(xlist, list):
            raise TypeError('Input must be a list')
        if len(xlist) != 4:
            raise ValueError('List must contain exactly 4 arguments ([type, min, max, n])')
        scale = xlist[0]
        if scale not in ('log', 'lin'):
            raise ValueError('Unknown distribution type (must be "lin" or "log")')
        xmin, xmax = [float(x) for x in xlist[1:-1]]
        if xmin >= xmax:
            raise ValueError('Specified minimum higher or equal than specified maximum')
        nx = int(xlist[-1])
        if nx < 2:
            raise ValueError('Specified number must be at least 2')
        return self.getDistribution(xmin, xmax, nx, scale=scale)

    def addVerbose(self):
        self.add_argument(
            '-v', '--verbose', default=False, action='store_true', help='Increase verbosity')

    def addPlot(self):
        self.add_argument(
            '-p', '--plot', type=str, nargs='+', help='Variables to plot')

    def addMPI(self):
        self.add_argument(
            '--mpi', default=False, action='store_true', help='Use multiprocessing')

    def addOutputDir(self):
        self.add_argument(
            '-o', '--outputdir', type=str, help='Output directory')

    def parseDir(self, key, args):
        directory = args[key] if args[key] is not None else selectDirDialog()
        if directory == '':
            raise ValueError('No {} selected'.format(key))
        return directory

    def parseOutputDir(self, args):
        return self.parseDir('outputdir', args)

    def parseLogLevel(self, args):
        return logging.DEBUG if args.pop('verbose') else logging.INFO

    def parsePltScheme(self, args):
        if args['plot'] == ['all']:
            return None
        else:
            return {x: [x] for x in args['plot']}

    def restrict(self, args, keys):
        if sum([args[x] is not None for x in keys]) > 1:
            raise ValueError(
                'You must provide only one of the following arguments: {}'.format(', '.join(keys)))

    def parse2array(self, args, key, factor=1):
        return np.array(args[key]) * factor

    def parse(self):
        args = vars(super().parse_args())
        args['loglevel'] = self.parseLogLevel(args)
        args['outputdir'] = self.parseOutputDir(args)
        for k, v in self.defaults.items():
            if args[k] is None:
                args[k] = [v]
        return args


class MechSimParser(GenericParser):
    ''' Parser to run mechanical simulations from the command line. '''

    def __init__(self):
        super().__init__()

        self.defaults.update({
            'radius': 32.0,  # nm
            'embedding': 0.,  # um
            'Cm0': CorticalRS().Cm0 * 1e2,  # uF/m2
            'Qm0': CorticalRS().Qm0 * 1e5,  # nC/m2
            'freq': 500.0,  # kHz
            'amp': 100.0,  # kPa
            'charge': 0.  # nC/cm2
        })

        self.factors.update({
            'radius': 1e-9,
            'embedding': 1e-6,
            'Cm0': 1e-2,
            'Qm0': 1e-5,
            'freq': 1e3,
            'amp': 1e3,
            'charge': 1e-5
        })

        self.addRadius()
        self.addEmbedding()
        self.addCm0()
        self.addQm0()
        self.addFdrive()
        self.addAdrive()
        self.addCharge()

    def addRadius(self):
        self.add_argument(
            '-a', '--radius', nargs='+', type=float, help='Sonophore radius (nm)')

    def addEmbedding(self):
        self.add_argument(
            '--embedding', nargs='+', type=float, help='Embedding depth (um)')

    def addCm0(self):
        self.add_argument(
            '--Cm0', type=float, help='Resting membrane capacitance (uF/cm2)')

    def addQm0(self):
        self.add_argument(
            '--Qm0', type=float, help='Resting membrane charge density (nC/cm2)')

    def addFdrive(self):
        self.add_argument(
            '-f', '--freq', nargs='+', type=float, help='US frequency (kHz)')

    def addAdrive(self):
        self.add_argument(
            '-A', '--amp', nargs='+', type=float, help='Acoustic pressure amplitude (kPa)')
        self.add_argument(
            '--Arange', type=str, nargs='+',
            help='Amplitude range {} (kPa)'.format(self.dist_str))
        self.add_argument(
            '-I', '--intensity', nargs='+', type=float, help='Acoustic intensity (W/cm2)')
        self.add_argument(
            '--Irange', type=str, nargs='+',
            help='Intensity range {} (W/cm2)'.format(self.dist_str))

    def addCharge(self):
        self.add_argument(
            '-Q', '--charge', nargs='+', type=float, help='Membrane charge density (nC/cm2)')

    def parseAmp(self, args):
        params = ['Irange', 'Arange', 'intensity', 'amp']
        self.restrict(args, params[:-1])
        Irange, Arange, Int, Adrive = [args.pop(k) for k in params]
        if Irange is not None:
            return Intensity2Pressure(self.getDistFromList(Irange) * 1e4)  # Pa
        elif Int is not None:
            return Intensity2Pressure(np.array(Int) * 1e4)  # Pa
        elif Arange is not None:
            return self.getDistFromList(Arange) * self.factors['amp']  # Pa
        else:
            return np.array(Adrive) * self.factors['amp']  # Pa

    def parse(self):
        args = super().parse()
        args['amp'] = self.parseAmp(args)
        for key in ['radius', 'embedding', 'Cm0', 'Qm0', 'freq', 'charge']:
            args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args


class PWStimParser(GenericParser):
    ''' Generic parser interface to run PW patterned simulations from the command line. '''

    def __init__(self):
        super().__init__()

        self.defaults.update({
            'neuron': 'RS',
            'tstim': 100.0,  # ms
            'toffset': 50.,  # ms
            'PRF': 100.0,  # Hz
            'DC': 100.0  # %
        })

        self.factors.update({
            'tstim': 1e-3,
            'toffset': 1e-3,
            'PRF': 1.,
            'DC': 1e-2
        })

        self.allowed.update({
            'neuron': list(getNeuronsDict().keys()),
            'DC': range(101)
        })

        self.addNeuron()
        self.addTstim()
        self.addToffset()
        self.addPRF()
        self.addDC()
        self.addSpanDC()
        self.addTitrate()

    def addNeuron(self):
        self.add_argument(
            '-n', '--neuron', type=str, nargs='+',
            choices=self.allowed['neuron'], help='Neuron name (string)')

    def addTstim(self):
        self.add_argument(
            '-t', '--tstim', nargs='+', type=float, help='Stimulus duration (ms)')

    def addToffset(self):
        self.add_argument(
            '--toffset', nargs='+', type=float, help='Offset duration (ms)')

    def addPRF(self):
        self.add_argument(
            '--PRF', nargs='+', type=float, help='PRF (Hz)')

    def addDC(self):
        self.add_argument(
            '--DC', nargs='+', type=float, help='Duty cycle (%%)')

    def addSpanDC(self):
        self.add_argument(
            '--spanDC', default=False, action='store_true', help='Span DC from 1 to 100%%')

    def addTitrate(self):
        self.add_argument(
            '--titrate', default=False, action='store_true', help='Perform titration')

    def parseNeuron(self, args):
        for item in args['neuron']:
            if item not in self.allowed['neuron']:
                raise ValueError('Unknown neuron type: "{}"'.format(item))
        return [getNeuronsDict()[n]() for n in args['neuron']]

    def parseAmp(self, args):
        return NotImplementedError

    def parseDC(self, args):
        if args.pop('spanDC'):
            return np.arange(1, 101) * self.factors['DC']  # (-)
        else:
            return np.array(args['DC']) * self.factors['DC']  # (-)

    def parse(self, args=None):
        if args is None:
            args = super().parse()
        args['neuron'] = self.parseNeuron(args)
        args['DC'] = self.parseDC(args)
        for key in ['tstim', 'toffset', 'PRF']:
            args[key] = self.parse2array(args, key, factor=self.factors[key])
        return args


class EStimParser(PWStimParser):
    ''' Parser to run E-STIM simulations from the command line. '''

    def __init__(self):
        super().__init__()
        self.defaults.update({'amp': 10.0})  # mA/m2
        self.factors.update({'amp': 1.})
        self.addAstim()

    def addAstim(self):
        self.add_argument(
            '-A', '--amp', nargs='+', type=float,
            help='Amplitude of injected current density (mA/m2)')
        self.add_argument(
            '--Arange', type=str, nargs='+',
            help='Amplitude range {} (mA/m2)'.format(self.dist_str))

    def parseAmp(self, args):
        if args.pop('titrate'):
            return None
        Arange, Astim = [args.pop(k) for k in ['Arange', 'amp']]
        if Arange is not None:
            return self.getDistFromList(Arange) * self.factors['amp']  # mA/m2
        else:
            return np.array(Astim) * self.factors['amp']  # mA/m2

    def parse(self):
        args = super().parse()
        args['amp'] = self.parseAmp(args)
        return args


class AStimParser(PWStimParser, MechSimParser):
    ''' Parser to run A-STIM simulations from the command line. '''

    def __init__(self):
        MechSimParser.__init__(self)
        PWStimParser.__init__(self)
        self.defaults.update({'method': 'sonic'})
        self.allowed.update({'method': ['classic', 'hybrid', 'sonic']})
        self.addMethod()

    def addMethod(self):
        self.add_argument(
            '-m', '--method', nargs='+', type=str,
            help='Numerical integration method ({})'.format(', '.join(self.allowed['method'])))

    def parseMethod(self, args):
        for item in args['method']:
            if item not in self.allowed['method']:
                raise ValueError('Unknown neuron type: "{}"'.format(item))

    def parseAmp(self, args):
        if args.pop('titrate'):
            return None
        return MechSimParser.parseAmp(self, args)

    def parse(self):
        args = PWStimParser.parse(self, args=MechSimParser.parse(self))
        for k in ['Cm0', 'Qm0', 'embedding', 'charge']:
            del args[k]
        self.parseMethod(args)
        return args
