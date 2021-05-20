# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2021-05-14 19:42:00
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-05-20 10:09:19

import numpy as np
import matplotlib.pyplot as plt

from ..core import NeuronalBilayerSonophore, PulsedProtocol, Batch
from ..utils import si_format
from ..neurons import passiveNeuron
from ..postpro import gamma
from ..plt import harmonizeAxesLimits, hideSpines, hideTicks, addXscale, addYscale
from .coupled_nbls import CoupledSonophores


class Benchmark:

    def __init__(self, a, nnodes, outdir=None):
        self.a = a
        self.nnodes = nnodes
        self.outdir = outdir

    def runSims(self, model, drives, tstim, covs):
        ''' Run full and sonic simulations for a specific combination drives,
            pulsed protocol and coverage fractions, harmonize outputs and compute
            normalized charge density profiles.
        '''
        Fdrive = drives[0].f
        assert all(x.f == Fdrive for x in drives), 'frequencies do not match'
        assert len(covs) == model.nnodes, 'coverages do not match model dimensions'
        assert len(drives) == model.nnodes, 'drives do not match model dimensions'

        # If not provided, compute stimulus duration from model passive properties
        min_ncycles = 10
        ntaumax_conv = 5
        if tstim is None:
            tstim = max(ntaumax_conv * model.taumax, min_ncycles / Fdrive)
        # Recast stimulus duration as finite multiple of acoustic period
        tstim = int(np.ceil(tstim * Fdrive)) / Fdrive  # s

        # Pulsed protocol
        pp = PulsedProtocol(tstim, 0)

        # Simulate/Load with full and sonic methods
        data, meta = {}, {}
        for method in ['full', 'sonic']:
            data[method], meta[method] = model.simAndSave(
                drives, pp, covs, method, outdir=self.outdir,
                overwrite=False, minimize_output=True)

        # Cycle-average full solution and interpolate sonic solution along same time vector
        data['cycleavg'] = data['full'].cycleAveraged(1 / Fdrive)
        data['sonic'] = data['sonic'].interpolate(data['cycleavg'].time)

        # Compute normalized charge density profiles and add them to dataset
        for simkey, simdata in data.items():
            for nodekey, nodedata in simdata.items():
                nodedata['Qnorm'] = nodedata['Qm'] / model.refpneuron.Cm0 * 1e3  # mV

        # Return dataset
        return data, meta

    def computeGamma(self, data, *args):
        ''' Perform per-node gamma evaluation on charge density profiles. '''
        gamma_dict = {}
        for k in data['cycleavg'].keys():
            Qnorms = [data[simkey][k]['Qnorm'].values for simkey in ['cycleavg', 'sonic']]
            gamma_dict[k] = gamma(*Qnorms, *args)
            # Discard 1st and last indexes of evaluation
            gamma_dict[k] = np.hstack(([np.nan], gamma_dict[k][1:-1], [np.nan]))
        return gamma_dict

    def plotSignals(self, ax, data):
        ''' Plot normalized charge density signals on an axis. '''
        markers = {'full': '-', 'cycleavg': '--', 'sonic': '-'}
        alphas = {'full': 0.5, 'cycleavg': 1., 'sonic': 1.}
        for simkey, simdata in data.items():
            for i, (nodekey, nodedata) in enumerate(simdata.items()):
                ax.plot(nodedata.time * 1e3, nodedata['Qnorm'], markers[simkey], c=f'C{i}',
                        alpha=alphas[simkey], label=f'{simkey} - {nodekey}')


class PassiveBenchmark(Benchmark):

    def __init__(self, a, nnodes, Cm0, ELeak, **kwargs):
        super().__init__(a, nnodes, **kwargs)
        self.Cm0 = Cm0
        self.ELeak = ELeak

    def getModelAndRunSims(self, drives, covs, taum, tauax):
        ''' Create passive model for a combination of time constants. '''
        gLeak = self.Cm0 / taum
        ga = self.Cm0 / tauax
        pneuron = passiveNeuron(self.Cm0, gLeak, self.ELeak)
        model = CoupledSonophores([
            NeuronalBilayerSonophore(self.a, pneuron) for i in range(self.nnodes)], ga)
        return self.runSims(model, drives, None, covs)

    def runSimsOverTauSpace(self, drives, covs, taum_range, tauax_range, mpi=False):
        ''' Run simulations over 2D time constant space. '''
        queue = [[drives, covs] + x for x in Batch.createQueue(taum_range, tauax_range)]
        batch = Batch(self.getModelAndRunSims, queue)
        # batch.printQueue(queue)
        return batch.run(mpi=mpi)

    def plotSignalsOverTauSpace(self, taum_range, tauax_range, results, fs=10):
        ''' Plot signals over 2D time constants space. '''
        fig, axes = plt.subplots(taum_range.size, tauax_range.size)
        fig.suptitle('passive neuron - normalized charge densities', fontsize=fs + 2)
        fig.supxlabel('tau_ax', fontsize=fs + 2)
        fig.supylabel('tau_m', fontsize=fs + 2)
        i = 0
        tmin = np.inf
        for axrow in axes[::-1]:
            for ax in axrow:
                self.plotSignals(ax, results[i])
                hideSpines(ax)
                hideTicks(ax)
                ax.margins(0)
                tmin = min(tmin, np.ptp(ax.get_xlim()))
                # addXscale(ax, 0, 0, unit='ms', fmt='.2f', fs=fs)
                i += 1
        for axrow in axes[::-1]:
            for ax in axrow:
                trans = (ax.transData + ax.transAxes.inverted())
                xpoints = [trans.transform([x, 0])[0] for x in [0, tmin]]
                ax.plot(xpoints, [-0.05] * 2, c='k', lw=2, transform=ax.transAxes, clip_on=False)
        harmonizeAxesLimits(axes, dim='y')
        addYscale(axes[-1, -1], 0.05, 0, unit='mV', fmt='.0f', fs=fs)
        for ax, tauax in zip(axes[-1, :], tauax_range):
            ax.set_xlabel(f'{si_format(tauax)}s', labelpad=15, fontsize=fs + 2)
        for ax, taum in zip(axes[:, 0], taum_range[::-1]):
            ax.set_ylabel(f'{si_format(taum)}s', labelpad=15, fontsize=fs + 2)
        fig.tight_layout()
        return fig
