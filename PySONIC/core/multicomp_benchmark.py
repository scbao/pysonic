# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-09-24 15:30:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-09-24 18:44:18
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from PySONIC.core import EffectiveVariablesLookup
from ..utils import logger, timer, isWithin, si_format, rmse
from ..neurons import passiveNeuron


class SonicBenchmark:
    ''' Interface allowing to run benchmark simulations of a two-compartment model
        incorporating the SONIC paradigm, with a simplified sinusoidal capacitive drive.
    '''
    npc = 25          # number of samples per cycle
    min_ncycles = 10  # minimum number of cycles per simulation
    varunits = {
        't': 'ms',
        'Cm': 'uF/cm2',
        'Vm': 'mV',
        'Qm': 'nC/cm2'
    }
    varfactors = {
        't': 1e3,
        'Cm': 1e2,
        'Vm': 1e0,
        'Qm': 1e5
    }
    nodelabels = ['node 1', 'node 2']
    ga_bounds = [1e-10, 1e10]  # S/m2

    def __init__(self, pneuron, ga, f, gammas, passive=False):
        ''' Initialization.

            :param pneuron: point-neuron object
            :param ga: axial conductance (S/m2)
            :param f: US frequency (Hz)
            :param gammas: pair of relative capacitance oscillation ranges
        '''
        self.pneuron = pneuron
        self.ga = ga
        self.f = f
        self.gammas = gammas
        self.passive = passive
        self.computeLookups()

    def copy(self):
        return self.__class__(self.pneuron, self.ga, self.f, self.gammas, passive=self.passive)

    @property
    def strGammas(self):
        return f"({', '.join([f'{x:.2f}' for x in self.gammas])})"

    def __repr__(self):
        params = [
            f'ga = {self.ga:.2e} S/m2',
            f'f = {si_format(self.f)}Hz',
            f'gamma = {self.strGammas}'
        ]
        dynamics = 'passive ' if self.passive else ''
        mech = f'{dynamics}{self.pneuron.name} dynamics'
        return f'{self.__class__.__name__}({mech}, {", ".join(params)})'

    @property
    def pneuron(self):
        return self._pneuron

    @pneuron.setter
    def pneuron(self, value):
        self._pneuron = value.copy()
        self.states = self._pneuron.statesNames()
        if hasattr(self, 'lkps'):
            self.computeLookups()

    def isPassive(self):
        return self.pneuron.name.startswith('pas_')

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, value):
        self._f = value
        if hasattr(self, 'lkps'):
            self.computeLookups()

    @property
    def gammas(self):
        return self._gammas

    @gammas.setter
    def gammas(self, value):
        self._gammas = value
        if hasattr(self, 'lkps'):
            self.computeLookups()

    @property
    def passive(self):
        return self._passive

    @passive.setter
    def passive(self, value):
        assert isinstance(value, bool), 'passive must be boolean typed'
        self._passive = value
        if hasattr(self, 'lkps'):
            self.computeLookups()

    @property
    def ga(self):
        return self._ga

    @ga.setter
    def ga(self, value):
        if value != 0.:
            assert isWithin('ga', value, self.ga_bounds)
        self._ga = value

    @property
    def gPas(self):
        ''' Passive membrane conductance (S/m2). '''
        return self.pneuron.gLeak

    @property
    def Cm0(self):
        ''' Resting capacitance (F/m2). '''
        return self.pneuron.Cm0

    @property
    def Vm0(self):
        ''' Resting membrane potential (mV). '''
        return self.pneuron.Vm0

    @property
    def Qm0(self):
        ''' Resting membrane charge density (C/m2). '''
        return self.Vm0 * self.Cm0 * 1e-3

    @property
    def Qref(self):
        ''' Reference charge linear space. '''
        return np.arange(*self.pneuron.Qbounds, 1e-5)  # C/cm2

    @property
    def Cmeff(self):
        ''' Analytical solution for effective membrane capacitance (F/m2). '''
        return self.Cm0 * np.sqrt(1 - np.array(self.gammas)**2 / 4)

    @property
    def Qminf(self):
        ''' Analytical solution for steady-state charge density (C/m2). '''
        return self.Cmeff * self.pneuron.ELeak * 1e-3

    def capct(self, gamma, t):
        ''' Time-varying sinusoidal capacitance (in F/m2) '''
        return self.Cm0 * (1 + 0.5 * gamma * np.sin(2 * np.pi * self.f * t))

    def vCapct(self, t):
        ''' Vector of time-varying capacitance (in F/m2) '''
        return np.array([self.capct(gamma, t) for gamma in self.gammas])

    def getLookup(self, Cm):
        ''' Get a lookup object of effective variables for a given capacitance cycle vector. '''
        Vmarray = np.array([Q / Cm for Q in self.Qref]) * 1e3  # mV
        refs = {'Q': self.Qref}  # C/m2
        tables = {
            k: np.array([np.mean(np.vectorize(v)(Vmvec)) for Vmvec in Vmarray])
            for k, v in self.pneuron.effRates().items()
        }
        return EffectiveVariablesLookup(refs, tables)

    @property
    def tcycle(self):
        ''' Time vector over 1 acoustic cycle (s). '''
        return np.linspace(0, 1 / self.f, self.npc)

    @property
    def dt_full(self):
        ''' Full time step (s). '''
        return 1 / (self.npc * self.f)

    @property
    def dt_sparse(self):
        ''' Sparse time step (s). '''
        return 1 / self.f

    def computeLookups(self):
        ''' Compute benchmark lookups. '''
        self.lkps = []
        if not self.passive:
            self.lkps = [self.getLookup(Cm_cycle) for Cm_cycle in self.vCapct(self.tcycle)]

    def getCmeff(self, Cm_cycle):
        ''' Compute effective capacitance from capacitance profile over 1 cycle. '''
        return 1 / np.mean(1 / Cm_cycle)  # F/m2

    def iax(self, Vm, Vmother):
        ''' Compute axial current flowing in the compartment from another compartment (in mA/m2).

            [iax] = S/m2 * mV = 1e-3 A/m2 = 1 mA/m2
        '''
        return self.ga * (Vmother - Vm)

    def vIax(self, Vm):
        ''' Compute array of axial currents in each compartment based on array of potentials. '''
        return np.array([self.iax(*Vm), self.iax(*Vm[::-1])])  # mA/m2

    def serialize(self, y):
        ''' Serialize a single state vector into a state-per-node matrix. '''
        return np.reshape(y.copy(), (self.npernode * 2))

    def deserialize(self, y):
        ''' Deserialize a state per node matrix into a single state vector. '''
        return np.reshape(y.copy(), (2, self.npernode))

    def derivatives(self, t, y, Cm, dstates_func):
        ''' Generic derivatives method. '''
        # Deserialize states vector and initialize derivatives array
        y = self.deserialize(y)
        dydt = np.empty(y.shape)

        # Extract charge density and membrane potential vectors
        Qm = y[:, 0]  # C/m2
        Vm = y[:, 0] / Cm * 1e3  # mV

        # Extract states array
        states_array = y[:, 1:]

        # Compute membrane dynamics for each node
        for i, (qm, vm, states) in enumerate(zip(Qm, Vm, states_array)):
            # If passive, compute only leakage current
            if self.passive:
                im = self.pneuron.iLeak(vm)  # mA/m2
            # Otherwise, compute states derivatives and total membrane current
            if not self.passive:
                states_dict = dict(zip(self.states, states))
                dydt[i, 1:] = dstates_func(i, qm, vm, states_dict)  # s-1
                im = self.pneuron.iNet(vm, states_dict)  # mA/m2
            dydt[i, 0] = -im  # mA/m2

        # Add axial currents to currents column
        dydt[:, 0] += self.vIax(Vm)  # mA/m2

        # Rescale currents column into charge derivative units
        dydt[:, 0] *= 1e-3  # C/m2.s

        # Return serialized derivatives vector
        return self.serialize(dydt)

    def dstatesFull(self, i, qm, vm, states):
        ''' Compute detailed states derivatives. '''
        return self.pneuron.getDerStates(vm, states)

    def dfull(self, t, y):
        ''' Compute detailed derivatives vector. '''
        return self.derivatives(t, y, self.vCapct(t), self.dstatesFull)

    def dstatesEff(self, i, qm, vm, states):
        ''' Compute effective states derivatives. '''
        lkp0d = self.lkps[i].interpolate1D(qm)
        return np.array([self.pneuron.derEffStates()[k](lkp0d, states) for k in self.states])

    def deff(self, t, y):
        ''' Compute effective derivatives vector. '''
        return self.derivatives(t, y, self.Cmeff, self.dstatesEff)

    @property
    def y0node(self):
        ''' Get initial conditions vector (common to every node). '''
        if self.passive:
            return [self.Qm0]
        else:
            return [self.Qm0, *[self.pneuron.steadyStates()[k](self.Vm0) for k in self.states]]

    @property
    def y0(self):
        ''' Get full initial conditions vector (duplicated ynode vector). '''
        self.npernode = len(self.y0node)
        return self.y0node + self.y0node

    def integrate(self, dfunc, t):
        ''' Integrate over a time vector and return charge density arrays. '''
        # Integrate system
        y = odeint(dfunc, self.y0, t, tfirst=True).T

        # Cast each solution variable as a time-per-node matrix
        sol = {'Qm': y[::self.npernode]}
        if not self.passive:
            for i, k in enumerate(self.states):
                sol[k] = y[i + 1::self.npernode]

        # Return recast solution dictionary
        return sol

    def orderedKeys(self, varkeys):
        ''' Get ordered list of solution keys. '''
        mainkeys = ['Qm', 'Vm', 'Cm']
        otherkeys = list(set(varkeys) - set(mainkeys))
        return mainkeys + otherkeys

    def orderedSol(self, sol):
        ''' Re-order solution according to keys list. '''
        return {k: sol[k] for k in self.orderedKeys(sol.keys())}

    def nsamples(self, tstop):
        ''' Compute the number of samples required to integrate over a given time interval. '''
        return self.getNCycles(tstop) * self.npc

    @timer
    def simFull(self, tstop):
        ''' Simulate the full system until a specific stop time. '''
        t = np.linspace(0, tstop, self.nsamples(tstop))
        sol = self.integrate(self.dfull, t)
        sol['Cm'] = self.vCapct(t)
        sol['Vm'] = sol['Qm'] / sol['Cm'] * 1e3
        return t, self.orderedSol(sol)

    @timer
    def simEff(self, tstop):
        ''' Simulate the effective system until a specific stop time. '''
        t = np.linspace(0, tstop, self.getNCycles(tstop))
        sol = self.integrate(self.deff, t)
        sol['Cm'] = np.array([np.ones(t.size) * Cmeff for Cmeff in self.Cmeff])
        sol['Vm'] = sol['Qm'] / sol['Cm'] * 1e3
        return t, self.orderedSol(sol)

    @property
    def methods(self):
        ''' Dictionary of simulation methods. '''
        return {'full': self.simFull, 'effective': self.simEff}

    def getNCycles(self, duration):
        ''' Compute number of cycles from a duration. '''
        return int(np.ceil(duration * self.f))

    def simulate(self, mtype, tstop):
        ''' Simulate the system with a specific method for a given duration. '''
        # Cast tstop as a multiple of the acoustic period
        tstop = self.getNCycles(tstop) / self.f  # s

        # Retrieve simulation method
        try:
            method = self.methods[mtype]
        except KeyError:
            raise ValueError(f'"{mtype}" is not a valid method type')

        # Run simulation and return output
        logger.debug(f'running {mtype} {tstop:.2f} ms simulation')
        output, tcomp = method(tstop)
        logger.debug(f'completed in {tcomp:.2f} s')
        return output

    def cycleAvg(self, y):
        ''' Cycle-average a solution vector according to the number of samples per cycle. '''
        ypercycle = np.reshape(y, (int(y.shape[0] / self.npc), self.npc))
        return np.mean(ypercycle, axis=1)

    def cycleAvgSol(self, t, sol):
        ''' Cycle-average a time vector and a solution dictionary. '''
        solavg = {}
        # For each per-node-matrix in the solution
        for k, ymat in sol.items():
            # Cycle-average each node vector of the matrix
            solavg[k] = np.array([self.cycleAvg(yvec) for yvec in ymat])

        # Re-sample time vector at system periodicity
        tavg = t[::self.npc]  # + 0.5 / self.f

        # Return cycle-averaged time vector and solution dictionary
        return tavg, solavg

    def g2tau(self, g):
        ''' Convert conductance per unit membrane area (S/m2) to time constant (s). '''
        return self.Cm0 / g  # ms

    def tau2g(self, tau):
        ''' Convert time constant (s) to conductance per unit membrane area (S/m2). '''
        return self.Cm0 / tau  # s

    @property
    def taum(self):
        ''' Passive membrane time constant (s). '''
        return self.pneuron.tau_pas

    @taum.setter
    def taum(self, value):
        ''' Update point-neuron leakage conductance to match time new membrane time constant. '''
        if not self.isPassive():
            raise ValueError('taum can only be set for passive neurons')
        self.pneuron = passiveNeuron(
            self.pneuron.Cm0,
            self.tau2g(value),  # S/m2
            self.pneuron.ELeak)

    @property
    def tauax(self):
        ''' Axial time constant (s). '''
        return self.g2tau(self.ga)

    @tauax.setter
    def tauax(self, value):
        ''' Update axial conductance per unit area to match time new axial time constant. '''
        self.ga = self.tau2g(value)  # S/m2

    @property
    def taumax(self):
        ''' Maximal time constant of the model (s). '''
        return max(self.taum, self.tauax)

    def setTimeConstants(self, taum, tauax):
        ''' Update benchmark according to pair of time constants (in s). '''
        self.taum = taum  # s
        self.tauax = tauax  # s

    def setDrive(self, f, gammas):
        ''' Update benchmark drive to a new frequency and amplitude. '''
        self.f = f
        self.gammas = gammas

    def getPassiveTstop(self, f):
        ''' Compute minimum simulation time for a passive model (s). '''
        return max(5 * self.taumax, self.min_ncycles / f)

    @property
    def passive_tstop(self):
        return self.getPassiveTstop(self.f)

    def simAllMethods(self, tstop):
        ''' Simulate the model with both methods. '''
        logger.info(f'{self}: {si_format(tstop)}s simulation')
        # Simulate with full and effective systems
        t, sol = {}, {}
        for method in self.methods.keys():
            t[method], sol[method] = self.simulate(method, tstop)
        t, sol = self.postproSol(t, sol)
        return t, sol

    def computeGradient(self, sol):
        ''' compute the gradient of a solution array. '''
        return {k: np.vstack((y, np.diff(y, axis=0))) for k, y in sol.items()}

    def addOnset(self, ymat, y0):
        return np.hstack((np.ones((2, 2)) * y0, ymat))

    def postproSol(self, t, sol):
        ''' Post-process solution. '''
        # Add cycle-average of full solution
        t['cycle-avg'], sol['cycle-avg'] = self.cycleAvgSol(t['full'], sol['full'])
        keys = list(sol.keys())

        tonset = 0.05 * np.ptp(t['full'])
        # Add onset
        y0dict = {'Cm': self.Cm0, 'Qm': self.Qm0, 'Vm': self.Vm0}
        for k in keys:
            t[k] = np.hstack(([-tonset, 0], t[k]))
            sol[k] = {vk: self.addOnset(ymat, y0dict[vk]) for vk, ymat in sol[k].items()}

        # Add gradient across nodes for each variable
        for k in keys:
            sol[f'{k}-grad'] = self.computeGradient(sol[k])
        return t, sol

    def plot(self, t, sol, Qonly=False, gradient=False):
        ''' Plot results of benchmark simulations of the model. '''
        colors = ['C0', 'C1', 'darkgrey']
        markers = ['-', '--', '-']
        alphas = [0.5, 1., 1.]

        # Reduce solution dictionary if only Q needs to be plotted
        if Qonly:
            sol = {key: {'Qm': value['Qm']} for key, value in sol.items()}

        # Extract simulation duration
        tstop = t[list(t.keys())[0]][-1]  # s

        # Gather keys of methods and variables to plot
        mkeys = list(sol.keys())
        varkeys = list(sol[mkeys[0]].keys())
        naxes = len(varkeys)

        # Get node labels
        lbls = self.nodelabels
        if gradient:
            lbls.append('gradient')

        # Create figure
        fig, axes = plt.subplots(naxes, 1, sharex=True, figsize=(10, min(3 * naxes, 10)))
        if naxes == 1:
            axes = [axes]
        axes[0].set_title(f'{self} - {si_format(tstop)}s simulation')
        axes[-1].set_xlabel(f'time ({self.varunits["t"]})')
        for ax, vk in zip(axes, varkeys):
            ax.set_ylabel(f'{vk} ({self.varunits.get(vk, "-")})')

        # Add horizontal lines for node-specific SONIC steady-states on charge density plot
        Qm_ax = axes[varkeys.index('Qm')]
        for Qm, c in zip(self.Qminf, colors):
            Qm_ax.axhline(Qm * self.varfactors['Qm'], c=c, linestyle=':')
        if gradient:
            Qm_ax.axhline(np.diff(self.Qminf) * self.varfactors['Qm'], c=colors[-1], linestyle=':')

        # For each solution type
        for m, alpha, (mkey, varsdict) in zip(markers, alphas, sol.items()):
            tplt = t[mkey] * self.varfactors['t']
            # For each solution variable
            for ax, (vkey, v) in zip(axes, varsdict.items()):
                # For each node
                for y, c, lbl in zip(v, colors, lbls):
                    # Plot node variable with appropriate color and marker
                    ax.plot(tplt, y * self.varfactors[vkey],
                            m, alpha=alpha, c=c, label=f'{lbl} - {mkey}')

        # Add legend
        fig.subplots_adjust(bottom=0.2)
        axes[-1].legend(
            bbox_to_anchor=(0., -0.7, 1., .1), loc='upper center',
            ncol=3, mode="expand", borderaxespad=0.)

        # Return figure
        return fig

    def plotV(self, t, sol):
        ''' Plot results of benchmark simulations of the model. '''
        colors = ['C0', 'C1']
        markers = ['-', '--', '-']
        alphas = [0.5, 1., 1.]
        V = {key: value['Qm'] / self.Cm0 for key, value in sol.items()}
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_title(f'{self} - {t[list(t.keys())[0]][-1]:.2f} ms simulation')
        ax.set_xlabel(f'time ({self.varunits["t"]})')
        ax.set_ylabel(f'Qm / Cm0 (mV)')
        ax.set_ylim(-100.0, 50.)
        for m, alpha, (key, varsdict) in zip(markers, alphas, sol.items()):
            for y, c, lbl in zip(V[key], colors, self.nodelabels):
                ax.plot(t[key] * self.varfactors['t'], y * 1e3,
                        m, alpha=alpha, c=c, label=f'{lbl} - {key}')
        fig.subplots_adjust(bottom=0.2)
        ax.legend(bbox_to_anchor=(0., -0.7, 1., .1), loc='upper center', ncol=3,
                  mode="expand", borderaxespad=0.)
        return fig

    def simplot(self, *args, **kwargs):
        ''' Run benchmark simulation and plot results. '''
        return self.plot(*self.simAllMethods(*args, **kwargs))

    @property
    def eval_funcs(self):
        ''' Different functions to evaluate the divergence between two solutions. '''
        return {
            'rmse': lambda y1, y2: rmse(y1, y2),            # RMSE
            'ss': lambda y1, y2: np.abs(y1[-1] - y2[-1]),   # steady-state absolute difference
            'amax': lambda y1, y2: np.max(np.abs(y1 - y2))  # max absolute difference
        }

    def divergencePerNode(self, t, sol, eval_mode='RMSE'):
        ''' Evaluate the divergence between the effective and full, cycle-averaged solutions
            at a specific point in time, computing per-node differences in charge density values
            divided by resting capacitance.
        '''
        if eval_mode not in self.eval_funcs.keys():
            raise ValueError(f'{eval_mode} evaluation mode is not supported')

        # Extract charge matrices from solution dictionary
        Qsol = {k: sol[k]['Qm'] for k in ['effective', 'cycle-avg']}  # C/m2

        # Normalize matrices by resting capacitance
        Qnorm = {k: v / self.Cm0 * 1e3 for k, v in Qsol.items()}  # mV

        # Keep only the first two rows (3rd one, if any, is a gradient)
        Qnorm = {k: v[:2, :] for k, v in Qnorm.items()}

        # Discard the first 3 columns (artifical onset and first cycle artefact)
        Qnorm = {k: v[:, 3:] for k, v in Qnorm.items()}

        eval_func = self.eval_funcs[eval_mode]

        # Compute deviation across nodes saccording to evaluation mode
        div_per_node = [eval_func(*[v[i] for v in Qnorm.values()]) for i in range(2)]

        # Cast into dictionary and return
        div_per_node = dict(zip(self.nodelabels, div_per_node))
        logger.debug(f'divergence per node: ', {k: f'{v:.2e} mV' for k, v in div_per_node.items()})
        return div_per_node

    def divergence(self, *args, **kwargs):
        div_per_node = self.divergencePerNode(*args, **kwargs)  # mV
        return max(list(div_per_node.values()))                 # mV
