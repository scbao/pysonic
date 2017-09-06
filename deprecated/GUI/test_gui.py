#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-25 17:16:56
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-05 14:44:29

''' Simple GUI to run ASTIM and ESTIM simulations. '''

import logging
import tkinter as tk

from PointNICE.solvers import SolverUS, checkBatchLog, runAStim
from PointNICE.utils import getNeuronsDict, load_BLS_params



class UI(tk.Tk):
    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self.parent = parent
        self.initialize()

    def initialize(self):

        self.neurons = getNeuronsDict()
        self.bls_params = load_BLS_params()
        self.batch_dir = ''
        self.log_filepath = ''

        self.grid()

        # ---------------------- Neuron parameters ----------------------

        frameNeuron = tk.LabelFrame(self, text="Cell parameters", padx=10, pady=10)
        frameNeuron.pack(padx=20, pady=5, fill=tk.X)

        neurons_names = list(self.neurons.keys())
        labelNeuronType = tk.Label(frameNeuron, text='Neuron type', anchor="w")
        labelNeuronType.grid(column=0, row=0, sticky='E')
        self.neuron_name = tk.StringVar()
        self.neuron_name.set(neurons_names[0])
        neurons_drop = tk.OptionMenu(frameNeuron, self.neuron_name, *neurons_names)
        neurons_drop.grid(column=1, row=0, sticky='E')

        label_diam = tk.Label(frameNeuron, text='BLS diameter (nm)', anchor="w")
        label_diam.grid(column=0, row=1, sticky='E')
        self.diam = tk.DoubleVar()
        self.entry_diam = tk.Entry(frameNeuron, textvariable=self.diam, width=8)
        self.entry_diam.grid(column=1, row=1, sticky='W')
        self.diam.set(32.0)

        # ---------------------- ASTIM parameters ----------------------

        frameASTIM = tk.LabelFrame(self, text="Stimulation parameters", padx=10, pady=10)
        frameASTIM.pack(padx=20, pady=5, fill=tk.X)

        labelFdrive = tk.Label(frameASTIM, text='Frequency (kHz)', anchor="w")
        labelFdrive.grid(column=0, row=0, sticky='E')
        self.Fdrive = tk.DoubleVar()
        self.entryFdrive = tk.Entry(frameASTIM, textvariable=self.Fdrive, width=8)
        self.entryFdrive.grid(column=1, row=0, sticky='W')
        self.Fdrive.set(200.0)

        labelAdrive = tk.Label(frameASTIM, text='Amplitude (kPa)', anchor="w")
        labelAdrive.grid(column=0, row=1, sticky='E')
        self.Adrive = tk.DoubleVar()
        self.entryAdrive = tk.Entry(frameASTIM, textvariable=self.Adrive, width=8)
        self.entryAdrive.grid(column=1, row=1, sticky='W')
        self.Adrive.set(300.0)

        labeltstim = tk.Label(frameASTIM, text='Duration (ms)', anchor="w")
        labeltstim.grid(column=0, row=2, sticky='E')
        self.tstim = tk.DoubleVar()
        self.entrytstim = tk.Entry(frameASTIM, textvariable=self.tstim, width=8)
        self.entrytstim.grid(column=1, row=2, sticky='W')
        self.tstim.set(100.0)

        labeltoffset = tk.Label(frameASTIM, text='Offset (ms)', anchor="w")
        labeltoffset.grid(column=0, row=3, sticky='E')
        self.toffset = tk.DoubleVar()
        self.entrytoffset = tk.Entry(frameASTIM, textvariable=self.toffset, width=8)
        self.entrytoffset.grid(column=1, row=3, sticky='W')
        self.toffset.set(20.0)

        labelPRF = tk.Label(frameASTIM, text='PRF (Hz)', anchor="w")
        labelPRF.grid(column=0, row=4, sticky='E')
        self.PRF = tk.DoubleVar()
        self.entryPRF = tk.Entry(frameASTIM, textvariable=self.PRF, width=8)
        self.entryPRF.grid(column=1, row=4, sticky='W')
        self.PRF.set(100.0)

        labelDF = tk.Label(frameASTIM, text='Duty cycle (%)', anchor="w")
        labelDF.grid(column=0, row=5, sticky='E')
        self.DF = tk.DoubleVar()
        self.entryDF = tk.Entry(frameASTIM, textvariable=self.DF, width=8)
        self.entryDF.grid(column=1, row=5, sticky='W')
        self.DF.set(100.0)

        # ---------------------- Simulation settings  ----------------------

        frameRun = tk.LabelFrame(self, text="Simulation settings", padx=10, pady=10)
        frameRun.pack(padx=20, pady=5, fill=tk.X)

        # frameRun = tk.Frame(self, padx=10, pady=10)
        # frameRun.pack(padx=20, fill=tk.X)

        selectdir_button = tk.Button(frameRun, text='...', command=self.OnSelectDirClick)
        selectdir_button.grid(column=0, row=0, sticky='EW')

        self.simdir = tk.StringVar()
        label_simdir = tk.Label(frameRun, textvariable=self.simdir, bg="white", width=65)
        label_simdir.grid(column=1, row=0, sticky='E')
        self.simdir.set('Output directory')

        button = tk.Button(frameRun, text=u"Run", command=self.OnRunClick)
        button.grid(column=0, row=1, columnspan=2, sticky='EW')

        self.labelVariable = tk.StringVar()
        label = tk.Label(frameRun, textvariable=self.labelVariable, anchor="w",
                         fg="white", bg="blue", width=70)
        label.grid(column=0, row=2, columnspan=2, sticky='EW')
        self.labelVariable.set('...')


        # ---------------------- Grid settings ----------------------

        # self.grid_columnconfigure(1, weight=1)
        self.resizable(False, False)




    def OnSelectDirClick(self):
        try:
            self.batch_dir, self.log_filepath = checkBatchLog('A-STIM')
            self.simdir.set(self.batch_dir)
        except AssertionError:
            self.simdir.set('Output directory')


    def OnRunClick(self):

        neuron = self.neurons[self.neuron_name.get()]()
        a = float(self.entry_diam.get())
        Fdrive = float(self.entryFdrive.get()) * 1e3
        Adrive = float(self.entryAdrive.get()) * 1e3
        tstim = float(self.entrytstim.get()) * 1e-3
        toffset = float(self.entrytoffset.get()) * 1e-3
        PRF = float(self.entryPRF.get())
        DF = float(self.entryDF.get()) * 1e-2

        if DF == 100.0:
            log_str = ('Running ASTIM on {} neuron @ {:.0f} kHz, {:.0f} kPa, {:.0f} ms'
                       .format(neuron.name, Fdrive * 1e-3, Adrive * 1e-3, tstim * 1e3))
        else:
            log_str = ('Running ASTIM on {} neuron @ {:.0f} kHz, {:.0f} kPa, {:.0f} ms, '
                       '{:.0f} Hz PRF, {:.1f} % DC'.format(neuron.name, Fdrive * 1e-3,
                                                           Adrive * 1e-3, tstim * 1e3,
                                                           PRF, DF * 1e2))
        self.labelVariable.set(log_str + ' ...')

        solver = SolverUS({'a': a * 1e-9, 'd': 0.0}, self.bls_params, neuron, Fdrive)
        output_filepath = runAStim(self.batch_dir, self.log_filepath, solver, neuron,
                                   self.bls_params, Fdrive, Adrive, tstim, toffset, PRF, DF,
                                   'effective')
        self.labelVariable.set('results available @ {}'.format(output_filepath))


if __name__ == "__main__":
    app = UI(None)
    app.title('PointNICE UI')
    app.mainloop()
