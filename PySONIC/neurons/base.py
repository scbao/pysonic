#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-03 11:53:04
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-21 15:27:49

''' Module standard API for all neuron mechanisms.

    Each mechanism class can use different methods to define the membrane dynamics of a
    specific neuron type. However, they must contain some mandatory attributes and methods
    in order to be properly imported in other sonic modules and used in NICE simulations.
'''

import abc


class BaseMech(metaclass=abc.ABCMeta):
    ''' Abstract class defining the common API (i.e. mandatory attributes and methods) of all
        subclasses implementing the channels mechanisms of specific neurons.

        The mandatory attributes are:
            - **name**: a string defining the name of the mechanism.
            - **Cm0**: a float defining the membrane resting capacitance (in F/m2)
            - **Vm0**: a float defining the membrane resting potential (in mV)
            - **states_names**: a list of strings defining the names of the different state
              probabilities governing the channels behaviour (i.e. the differential HH variables).
            - **states0**: a 1D array of floats (NOT integers !!!) defining the initial values of
              the different state probabilities.
            - **coeff_names**: a list of strings defining the names of the different coefficients
              to be used in effective simulations.

        The mandatory methods are:
            - **currNet**: compute the net ionic current density (in mA/m2) across the membrane,
              given a specific membrane potential (in mV) and channel states.
            - **steadyStates**: compute the channels steady-state values for a specific membrane
              potential value (in mV).
            - **derStates**: compute the derivatives of channel states, given a specific membrane
              potential (in mV) and channel states. This method must return a list of derivatives
              ordered identically as in the states0 attribute.
            - **getEffRates**: get the effective rate constants of ion channels to be used in
              effective simulations. This method must return an array of effective rates ordered
              identically as in the coeff_names attribute.
            - **derStatesEff**: compute the effective derivatives of channel states, based on
              1-dimensional linear interpolators of "effective" coefficients. This method must
              return a list of derivatives ordered identically as in the states0 attribute.
    '''

    @property
    @abc.abstractmethod
    def name(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def Cm0(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def Vm0(self):
        return 'Should never reach here'

    # @property
    # @abc.abstractmethod
    # def states_names(self):
    #     return 'Should never reach here'

    # @property
    # @abc.abstractmethod
    # def states0(self):
    #     return 'Should never reach here'

    # @property
    # @abc.abstractmethod
    # def coeff_names(self):
    #     return 'Should never reach here'

    @abc.abstractmethod
    def currNet(self, Vm, states):
        ''' Compute the net ionic current per unit area.

            :param Vm: membrane potential (mV)
            :states: state probabilities of the ion channels
            :return: current per unit area (mA/m2)
        '''

    @abc.abstractmethod
    def steadyStates(self, Vm):
        ''' Compute the channels steady-state values for a specific membrane potential value.

            :param Vm: membrane potential (mV)
            :return: array of steady-states
        '''

    @abc.abstractmethod
    def derStates(self, Vm, states):
        ''' Compute the derivatives of channel states.

            :param Vm: membrane potential (mV)
            :states: state probabilities of the ion channels
            :return: current per unit area (mA/m2)
        '''

    @abc.abstractmethod
    def getEffRates(self, Vm):
        ''' Get the effective rate constants of ion channels, averaged along an acoustic cycle,
            for future use in effective simulations.

            :param Vm: array of membrane potential values for an acoustic cycle (mV)
            :return: an array of rate average constants (s-1)
        '''

    @abc.abstractmethod
    def derStatesEff(self, Qm, states, interp_data):
        ''' Compute the effective derivatives of channel states, based on
            1-dimensional linear interpolation of "effective" coefficients
            that summarize the system's behaviour over an acoustic cycle.

            :param Qm: membrane charge density (C/m2)
            :states: state probabilities of the ion channels
            :param interp_data: dictionary of 1D vectors of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
        '''


    def getGates(self):
        ''' Retrieve the names of the neuron's states that match an ion channel gating. '''

        gates = []
        for x in self.states_names:
            if 'alpha{}'.format(x.lower()) in self.coeff_names:
                gates.append(x)
        return gates


    def getRates(self, Vm):
        ''' Compute the ion channels rate constants for a given membrane potential.

            :param Vm: membrane potential (mV)
            :return: a dictionary of rate constants and their values at the given potential.
        '''

        rates = {}
        for x in self.getGates():
            x = x.lower()
            alpha_str, beta_str = ['{}{}'.format(s, x.lower()) for s in ['alpha', 'beta']]
            inf_str, tau_str = ['{}inf'.format(x.lower()), 'tau{}'.format(x.lower())]
            if hasattr(self, 'alpha{}'.format(x)):
                alphax = getattr(self, alpha_str)(Vm)
                betax = getattr(self, beta_str)(Vm)
            elif hasattr(self, '{}inf'.format(x)):
                xinf = getattr(self, inf_str)(Vm)
                taux = getattr(self, tau_str)(Vm)
                alphax = xinf / taux
                betax = 1 / taux - alphax
            rates[alpha_str] = alphax
            rates[beta_str] = betax
        return rates
