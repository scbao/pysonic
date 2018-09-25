#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-25 14:39:19

from enum import Enum
import time
import os
import json
import inspect
import logging
import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.optimize import brentq, curve_fit
from ..utils import *
from ..constants import *
from ..batches import xlslog


# Get package logger
logger = logging.getLogger('PySONIC')


class PmCompMethod(Enum):
    """ Enum: types of computation method for the intermolecular pressure """
    direct = 1
    predict = 2


def LennardJones(x, beta, alpha, C, m, n):
    """ Generic expression of a Lennard-Jones function, adapted for the context of
        symmetric deflection (distance = 2x).

        :param x: deflection (i.e. half-distance)
        :param beta: x-shifting factor
        :param alpha: x-scaling factor
        :param C: y-scaling factor
        :param m: exponent of the repulsion term
        :param n: exponent of the attraction term
        :return: Lennard-Jones potential at given distance (2x)
    """
    return C * (np.power((alpha / (2 * x + beta)), m) - np.power((alpha / (2 * x + beta)), n))


class BilayerSonophore:
    """ This class contains the geometric and mechanical parameters of the
        Bilayer Sonophore Model, as well as all the core functions needed to
        compute the dynamics (kinetics and kinematics) of the bilayer membrane
        cavitation, and run dynamic BLS simulations.
    """

    # BIOMECHANICAL PARAMETERS
    T = 309.15  # Temperature (K)
    Rg = 8.314  # Universal gas constant (Pa.m^3.mol^-1.K^-1)
    delta0 = 2.0e-9  # Thickness of the leaflet (m)
    Delta_ = 1.4e-9  # Initial gap between the two leaflets on a non-charged membrane at equil. (m)
    pDelta = 1.0e5  # Attraction/repulsion pressure coefficient (Pa)
    m = 5.0  # Exponent in the repulsion term (dimensionless)
    n = 3.3  # Exponent in the attraction term (dimensionless)
    rhoL = 1075.0  # Density of the surrounding fluid (kg/m^3)
    muL = 7.0e-4  # Dynamic viscosity of the surrounding fluid (Pa.s)
    muS = 0.035  # Dynamic viscosity of the leaflet (Pa.s)
    kA = 0.24  # Area compression modulus of the leaflet (N/m)
    alpha = 7.56  # Tissue shear loss modulus frequency coefficient (Pa.s)
    C0 = 0.62  # Initial gas molar concentration in the surrounding fluid (mol/m^3)
    kH = 1.613e5  # Henry's constant (Pa.m^3/mol)
    P0 = 1.0e5  # Static pressure in the surrounding fluid (Pa)
    Dgl = 3.68e-9  # Diffusion coefficient of gas in the fluid (m^2/s)
    xi = 0.5e-9  # Boundary layer thickness for gas transport across leaflet (m)
    c = 1515.0  # Speed of sound in medium (m/s)

    # BIOPHYSICAL PARAMETERS
    epsilon0 = 8.854e-12  # Vacuum permittivity (F/m)
    epsilonR = 1.0  # Relative permittivity of intramembrane cavity (dimensionless)

    def __init__(self, a, Cm0, Qm0, Fdrive=None, embedding_depth=0.0):
        """ Constructor of the class.
            :param a: in-plane diameter of the sonophore structure within the membrane (m)
            :param Cm0: membrane resting capacitance (F/m2)
            :param Qm0: membrane resting charge density (C/m2)
            :param Fdrive: frequency of acoustic perturbation (Hz)
            :param embedding_depth: depth of the embedding tissue around the membrane (m)
        """

        # Extract resting constants and geometry
        self.Cm0 = Cm0
        self.Qm0 = Qm0
        self.a = a
        self.d = embedding_depth
        self.S0 = np.pi * self.a**2

        # Derive frequency-dependent tissue elastic modulus
        if Fdrive is not None:
            G_tissue = self.alpha * Fdrive  # G'' (Pa)
            self.kA_tissue = 2 * G_tissue * self.d  # kA of the tissue layer (N/m)
        else:
            self.kA_tissue = 0.

        # Check existence of lookups for derived parameters
        lookups = self.getLookups()
        akey = '{:.1f}'.format(a * 1e9)
        Qkey = '{:.2f}'.format(Qm0 * 1e5)

        # If no lookup, compute parameters and store them in lookup
        if akey not in lookups or Qkey not in lookups[akey]:
            # Find Delta that cancels out Pm + Pec at Z = 0 (m)
            if self.Qm0 == 0.0:
                D_eq = self.Delta_
            else:
                (D_eq, Pnet_eq) = self.findDeltaEq(self.Qm0)
                assert Pnet_eq < PNET_EQ_MAX, 'High Pnet at Z = 0 with ∆ = %.2f nm' % (D_eq * 1e9)
            self.Delta = D_eq

            # Find optimal Lennard-Jones parameters to approximate PMavg
            (LJ_approx, std_err, _) = self.LJfitPMavg()
            assert std_err < PMAVG_STD_ERR_MAX, 'High error in PmAvg nonlinear fit:'\
                ' std_err =  %.2f Pa' % std_err
            self.LJ_approx = LJ_approx

            if akey not in lookups:
                lookups[akey] = {Qkey: {'LJ_approx': LJ_approx, 'Delta_eq': D_eq}}
            else:
                lookups[akey][Qkey] = {'LJ_approx': LJ_approx, 'Delta_eq': D_eq}
            logger.debug('Saving BLS derived parameters to lookup file')
            self.saveLookups(lookups)

        # If lookup exists, load parameters from it
        else:
            logger.debug('Loading BLS derived parameters from lookup file')
            self.LJ_approx = lookups[akey][Qkey]['LJ_approx']
            self.Delta = lookups[akey][Qkey]['Delta_eq']

        # Compute initial volume and gas content
        self.V0 = np.pi * self.Delta * self.a**2
        self.ng0 = self.gasPa2mol(self.P0, self.V0)


    def __repr__(self):
        return 'BilayerSonophore({}m, {}F/cm2, {}C/cm2, embedding_depth={}m'.format(
            si_format([self.a, self.Cm0 * 1e-4, self.Qm0 * 1e-4, self.embedding_depth],
                      precision=1, space=' '))

    def pprint(self):
        return '{}m diameter BilayerSonophore'.format(
            si_format(self.a, precision=0, space=' '))

    def getLookupsPath(self):
        return os.path.join(os.path.split(__file__)[0], 'bls_lookups.json')

    def getLookups(self):
        try:
            with open(self.getLookupsPath()) as fh:
                sample = json.load(fh)
            return sample
        except FileNotFoundError:
            return {}

    def saveLookups(self, lookups):
        with open(self.getLookupsPath(), 'w') as fh:
            json.dump(lookups, fh, indent=2)


    def pparams(self):
        s = '-------- Bilayer Sonophore --------\n'
        s += 'class attributes:\n'
        class_attrs = inspect.getmembers(self.__class__, lambda a: not(inspect.isroutine(a)))
        class_attrs = [a for a in class_attrs if not(a[0].startswith('__') and a[0].endswith('__'))]
        for ca in class_attrs:
            s += '{} = {}\n'.format(ca[0], ca[1])
        s += 'instance attributes:\n'
        inst_attrs = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)))
        inst_attrs = [a for a in inst_attrs if not(a[0].startswith('__') and a[0].endswith('__')) and
                      a not in class_attrs]
        for ia in inst_attrs:
            s += '{} = {}\n'.format(ia[0], ia[1])
        return s

    def reinit(self):

        logger.debug('Re-initializing BLS object')

        # Find Delta that cancels out Pm + Pec at Z = 0 (m)
        if self.Qm0 == 0.0:
            D_eq = self.Delta_
        else:
            (D_eq, Pnet_eq) = self.findDeltaEq(self.Qm0)
            assert Pnet_eq < PNET_EQ_MAX, 'High Pnet at Z = 0 with ∆ = %.2f nm' % (D_eq * 1e9)
        self.Delta = D_eq

        # Compute initial volume and gas content
        self.V0 = np.pi * self.Delta * self.a**2
        self.ng0 = self.gasPa2mol(self.P0, self.V0)


    def curvrad(self, Z):
        """ Return the (signed) instantaneous curvature radius of the leaflet.

            :param Z: leaflet apex outward deflection value (m)
            :return: leaflet curvature radius (m)
        """
        if Z == 0.0:
            return np.inf
        else:
            return (self.a**2 + Z**2) / (2 * Z)


    def v_curvrad(self, Z):
        ''' Vectorized curvrad function '''
        return np.array(list(map(self.curvrad, Z)))


    def surface(self, Z):
        """ Return the surface area of the stretched leaflet (spherical cap).

            :param Z: leaflet apex outward deflection value (m)
            :return: surface of the stretched leaflet (m^2)
        """
        return np.pi * (self.a**2 + Z**2)


    def volume(self, Z):
        """ Return the total volume of the inter-leaflet space (cylinder +/-
            spherical cap).

            :param Z: leaflet apex outward deflection value (m)
            :return: inner volume of the bilayer sonophore structure (m^3)
        """
        return np.pi * self.a**2 * self.Delta\
            * (1 + (Z / (3 * self.Delta) * (3 + Z**2 / self.a**2)))


    def arealstrain(self, Z):
        """ Compute the areal strain of the stretched leaflet.
            epsilon = (S - S0)/S0 = (Z/a)^2

            :param Z: leaflet apex outward deflection value (m)
            :return: areal strain (dimensionless)
        """
        return (Z / self.a)**2


    def Capct(self, Z):
        """ Compute the membrane capacitance per unit area,
            under the assumption of parallel-plate capacitor
            with average inter-layer distance.

            :param Z: leaflet apex outward deflection value (m)
            :return: capacitance per unit area (F/m2)
        """
        if Z == 0.0:
            return self.Cm0
        else:
            return ((self.Cm0 * self.Delta / self.a**2) *
                    (Z + (self.a**2 - Z**2 - Z * self.Delta) / (2 * Z) *
                     np.log((2 * Z + self.Delta) / self.Delta)))


    def v_Capct(self, Z):
        ''' Vectorized Capct function '''
        return np.array(list(map(self.Capct, Z)))


    def derCapct(self, Z, U):
        """ Compute the derivative of the membrane capacitance per unit area
            with respect to time, under the assumption of parallel-plate capacitor.

            :param Z: leaflet apex outward deflection value (m)
            :param U: leaflet apex outward deflection velocity (m/s)
            :return: derivative of capacitance per unit area (F/m2.s)
        """
        dCmdZ = ((self.Cm0 * self.Delta / self.a**2) *
                 ((Z**2 + self.a**2) / (Z * (2 * Z + self.Delta)) -
                  ((Z**2 + self.a**2) *
                   np.log((2 * Z + self.Delta) / self.Delta)) / (2 * Z**2)))
        return dCmdZ * U


    def localdef(self, r, Z, R):
        """ Compute the (signed) local transverse leaflet deviation at a distance
            r from the center of the dome.

            :param r: in-plane distance from center of the sonophore (m)
            :param Z: leaflet apex outward deflection value (m)
            :param R: leaflet curvature radius (m)
            :return: local transverse leaflet deviation (m)
        """
        if np.abs(Z) == 0.0:
            return 0.0
        else:
            return np.sign(Z) * (np.sqrt(R**2 - r**2) - np.abs(R) + np.abs(Z))


    def Pacoustic(self, t, Adrive, Fdrive, phi=np.pi):
        """ Compute the acoustic pressure at a specific time, given
            the amplitude, frequency and phase of the acoustic stimulus.

            :param t: time of interest
            :param Adrive: acoustic drive amplitude (Pa)
            :param Fdrive: acoustic drive frequency (Hz)
            :param phi: acoustic drive phase (rad)
        """
        return Adrive * np.sin(2 * np.pi * Fdrive * t - phi)


    def PMlocal(self, r, Z, R):
        """ Compute the local intermolecular pressure.

            :param r: in-plane distance from center of the sonophore (m)
            :param Z: leaflet apex outward deflection value (m)
            :param R: leaflet curvature radius (m)
            :return: local intermolecular pressure (Pa)
        """
        z = self.localdef(r, Z, R)
        relgap = (2 * z + self.Delta) / self.Delta_
        return self.pDelta * ((1 / relgap)**self.m - (1 / relgap)**self.n)


    def PMavg(self, Z, R, S):
        """ Compute the average intermolecular pressure felt across the leaflet
            by quadratic integration.

            :param Z: leaflet apex outward deflection value (m)
            :param R: leaflet curvature radius (m)
            :param S: surface of the stretched leaflet (m^2)
            :return: averaged intermolecular resultant pressure across the leaflet (Pa)

            .. warning:: quadratic integration is computationally expensive.
        """

        # Integrate intermolecular force over an infinitely thin ring of radius r from 0 to a
        fTotal, _ = integrate.quad(lambda r, Z, R: 2 * np.pi * r * self.PMlocal(r, Z, R),
                                   0, self.a, args=(Z, R))
        return fTotal / S


    def v_PMavg(self, Z, R, S):
        ''' Vectorized PMavg function '''
        return np.array(list(map(self.PMavg, Z, R, S)))


    def LJfitPMavg(self):
        """ Determine optimal parameters of a Lennard-Jones expression
            approximating the average intermolecular pressure.

            These parameters are obtained by a nonlinear fit of the
            Lennard-Jones function for a range of deflection values
            between predetermined Zmin and Zmax.

            :return: 3-tuple with optimized LJ parameters for PmAvg prediction (Map) and
                the standard and max errors of the prediction in the fitting range (in Pascals)
        """

        # Determine lower bound of deflection range: when Pm = Pmmax
        PMmax = LJFIT_PM_MAX  # Pa
        Zminlb = -0.49 * self.Delta
        Zminub = 0.0
        Zmin = brentq(lambda Z, Pmmax: self.PMavg(Z, self.curvrad(Z), self.surface(Z)) - PMmax,
                      Zminlb, Zminub, args=(PMmax), xtol=1e-16)

        # Create vectors for geometric variables
        Zmax = 2 * self.a
        Z = np.arange(Zmin, Zmax, 1e-11)
        Pmavg = self.v_PMavg(Z, self.v_curvrad(Z), self.surface(Z))

        # Compute optimal nonlinear fit of custom LJ function with initial guess
        x0_guess = self.delta0
        C_guess = 0.1 * self.pDelta
        nrep_guess = self.m
        nattr_guess = self.n
        pguess = (x0_guess, C_guess, nrep_guess, nattr_guess)
        popt, _ = curve_fit(lambda x, x0, C, nrep, nattr:
                            LennardJones(x, self.Delta, x0, C, nrep, nattr),
                            Z, Pmavg, p0=pguess, maxfev=10000)
        (x0_opt, C_opt, nrep_opt, nattr_opt) = popt
        Pmavg_fit = LennardJones(Z, self.Delta, x0_opt, C_opt, nrep_opt, nattr_opt)

        # Compute prediction error
        residuals = Pmavg - Pmavg_fit
        ss_res = np.sum(residuals**2)
        N = residuals.size
        std_err = np.sqrt(ss_res / N)
        max_err = max(np.abs(residuals))

        logger.debug('LJ approx: x0 = %.2f nm, C = %.2f kPa, m = %.2f, n = %.2f',
                     x0_opt * 1e9, C_opt * 1e-3, nrep_opt, nattr_opt)

        LJ_approx = {"x0": x0_opt, "C": C_opt, "nrep": nrep_opt, "nattr": nattr_opt}
        return (LJ_approx, std_err, max_err)


    def PMavgpred(self, Z):
        """ Return the predicted intermolecular pressure based on a specific Lennard-Jones
            function fitted on the deflection physiological range.

            :param Z: leaflet apex outward deflection value (m)
            :return: predicted average intermolecular pressure (Pa)
        """
        return LennardJones(Z, self.Delta, self.LJ_approx['x0'], self.LJ_approx['C'],
                            self.LJ_approx['nrep'], self.LJ_approx['nattr'])


    def Pelec(self, Z, Qm):
        """ Compute the electric equivalent pressure term.

            :param Z: leaflet apex outward deflection value (m)
            :param Qm: membrane charge density (C/m2)
            :return: electric equivalent pressure (Pa)
        """
        relS = self.S0 / self.surface(Z)
        abs_perm = self.epsilon0 * self.epsilonR  # F/m
        return -relS * Qm**2 / (2 * abs_perm)  # Pa


    def findDeltaEq(self, Qm):
        """ Compute the Delta that cancels out the (Pm + Pec) equation at Z = 0
            for a given membrane charge density, using the Brent method to refine
            the pressure root iteratively.

            :param Qm: membrane charge density (C/m2)
            :return: equilibrium value (m) and associated pressure (Pa)
        """

        f = lambda Delta: (self.pDelta *
                           ((self.Delta_ / Delta)**self.m - (self.Delta_ / Delta)**self.n) +
                           self.Pelec(0.0, Qm))

        Delta_lb = 0.1 * self.Delta_
        Delta_ub = 2.0 * self.Delta_

        Delta_eq = brentq(f, Delta_lb, Delta_ub, xtol=1e-16)
        logger.debug('∆eq = %.2f nm', Delta_eq * 1e9)
        return (Delta_eq, f(Delta_eq))


    def gasflux(self, Z, P):
        """ Compute the gas molar flux through the BLS boundary layer for
            an unsteady system.

            :param Z: leaflet apex outward deflection value (m)
            :param P: internal gas pressure in the inter-leaflet space (Pa)
            :return: gas molar flux (mol/s)
        """
        dC = self.C0 - P / self.kH
        return 2 * self.surface(Z) * self.Dgl * dC / self.xi


    def gasmol2Pa(self, ng, V):
        """ Compute the gas pressure in the inter-leaflet space for an
            unsteady system, from the value of gas molar content.

            :param ng: internal molar content (mol)
            :param V: inner volume of the bilayer sonophore structure (m^3)
            :return: internal gas pressure (Pa)
        """
        return ng * self.Rg * self.T / V


    def gasPa2mol(self, P, V):
        """ Compute the gas molar content in the inter-leaflet space for
            an unsteady system, from the value of internal gas pressure.

            :param P: internal gas pressure in the inter-leaflet space (Pa)
            :param V: inner volume of the bilayer sonophore structure (m^3)
            :return: internal gas molar content (mol)
        """
        return P * V / (self.Rg * self.T)


    def PtotQS(self, Z, ng, Qm, Pac, Pm_comp_method):
        """ Compute the balance pressure of the quasi-steady system, upon application
            of an external perturbation on a charged membrane:
            Ptot = Pm + Pg + Pec - P0 - Pac.

            :param Z: leaflet apex outward deflection value (m)
            :param ng: internal molar content (mol)
            :param Qm: membrane charge density (C/m2)
            :param Pac: external acoustic perturbation (Pa)
            :param Pm_comp_method: type of method used to compute average intermolecular pressure
            :return: total balance pressure (Pa)

        """
        if Pm_comp_method is PmCompMethod.direct:
            Pm = self.PMavg(Z, self.curvrad(Z), self.surface(Z))
        elif Pm_comp_method is PmCompMethod.predict:
            Pm = self.PMavgpred(Z)
        return Pm + self.gasmol2Pa(ng, self.volume(Z)) - self.P0 - Pac + self.Pelec(Z, Qm)


    def balancedefQS(self, ng, Qm, Pac=0.0, Pm_comp_method=PmCompMethod.predict):
        """ Compute the leaflet deflection upon application of an external
            perturbation to a quasi-steady system with a charged membrane.

            This function uses the Brent method (progressive approximation of
            function root) to solve the following transcendental equation for Z:
            Pm + Pg + Pec - P0 - Pac = 0.

            :param ng: internal molar content (mol)
            :param Qm: membrane charge density (C/m2)
            :param Pac: external acoustic perturbation (Pa)
            :param Pm_comp_method: type of method used to compute average intermolecular pressure
            :return: leaflet deflection (Z) canceling out the balance equation
        """
        lb = -0.49 * self.Delta
        ub = self.a
        Plb = self.PtotQS(lb, ng, Qm, Pac, Pm_comp_method)
        Pub = self.PtotQS(ub, ng, Qm, Pac, Pm_comp_method)
        assert (Plb > 0 > Pub), '[%d, %d] is not a sign changing interval for PtotQS' % (lb, ub)
        return brentq(self.PtotQS, lb, ub, args=(ng, Qm, Pac, Pm_comp_method), xtol=1e-16)


    def TEleaflet(self, Z):
        """ Compute the circumferential elastic tension felt across the
            entire leaflet upon stretching.

            :param Z: leaflet apex outward deflection value (m)
            :return: circumferential elastic tension (N/m)
        """
        return self.kA * self.arealstrain(Z)


    def TEtissue(self, Z):
        """ Compute the circumferential elastic tension felt across the
            embedding viscoelastic tissue layer upon stretching.

            :param Z: leaflet apex outward deflection value (m)
            :return: circumferential elastic tension (N/m)
        """
        return self.kA_tissue * self.arealstrain(Z)


    def TEtot(self, Z):
        """ Compute the total circumferential elastic tension (leaflet
            and embedding tissue) felt upon stretching.

            :param Z: leaflet apex outward deflection value (m)
            :return: circumferential elastic tension (N/m)
        """
        return self.TEleaflet(Z) + self.TEtissue(Z)


    def PEtot(self, Z, R):
        """ Compute the total elastic tension pressure (leaflet + embedding
            tissue) felt upon stretching.

            :param Z: leaflet apex outward deflection value (m)
            :param R: leaflet curvature radius (m)
            :return: elastic tension pressure (Pa)
        """
        return - self.TEtot(Z) / R


    def PVleaflet(self, U, R):
        """ Compute the viscous stress felt across the entire leaflet
            upon stretching.

            :param U: leaflet apex outward deflection velocity (m/s)
            :param R: leaflet curvature radius (m)
            :return: leaflet viscous stress (Pa)
        """
        return - 12 * U * self.delta0 * self.muS / R**2


    def PVfluid(self, U, R):
        """ Compute the viscous stress felt across the entire fluid
            upon stretching.

            :param U: leaflet apex outward deflection velocity (m/s)
            :param R: leaflet curvature radius (m)
            :return: fluid viscous stress (Pa)
        """
        return - 4 * U * self.muL / np.abs(R)


    def accP(self, Pres, R):
        """ Compute the pressure-driven acceleration of the leaflet in the
            unsteady system, upon application of an external perturbation.

            :param Pres: net resultant pressure (Pa)
            :param R: leaflet curvature radius (m)
            :return: pressure-driven acceleration (m/s^2)
        """
        return Pres / (self.rhoL * np.abs(R))


    def accNL(self, U, R):
        """ Compute the non-linear term of the leaflet acceleration in the
        unsteady system, upon application of an external perturbation.

        :param U: leaflet apex outward deflection velocity (m/s)
        :param R: leaflet curvature radius (m)
        :return: nonlinear acceleration (m/s^2)

        .. note:: A simplified version of nonlinear acceleration (neglecting
            dR/dH) is used here.

        """
        # return - (3/2 - 2*R/H) * U**2 / R
        return -(3 * U**2) / (2 * R)


    def derivatives(self, y, t, Adrive, Fdrive, Qm, phi, Pm_comp_method=PmCompMethod.predict):
        """ Compute the derivatives of the 3-ODE mechanical system variables,
            with an imposed constant charge density.

            :param y: vector of HH system variables at time t
            :param t: specific instant in time (s)
            :param Adrive: acoustic drive amplitude (Pa)
            :param Fdrive: acoustic drive frequency (Hz)
            :param Qm: membrane charge density (F/m2)
            :param phi: acoustic drive phase (rad)
            :param Pm_comp_method: type of method used to compute average intermolecular pressure
            :return: vector of mechanical system derivatives at time t
        """

        # Split input vector explicitly
        (U, Z, ng) = y

        # Correct deflection value is below critical compression
        if Z < -0.5 * self.Delta:
            logger.warning('Deflection out of range: Z = %.2f nm', Z * 1e9)
            Z = -0.49 * self.Delta

        # Compute curvature radius
        R = self.curvrad(Z)

        # Compute total pressure
        Pg = self.gasmol2Pa(ng, self.volume(Z))
        if Pm_comp_method is PmCompMethod.direct:
            Pm = self.PMavg(Z, self.curvrad(Z), self.surface(Z))
        elif Pm_comp_method is PmCompMethod.predict:
            Pm = self.PMavgpred(Z)
        Ptot = (Pm + Pg - self.P0 - self.Pacoustic(t, Adrive, Fdrive, phi) +
                self.PEtot(Z, R) + self.PVleaflet(U, R) + self.PVfluid(U, R) + self.Pelec(Z, Qm))

        # Compute derivatives
        dUdt = self.accP(Ptot, R) + self.accNL(U, R)
        dZdt = U
        dngdt = self.gasflux(Z, Pg)

        # Return derivatives vector
        return [dUdt, dZdt, dngdt]


    def checkInputs(self, Fdrive, Adrive, Qm, phi):
        ''' Check validity of stimulation parameters.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param phi: acoustic drive phase (rad)
            :param Qm: imposed membrane charge density (C/m2)
        '''
        if not all(isinstance(param, float) for param in [Fdrive, Adrive, Qm, phi]):
            raise TypeError('Invalid stimulation parameters (must be float typed)')
        if Fdrive <= 0:
            raise ValueError('Invalid US driving frequency: {} kHz (must be strictly positive)'
                             .format(Fdrive * 1e-3))
        if Adrive < 0:
            raise ValueError('Invalid US pressure amplitude: {} kPa (must be positive or null)'
                             .format(Adrive * 1e-3))
        if Qm < CHARGE_RANGE[0] or Qm > CHARGE_RANGE[1]:
            raise ValueError('Invalid applied charge: {} nC/cm2 (must be within [{}, {}] interval'
                             .format(Qm * 1e5, CHARGE_RANGE[0] * 1e5, CHARGE_RANGE[1] * 1e5))
        if phi < 0 or phi >= 2 * np.pi:
            raise ValueError('Invalid US pressure phase: {:.2f} rad (must be within [0, 2 PI[ rad'
                             .format(phi))


    def simulate(self, Fdrive, Adrive, Qm, phi=np.pi, Pm_comp_method=PmCompMethod.predict):
        """ Compute short solutions of the mechanical system for specific
            US stimulation parameters and with an imposed membrane charge density.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param phi: acoustic drive phase (rad)
            :param Qm: imposed membrane charge density (C/m2)
            :param Pm_comp_method: type of method used to compute average intermolecular pressure
            :return: 3-tuple with the time profile, the solution matrix and a state vector
        """

        # Check validity of stimulation parameters
        self.checkInputs(Fdrive, Adrive, Qm, phi)

        # Determine mechanical system time step
        Tdrive = 1 / Fdrive
        dt_mech = Tdrive / NPC_FULL
        t_mech_cycle = np.linspace(0, Tdrive - dt_mech, NPC_FULL)

        # Initialize system variables
        t0 = 0.0
        Z0 = 0.0
        U0 = 0.0
        ng0 = self.ng0

        # Solve quasi-steady equation to compute first deflection value
        Pac1 = self.Pacoustic(t0 + dt_mech, Adrive, Fdrive, phi)
        Z1 = self.balancedefQS(ng0, Qm, Pac1, Pm_comp_method)
        U1 = (Z1 - Z0) / dt_mech

        # Construct arrays to hold system variables
        states = np.array([1, 1])
        t = np.array([t0, t0 + dt_mech])
        y = np.array([[U0, U1], [Z0, Z1], [ng0, ng0]])

        # Integrate mechanical system for a few acoustic cycles until stabilization
        j = 0
        ng_last = None
        Z_last = None
        periodic_conv = False

        while not periodic_conv and j < NCYCLES_MAX:
            t_mech = t_mech_cycle + t[-1] + dt_mech
            y_mech = integrate.odeint(self.derivatives, y[:, -1], t_mech,
                                      args=(Adrive, Fdrive, Qm, phi, Pm_comp_method)).T

            # Compare Z and ng signals over the last 2 acoustic periods
            if j > 0:
                Z_rmse = rmse(Z_last, y_mech[1, :])
                ng_rmse = rmse(ng_last, y_mech[2, :])
                logger.debug('step %u: Z_rmse = %.2e m, ng_rmse = %.2e mol', j, Z_rmse, ng_rmse)
                if Z_rmse < Z_ERR_MAX and ng_rmse < NG_ERR_MAX:
                    periodic_conv = True

            # Update last vectors for next comparison
            Z_last = y_mech[1, :]
            ng_last = y_mech[2, :]

            # Concatenate time and solutions to global vectors
            states = np.concatenate([states, np.ones(NPC_FULL)], axis=0)
            t = np.concatenate([t, t_mech], axis=0)
            y = np.concatenate([y, y_mech], axis=1)

            # Increment loop index
            j += 1

        if j == NCYCLES_MAX:
            logger.warning('No convergence: stopping after %u cycles', j)
        else:
            logger.debug('Periodic convergence after %u cycles', j)

        states[-1] = 0

        # return output variables
        return (t, y[1:, :], states)

    def runAndSave(self, outdir, Fdrive, Adrive, Qm):
        ''' Run a simulation of the mechanical system with specific stimulation parameters
            and an imposed value of charge density, and save the results in a PKL file.

            :param outdir: full path to output directory
            :param Fdrive: US frequency (Hz)
            :param Adrive: acoustic pressure amplitude (Pa)
            :param Qm: applided membrane charge density (C/m2)
        '''

        # Get date and time info
        date_str = time.strftime("%Y.%m.%d")
        daytime_str = time.strftime("%H:%M:%S")

        logger.info('%s: simulation @ f = %sHz, A = %sPa, Q = %sC/cm2',
                    self.pprint(), *si_format([Fdrive, Adrive, Qm * 1e-4], 2, space=' '))

        # Run simulation
        tstart = time.time()
        (t, y, states) = self.simulate(Fdrive, Adrive, Qm)
        (Z, ng) = y
        tcomp = time.time() - tstart
        logger.debug('completed in %s', si_format(tcomp, 1))
        U = np.insert(np.diff(Z) / np.diff(t), 0, 0.0)

        # Store dataframe and metadata
        df = pd.DataFrame({'t': t, 'states': states, 'U': U, 'Z': Z, 'ng': ng})
        meta = {'a': self.a, 'd': self.d, 'Cm0': self.Cm0, 'Qm0': self.Qm0,
                'Fdrive': Fdrive, 'Adrive': Adrive, 'phi': np.pi, 'Qm': Qm,
                'tcomp': tcomp}

        # Export into to PKL file
        simcode = 'MECH_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.1f}nCcm2'.format(
            self.a * 1e9, Fdrive * 1e-3, Adrive * 1e-3, Qm * 1e5)
        outpath = '{}/{}.pkl'.format(outdir, simcode)
        with open(outpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': df}, fh)
        logger.debug('simulation data exported to "%s"', outpath)

        # Compute key output metrics
        Zmax = np.amax(Z)
        Zmin = np.amin(Z)
        Zabs_max = np.amax(np.abs([Zmin, Zmax]))
        eAmax = self.arealstrain(Zabs_max)
        Tmax = self.TEtot(Zabs_max)
        Pmmax = self.PMavgpred(Zmin)
        ngmax = np.amax(ng)
        dUdtmax = np.amax(np.abs(np.diff(U) / np.diff(t)**2))

        # Export key metrics to log file
        logpath = os.path.join(outdir, 'log_MECH.xlsx')
        logentry = {
            'Date': date_str,
            'Time': daytime_str,
            'Radius (nm)': self.a * 1e9,
            'Thickness (um)': self.d * 1e6,
            'Fdrive (kHz)': Fdrive * 1e-3,
            'Adrive (kPa)': Adrive * 1e-3,
            'Charge (nC/cm2)': Qm * 1e5,
            '# samples': t.size,
            'Comp. time (s)': round(tcomp, 2),
            'kA total (N/m)': self.kA + self.kA_tissue,
            'Max Z (nm)': Zmax * 1e9,
            'Max eA (-)': eAmax,
            'Max Te (mN/m)': Tmax * 1e3,
            'Max rel. ng increase (-)': (ngmax - self.ng0) / self.ng0,
            'Max Pm (kPa)': Pmmax * 1e-3,
            'Max acc. (m/s2)': dUdtmax
        }
        if xlslog(logpath, logentry) == 1:
            logger.debug('log exported to "%s"', logpath)
        else:
            logger.error('log export to "%s" aborted', logpath)

        return outpath

    def getCycleProfiles(self, Fdrive, Adrive, Qm):
        ''' Run a mechanical simulation until periodic stabilization, and compute pressure profiles
            over the last acoustic cycle.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param Qm: imposed membrane charge density (C/m2)
            :return: dataframe with the time, kinematic and pressure profiles over the last cycle.
        '''

        # Run default simulation and compute relevant profiles
        logger.info('Running mechanical simulation (a = %sm, f = %sHz, A = %sPa)',
                    si_format(self.a, 1), si_format(Fdrive, 1), si_format(Adrive, 1))
        t, y, _ = self.simulate(Fdrive, Adrive, Qm, Pm_comp_method=PmCompMethod.direct)
        dt = (t[-1] - t[0]) / (t.size - 1)
        Z, ng = y[:, -NPC_FULL:]
        t = t[-NPC_FULL:]
        t -= t[0]

        logger.info('Computing pressure cyclic profiles')
        R = self.v_curvrad(Z)
        U = np.diff(Z) / dt
        U = np.hstack((U, U[-1]))
        data = {
            't': t,
            'Z': Z,
            'Cm': self.v_Capct(Z),
            'P_M': self.v_PMavg(Z, R, self.surface(Z)),
            'P_Q': self.Pelec(Z, Qm),
            'P_{VE}': self.PEtot(Z, R) + self.PVleaflet(U, R),
            'P_V': self.PVfluid(U, R),
            'P_G': self.gasmol2Pa(ng, self.volume(Z)),
            'P_0': - np.ones(Z.size) * self.P0
        }
        return pd.DataFrame(data, columns=data.keys())
