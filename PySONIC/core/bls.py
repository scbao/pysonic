#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-31 14:52:30

from enum import Enum
import os
import json
import inspect
import pickle
import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.optimize import brentq, curve_fit
from ..utils import logger, si_format
from ..constants import *
from .simulators import PeriodicSimulator


class PmCompMethod(Enum):
    ''' Enum: types of computation method for the intermolecular pressure '''
    direct = 1
    predict = 2


def LennardJones(x, beta, alpha, C, m, n):
    ''' Generic expression of a Lennard-Jones function, adapted for the context of
        symmetric deflection (distance = 2x).

        :param x: deflection (i.e. half-distance)
        :param beta: x-shifting factor
        :param alpha: x-scaling factor
        :param C: y-scaling factor
        :param m: exponent of the repulsion term
        :param n: exponent of the attraction term
        :return: Lennard-Jones potential at given distance (2x)
    '''
    return C * (np.power((alpha / (2 * x + beta)), m) - np.power((alpha / (2 * x + beta)), n))


class BilayerSonophore:
    ''' This class contains the geometric and mechanical parameters of the
        Bilayer Sonophore Model, as well as all the core functions needed to
        compute the dynamics (kinetics and kinematics) of the bilayer membrane
        cavitation, and run dynamic BLS simulations.
    '''

    # BIOMECHANICAL PARAMETERS
    T = 309.15  # Temperature (K)
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

    tscale = 'us'  # relevant temporal scale of the model

    def __init__(self, a, Cm0, Qm0, Fdrive=None, embedding_depth=0.0):
        ''' Constructor of the class.
            :param a: in-plane radius of the sonophore structure within the membrane (m)
            :param Cm0: membrane resting capacitance (F/m2)
            :param Qm0: membrane resting charge density (C/m2)
            :param Fdrive: frequency of acoustic perturbation (Hz)
            :param embedding_depth: depth of the embedding tissue around the membrane (m)
        '''

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
        return '{}m radius BilayerSonophore'.format(
            si_format(self.a, precision=0, space=' '))

    def filecode(self, Fdrive, Adrive, Qm):
        return 'MECH_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.1f}nCcm2'.format(
            self.a * 1e9, Fdrive * 1e-3, Adrive * 1e-3, Qm * 1e5)

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
        inst_attrs = [
            a for a in inst_attrs if not(a[0].startswith('__') and a[0].endswith('__')) and
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

    def getPltScheme(self):
        return {
            'P_{AC}': ['Pac'],
            'Z': ['Z'],
            'n_g': ['ng']
        }

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        ''' Return a dictionary with information about all plot variables related to the model. '''
        return {
            'Pac': {
                'desc': 'acoustic pressure',
                'label': 'P_{AC}',
                'unit': 'kPa',
                'factor': 1e-3,
                'func': 'Pacoustic({0}t{1}, meta["Adrive"] * {0}stimstate{1}, meta["Fdrive"])'.format(
                    wrapleft, wrapright)
            },

            'Z': {
                'desc': 'leaflets deflection',
                'label': 'Z',
                'unit': 'nm',
                'factor': 1e9,
                'bounds': (-1.0, 10.0)
            },

            'ng': {
                'desc': 'gas content',
                'label': 'n_g',
                'unit': '10^{-22}\ mol',
                'factor': 1e22,
                'bounds': (1.0, 15.0)
            },

            'Pmavg': {
                'desc': 'average intermolecular pressure',
                'label': 'P_M',
                'unit': 'kPa',
                'factor': 1e-3,
                'func': 'PMavgpred({0}Z{1})'.format(wrapleft, wrapright)
            },

            'Telastic': {
                'desc': 'leaflet elastic tension',
                'label': 'T_E',
                'unit': 'mN/m',
                'factor': 1e3,
                'func': 'TEleaflet({0}Z{1})'.format(wrapleft, wrapright)
            },

            'Cm': {
                'desc': 'membrane capacitance',
                'label': 'C_m',
                'unit': 'uF/cm^2',
                'factor': 1e2,
                'bounds': (0.0, 1.5),
                'func': 'v_Capct({0}Z{1})'.format(wrapleft, wrapright)
            }
        }

    def curvrad(self, Z):
        ''' Leaflet curvature radius
            (signed variable)

            :param Z: leaflet apex deflection (m)
            :return: leaflet curvature radius (m)
        '''
        if Z == 0.0:
            return np.inf
        else:
            return (self.a**2 + Z**2) / (2 * Z)

    def v_curvrad(self, Z):
        ''' Vectorized curvrad function '''
        return np.array(list(map(self.curvrad, Z)))

    def surface(self, Z):
        ''' Surface area of the stretched leaflet
            (spherical cap formula)

            :param Z: leaflet apex deflection (m)
            :return: stretched leaflet surface (m^2)
        '''
        return np.pi * (self.a**2 + Z**2)

    def volume(self, Z):
        ''' Volume of the inter-leaflet space
            (cylinder +/- 2 spherical caps)

            :param Z: leaflet apex deflection (m)
            :return: bilayer sonophore inner volume (m^3)
        '''
        return np.pi * self.a**2 * self.Delta\
            * (1 + (Z / (3 * self.Delta) * (3 + Z**2 / self.a**2)))

    def arealstrain(self, Z):
        ''' Areal strain of the stretched leaflet
            epsilon = (S - S0)/S0 = (Z/a)^2

            :param Z: leaflet apex deflection (m)
            :return: areal strain (dimensionless)
        '''
        return (Z / self.a)**2

    def Capct(self, Z):
        ''' Membrane capacitance
            (parallel-plate capacitor evaluated at average inter-layer distance)

            :param Z: leaflet apex deflection (m)
            :return: capacitance per unit area (F/m2)
        '''
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
        ''' Evolution of membrane capacitance

            :param Z: leaflet apex deflection (m)
            :param U: leaflet apex deflection velocity (m/s)
            :return: time derivative of capacitance per unit area (F/m2.s)
        '''
        dCmdZ = ((self.Cm0 * self.Delta / self.a**2) *
                 ((Z**2 + self.a**2) / (Z * (2 * Z + self.Delta)) -
                  ((Z**2 + self.a**2) *
                   np.log((2 * Z + self.Delta) / self.Delta)) / (2 * Z**2)))
        return dCmdZ * U

    def localdef(self, r, Z, R):
        ''' Local leaflet deflection at specific radial distance
            (signed)

            :param r: in-plane distance from center of the sonophore (m)
            :param Z: leaflet apex deflection (m)
            :param R: leaflet curvature radius (m)
            :return: local transverse leaflet deviation (m)
        '''
        if np.abs(Z) == 0.0:
            return 0.0
        else:
            return np.sign(Z) * (np.sqrt(R**2 - r**2) - np.abs(R) + np.abs(Z))

    def Pacoustic(self, t, Adrive, Fdrive, phi=np.pi):
        ''' Time-varying acoustic pressure

            :param t: time (s)
            :param Adrive: acoustic drive amplitude (Pa)
            :param Fdrive: acoustic drive frequency (Hz)
            :param phi: acoustic drive phase (rad)
        '''
        return Adrive * np.sin(2 * np.pi * Fdrive * t - phi)

    def PMlocal(self, r, Z, R):
        ''' Local intermolecular pressure

            :param r: in-plane distance from center of the sonophore (m)
            :param Z: leaflet apex deflection (m)
            :param R: leaflet curvature radius (m)
            :return: local intermolecular pressure (Pa)
        '''
        z = self.localdef(r, Z, R)
        relgap = (2 * z + self.Delta) / self.Delta_
        return self.pDelta * ((1 / relgap)**self.m - (1 / relgap)**self.n)

    def PMavg(self, Z, R, S):
        ''' Average intermolecular pressure across the leaflet
            (computed by quadratic integration)

            :param Z: leaflet apex outward deflection value (m)
            :param R: leaflet curvature radius (m)
            :param S: surface of the stretched leaflet (m^2)
            :return: averaged intermolecular resultant pressure (Pa)

            .. warning:: quadratic integration is computationally expensive.
        '''
        # Integrate intermolecular force over an infinitely thin ring of radius r from 0 to a
        fTotal, _ = integrate.quad(lambda r, Z, R: 2 * np.pi * r * self.PMlocal(r, Z, R),
                                   0, self.a, args=(Z, R))
        return fTotal / S

    def v_PMavg(self, Z, R, S):
        ''' Vectorized PMavg function '''
        return np.array(list(map(self.PMavg, Z, R, S)))

    def LJfitPMavg(self):
        ''' Determine optimal parameters of a Lennard-Jones expression
            approximating the average intermolecular pressure.

            These parameters are obtained by a nonlinear fit of the
            Lennard-Jones function for a range of deflection values
            between predetermined Zmin and Zmax.

            :return: 3-tuple with optimized LJ parameters for PmAvg prediction (Map) and
                the standard and max errors of the prediction in the fitting range (in Pascals)
        '''

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
        ''' Approximated average intermolecular pressure
            (using nonlinearly fitted Lennard-Jones function)

            :param Z: leaflet apex deflection (m)
            :return: predicted average intermolecular pressure (Pa)
        '''
        return LennardJones(Z, self.Delta, self.LJ_approx['x0'], self.LJ_approx['C'],
                            self.LJ_approx['nrep'], self.LJ_approx['nattr'])

    def Pelec(self, Z, Qm):
        ''' Electrical pressure term

            :param Z: leaflet apex deflection (m)
            :param Qm: membrane charge density (C/m2)
            :return: electrical pressure (Pa)
        '''
        relS = self.S0 / self.surface(Z)
        abs_perm = self.epsilon0 * self.epsilonR  # F/m
        return - relS * Qm**2 / (2 * abs_perm)  # Pa

    def findDeltaEq(self, Qm):
        ''' Compute the Delta that cancels out the (Pm + Pec) equation at Z = 0
            for a given membrane charge density, using the Brent method to refine
            the pressure root iteratively.

            :param Qm: membrane charge density (C/m2)
            :return: equilibrium value (m) and associated pressure (Pa)
        '''
        f = lambda Delta: (self.pDelta * (
            (self.Delta_ / Delta)**self.m - (self.Delta_ / Delta)**self.n) + self.Pelec(0.0, Qm))
        Delta_eq = brentq(f, 0.1 * self.Delta_, 2.0 * self.Delta_, xtol=1e-16)
        logger.debug('∆eq = %.2f nm', Delta_eq * 1e9)
        return (Delta_eq, f(Delta_eq))

    def gasFlux(self, Z, P):
        ''' Gas molar flux through the sonophore boundary layers

            :param Z: leaflet apex deflection (m)
            :param P: internal gas pressure (Pa)
            :return: gas molar flux (mol/s)
        '''
        dC = self.C0 - P / self.kH
        return 2 * self.surface(Z) * self.Dgl * dC / self.xi

    def gasmol2Pa(self, ng, V):
        ''' Internal gas pressure for a given molar content

            :param ng: internal molar content (mol)
            :param V: sonophore inner volume (m^3)
            :return: internal gas pressure (Pa)
        '''
        return ng * Rg * self.T / V

    def gasPa2mol(self, P, V):
        ''' Internal gas molar content for a given pressure

            :param P: internal gas pressure (Pa)
            :param V: sonophore inner volume (m^3)
            :return: internal gas molar content (mol)
        '''
        return P * V / (Rg * self.T)

    def PtotQS(self, Z, ng, Qm, Pac, Pm_comp_method):
        ''' Net quasi-steady pressure for a given acoustic pressure
            (Ptot = Pm + Pg + Pec - P0 - Pac)

            :param Z: leaflet apex deflection (m)
            :param ng: internal molar content (mol)
            :param Qm: membrane charge density (C/m2)
            :param Pac: acoustic pressure (Pa)
            :param Pm_comp_method: computation method for average intermolecular pressure
            :return: total balance pressure (Pa)
        '''
        if Pm_comp_method is PmCompMethod.direct:
            Pm = self.PMavg(Z, self.curvrad(Z), self.surface(Z))
        elif Pm_comp_method is PmCompMethod.predict:
            Pm = self.PMavgpred(Z)
        return Pm + self.gasmol2Pa(ng, self.volume(Z)) - self.P0 - Pac + self.Pelec(Z, Qm)

    def balancedefQS(self, ng, Qm, Pac=0.0, Pm_comp_method=PmCompMethod.predict):
        ''' Quasi-steady equilibrium deflection for a given acoustic pressure
            (computed by approximating the root of quasi-steady pressure)

            :param ng: internal molar content (mol)
            :param Qm: membrane charge density (C/m2)
            :param Pac: external acoustic perturbation (Pa)
            :param Pm_comp_method: computation method for average intermolecular pressure
            :return: leaflet deflection canceling quasi-steady pressure (m)
        '''
        lb = -0.49 * self.Delta
        ub = self.a
        Plb = self.PtotQS(lb, ng, Qm, Pac, Pm_comp_method)
        Pub = self.PtotQS(ub, ng, Qm, Pac, Pm_comp_method)
        assert (Plb > 0 > Pub), '[%d, %d] is not a sign changing interval for PtotQS' % (lb, ub)
        return brentq(self.PtotQS, lb, ub, args=(ng, Qm, Pac, Pm_comp_method), xtol=1e-16)

    def TEleaflet(self, Z):
        ''' Elastic tension in leaflet

            :param Z: leaflet apex deflection (m)
            :return: circumferential elastic tension (N/m)
        '''
        return self.kA * self.arealstrain(Z)

    def TEtissue(self, Z):
        ''' Elastic tension in surrounding viscoelastic layer

            :param Z: leaflet apex deflection (m)
            :return: circumferential elastic tension (N/m)
        '''
        return self.kA_tissue * self.arealstrain(Z)

    def TEtot(self, Z):
        ''' Total elastic tension (leaflet + surrounding viscoelastic layer)

            :param Z: leaflet apex deflection (m)
            :return: circumferential elastic tension (N/m)
        '''
        return self.TEleaflet(Z) + self.TEtissue(Z)

    def PEtot(self, Z, R):
        ''' Total elastic tension pressure (leaflet + surrounding viscoelastic layer)

            :param Z: leaflet apex deflection (m)
            :param R: leaflet curvature radius (m)
            :return: elastic tension pressure (Pa)
        '''
        return - self.TEtot(Z) / R

    def PVleaflet(self, U, R):
        ''' Viscous stress pressure in leaflet

            :param U: leaflet apex deflection velocity (m/s)
            :param R: leaflet curvature radius (m)
            :return: leaflet viscous stress pressure (Pa)
        '''
        return - 12 * U * self.delta0 * self.muS / R**2

    def PVfluid(self, U, R):
        ''' Viscous stress pressure in surrounding medium

            :param U: leaflet apex deflection velocity (m/s)
            :param R: leaflet curvature radius (m)
            :return: fluid viscous stress pressure (Pa)
        '''
        return - 4 * U * self.muL / np.abs(R)

    def accP(self, Ptot, R):
        ''' Leaflet transverse acceleration resulting from pressure imbalance

            :param Ptot: net pressure (Pa)
            :param R: leaflet curvature radius (m)
            :return: pressure-driven acceleration (m/s^2)
        '''
        return Ptot / (self.rhoL * np.abs(R))

    def accNL(self, U, R):
        ''' Leaflet transverse nonlinear acceleration

            :param U: leaflet apex deflection velocity (m/s)
            :param R: leaflet curvature radius (m)
            :return: nonlinear acceleration term (m/s^2)

            .. note:: A simplified version of nonlinear acceleration (neglecting
                dR/dH) is used here.
            '''
        # return - (3/2 - 2*R/H) * U**2 / R
        return -(3 * U**2) / (2 * R)

    def derivatives(self, y, t, Adrive, Fdrive, Qm, phi, Pm_comp_method=PmCompMethod.predict):
        ''' Evolution of the mechanical system

            :param y: vector of HH system variables at time t
            :param t: time instant (s)
            :param Adrive: acoustic drive amplitude (Pa)
            :param Fdrive: acoustic drive frequency (Hz)
            :param Qm: membrane charge density (F/m2)
            :param phi: acoustic drive phase (rad)
            :param Pm_comp_method: computation method for average intermolecular pressure
            :return: vector of mechanical system derivatives at time t
        '''

        # Split input vector explicitly
        U, Z, ng = y

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
        dngdt = self.gasFlux(Z, Pg)

        # Return derivatives vector
        return [dUdt, dZdt, dngdt]

    def checkInputs(self, Fdrive, Adrive, Qm, phi):
        ''' Check validity of stimulation parameters

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
        ''' Simulate system until periodic stabilization for a specific set of ultrasound parameters,
            and return output data in a dataframe.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param phi: acoustic drive phase (rad)
            :param Qm: imposed membrane charge density (C/m2)
            :param Pm_comp_method: type of method used to compute average intermolecular pressure
            :return: 2-tuple with the output dataframe and computation time.
        '''

        logger.info('%s: simulation @ f = %sHz, A = %sPa, Q = %sC/cm2',
                    self.pprint(), *si_format([Fdrive, Adrive, Qm * 1e-4], 2, space=' '))

        # Check validity of stimulation parameters
        self.checkInputs(Fdrive, Adrive, Qm, phi)

        # Determine time step
        dt = 1 / (NPC_FULL * Fdrive)

        # Compute non-zero deflection value for a small perturbation (solving quasi-steady equation)
        Pac = self.Pacoustic(dt, Adrive, Fdrive, phi)
        Z0 = self.balancedefQS(self.ng0, Qm, Pac, Pm_comp_method)

        # Set initial conditions
        y0 = np.array([0., Z0, self.ng0])

        # Initialize simulator and compute solution
        simulator = PeriodicSimulator(
            lambda y, t: self.derivatives(y, t, Adrive, Fdrive, Qm, phi, Pm_comp_method),
            ivars_to_check=[1, 2])
        (t, y, stim), tcomp = simulator(y0, dt, Fdrive, monitor_time=True)
        logger.debug('completed in %ss', si_format(tcomp, 1))

        # Set last stimulation state to zero
        stim[-1] = 0

        # Store output in dataframe
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Z': y[:, 1],
            'ng': y[:, 2]
        })

        # Return dataframe and computation time
        return data, tcomp

    def meta(self, Fdrive, Adrive, Qm):
        ''' Return information about object and simulation parameters.

            :param Fdrive: US frequency (Hz)
            :param Adrive: acoustic pressure amplitude (Pa)
            :param Qm: applied membrane charge density (C/m2)
            :return: meta-data dictionary
        '''
        return {
            'a': self.a,
            'd': self.d,
            'Cm0': self.Cm0,
            'Qm0': self.Qm0,
            'Fdrive': Fdrive,
            'Adrive': Adrive,
            'Qm': Qm
        }

    def runAndSave(self, outdir, Fdrive, Adrive, Qm):
        ''' Simulate system and save results in a PKL file.

            :param outdir: full path to output directory
            :param Fdrive: US frequency (Hz)
            :param Adrive: acoustic pressure amplitude (Pa)
            :param Qm: applied membrane charge density (C/m2)
            :return: full path to the data file
        '''
        data, tcomp = self.simulate(Fdrive, Adrive, Qm)
        meta = self.meta(Fdrive, Adrive, Qm)
        meta['tcomp'] = tcomp
        simcode = self.filecode(Fdrive, Adrive, Qm)
        outpath = '{}/{}.pkl'.format(outdir, simcode)
        with open(outpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': data}, fh)
        logger.debug('simulation data exported to "%s"', outpath)

        return outpath

    def getCycleProfiles(self, Fdrive, Adrive, Qm):
        ''' Simulate mechanical system and compute pressures over the last acoustic cycle

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param Qm: imposed membrane charge density (C/m2)
            :return: dataframe with the time, kinematic and pressure profiles over the last cycle.
        '''

        # Run default simulation and compute relevant profiles
        logger.info('Running mechanical simulation (a = %sm, f = %sHz, A = %sPa)',
                    si_format(self.a, 1), si_format(Fdrive, 1), si_format(Adrive, 1))
        data, tcomp = self.simulate(Fdrive, Adrive, Qm, Pm_comp_method=PmCompMethod.direct)
        t, Z, ng = [data.loc[-NPC_FULL:, key].values for key in ['t', 'Z', 'ng']]
        dt = (t[-1] - t[0]) / (NPC_FULL - 1)
        t -= t[0]

        # Compute pressure cyclic profiles
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
