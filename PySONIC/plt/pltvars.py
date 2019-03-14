# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-21 14:33:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-13 20:09:07

''' Dictionary of plotting settings for output variables of the model.  '''


pltvars = {

    't_ms': {
        'desc': 'time',
        'label': 'time',
        'unit': 'ms',
        'factor': 1e3,
        'onset': 1e-3
    },

    't_us': {
        'desc': 'time',
        'label': 'time',
        'unit': 'us',
        'factor': 1e6,
        'onset': 1e-6
    },

    'Z': {
        'desc': 'leaflets deflection',
        'label': 'Z',
        'unit': 'nm',
        'factor': 1e9,
        'min': -1.0,
        'max': 10.0
    },

    'ng': {
        'desc': 'gas content',
        'label': 'n_g',
        'unit': '10^{-22}\ mol',
        'factor': 1e22,
        'min': 1.0,
        'max': 15.0
    },

    'Pac': {
        'desc': 'acoustic pressure',
        'label': 'P_{AC}',
        'unit': 'kPa',
        'factor': 1e-3,
        'alias': 'bls.Pacoustic(t, meta["Adrive"] * states, Fdrive)'
    },

    'Pmavg': {
        'desc': 'average intermolecular pressure',
        'label': 'P_M',
        'unit': 'kPa',
        'factor': 1e-3,
        'alias': 'bls.PMavgpred(df["Z"])'
    },

    'Telastic': {
        'desc': 'leaflet elastic tension',
        'label': 'T_E',
        'unit': 'mN/m',
        'factor': 1e3,
        'alias': 'bls.TEleaflet(df["Z"])'
    },

    'Cm': {
        'desc': 'membrane capacitance',
        'label': 'C_m',
        'unit': 'uF/cm^2',
        'factor': 1e2,
        'min': 0.0,
        'max': 1.5,
        'alias': 'bls.v_Capct(df["Z"])'
    },

    'Nai': {
        'desc': 'sumbmembrane Na+ concentration',
        'label': '[Na^+]_i',
        'unit': 'uM',
        'factor': 1e6
    },

    'Nai_arb': {
        'key': 'Nai',
        'desc': 'submembrane Na+ concentration',
        'label': '[Na^+]',
        'unit': 'arb.',
        'factor': 1
    },

    'C_Na_arb_activation': {
        'key': 'A_Na',
        'desc': 'Na+ dependent PumpNa current activation',
        'label': 'A_{Na^+}',
        'unit': 'arb',
        'factor': 1
    },

    'C_Ca_arb': {
        'key': 'C_Ca',
        'desc': 'submembrane Ca2+ concentration',
        'label': '[Ca^{2+}]',
        'unit': 'arb.',
        'factor': 1
    },

    'C_Ca_arb_activation': {
        'key': 'A_Ca',
        'desc': 'Ca2+ dependent Potassium current activation',
        'label': 'A_{Ca^{2+}}',
        'unit': 'arb',
        'factor': 1
    },

}
