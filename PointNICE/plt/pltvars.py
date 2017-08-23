# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-21 14:33:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-23 14:52:26

''' Dictionary of plotting settings for output variables of the model.  '''


pltvars = {

    't': {
        'desc': 'time',
        'label': 'time',
        'unit': 'ms',
        'factor': 1e3,
        'onset': 3e-3
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
        'label': 'gas',
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
        'alias': 'bls.Pacoustic(t, data["Adrive"] * states, Fdrive)'
    },

    'Pmavg': {
        'desc': 'average intermolecular pressure',
        'label': 'P_M',
        'unit': 'kPa',
        'factor': 1e-3,
        'alias': 'bls.PMavgpred(data["Z"])'
    },

    'Telastic': {
        'desc': 'leaflet elastic tension',
        'label': 'T_E',
        'unit': 'mN/m',
        'factor': 1e3,
        'alias': 'bls.TEleaflet(data["Z"])'
    },

    'Qm': {
        'desc': 'charge density',
        'label': 'Q_m',
        'unit': 'nC/cm^2',
        'factor': 1e5,
        'min': -100,
        'max': 50
    },

    'Cm': {
        'desc': 'membrane capacitance',
        'label': 'C_m',
        'unit': 'uF/cm^2',
        'factor': 1e2,
        'min': 0.0,
        'max': 1.5,
        'alias': 'np.array([bls.Capct(ZZ) for ZZ in data["Z"]])'
    },

    'Vm': {
        'desc': 'membrane potential',
        'label': 'V_m',
        'unit': 'mV',
        'factor': 1e3,
        'alias': 'data["Qm"] / np.array([bls.Capct(ZZ) for ZZ in data["Z"]])'
    },


    'iL': {
        'desc': 'leakage current',
        'label': 'I_L',
        'unit': 'mA/cm^2',
        'factor': 1,
        'alias': 'neuron.currL(data["Qm"] * 1e3 / np.array([bls.Capct(ZZ) for ZZ in data["Z"]]))'
    },


    'm': {
        'desc': 'iNa activation gate opening',
        'label': 'm-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'h': {
        'desc': 'iNa inactivation gate opening',
        'label': 'h-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'n': {
        'desc': 'iK activation gate opening',
        'label': 'n-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'p': {
        'desc': 'iM activation gate opening',
        'label': 'p-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    's': {
        'desc': 'iCa activation gates opening',
        'label': 's-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'u': {
        'desc': 'iCa inactivation gates opening',
        'label': 'u-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'O': {
        'desc': 'iH activation gate opening',
        'label': 'O',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'OL': {
        'desc': 'iH activation gate locked-opening',
        'label': 'O_L',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': '1 - data["O"] - data["C"]'
    },

    'P0': {
        'desc': 'iH regulating factor activation',
        'label': 'P_0',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'C_Ca': {
        'desc': 'sumbmembrane Ca2+ concentration',
        'label': '[Ca^{2+}]_i',
        'unit': 'uM',
        'factor': 1e6,
        'min': 0,
        'max': 150.0
    },

    'C_Na_arb': {
        'key': 'C_Na',
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
        'label': 'A_{Na^{2+}}',
        'unit': 'arb',
        'factor': 1
    },

    'VL': {
        'constant': 'neuron.VL',
        'desc': 'non-specific leakage current resting potential',
        'label': 'A_{Na^{2+}}',
        'unit': 'mV',
        'factor': 1e0
    },

    'Veff': {
        'key': 'V',
        'desc': 'effective membrane potential',
        'label': 'V_{m, eff}',
        'unit': 'mV',
        'factor': 1e0
    },

    'alpham': {
        'desc': 'iNa m-gate activation rate',
        'label': '\\alpha_{m,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'betam': {
        'desc': 'iNa m-gate inactivation rate',
        'label': '\\beta_{m,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'alphah': {
        'desc': 'iNa h-gate activation rate',
        'label': '\\alpha_{h,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'betah': {
        'desc': 'iNa h-gate inactivation rate',
        'label': '\\beta_{h,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'alphan': {
        'desc': 'iK n-gate activation rate',
        'label': '\\alpha_{n,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'betan': {
        'desc': 'iK n-gate inactivation rate',
        'label': '\\beta_{n,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'alphap': {
        'desc': 'iM p-gate activation rate',
        'label': '\\alpha_{p,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'betap': {
        'desc': 'iM p-gate inactivation rate',
        'label': '\\beta_{p,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'alphas': {
        'desc': 'iT s-gate activation rate',
        'label': '\\alpha_{s,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'betas': {
        'desc': 'iT s-gate inactivation rate',
        'label': '\\beta_{s,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'alphau': {
        'desc': 'iT u-gate activation rate',
        'label': '\\alpha_{u,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'betau': {
        'desc': 'iT u-gate inactivation rate',
        'label': '\\beta_{u,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    }
}
