# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-21 14:33:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-13 14:54:34

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
        # 'unit': 'ymol',
        # 'factor': 1e24,
        # 'min': 100.0,
        # 'max': 1500.0
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
        'alias': 'bls.v_Capct(df["Z"])'
    },

    'Vm': {
        'desc': 'membrane potential',
        'label': 'V_m',
        'unit': 'mV',
        'factor': 1,
    },

    'Veff': {
        'key': 'V',
        'desc': 'effective membrane potential',
        'label': 'V_{m, eff}',
        'unit': 'mV',
        'factor': 1e0
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

    'm2h': {
        'desc': 'iNa relative conductance',
        'label': 'm^2h',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'df["m"]**2 * df["h"]'
    },

    'm3h': {
        'desc': 'iNa relative conductance',
        'label': 'm^3h',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'df["m"]**3 * df["h"]'
    },

    'm4h': {
        'desc': 'iNa relative conductance',
        'label': 'm^4h',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'df["m"]**4 * df["h"]'
    },

    'n': {
        'desc': 'iK activation gate opening',
        'label': 'n-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'n4': {
        'desc': 'iK relative conductance',
        'label': 'n^4',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'df["n"]**4'
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

    's2u': {
        'desc': 'iT relative conductance',
        'label': 's^2u',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'df["s"]**2 * df["u"]'
    },

    'p': {
        'desc': 'iT activation gates opening',
        'label': 'p-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'q': {
        'desc': 'iT inactivation gates opening',
        'label': 'q-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'p2q': {
        'desc': 'iT relative conductance',
        'label': 'p^2q',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'df["p"]**2 * df["q"]'
    },

    'r': {
        'desc': 'iKCa Ca2+-gated activation gates opening',
        'label': 'r-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'r2': {
        'desc': 'iKCa relative conductance',
        'label': 'r^2',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'df["r"]**2'
    },

    # 'q': {
    #     'desc': 'iCaL activation gates opening',
    #     'label': 'q-gate',
    #     'unit': None,
    #     'factor': 1,
    #     'min': -0.1,
    #     'max': 1.1
    # },

    # 'r': {
    #     'desc': 'iCaL inactivation gates opening',
    #     'label': 'r-gate',
    #     'unit': None,
    #     'factor': 1,
    #     'min': -0.1,
    #     'max': 1.1
    # },

    # 'q2r': {
    #     'desc': 'iCaL relative conductance',
    #     'label': 'q^2r',
    #     'unit': None,
    #     'factor': 1,
    #     'min': -0.1,
    #     'max': 1.1,
    #     'alias': 'df["q"]**2 * df["r"]'
    # },

    # 'c': {
    #     'desc': 'iKCa activation gates opening',
    #     'label': 'c-gate',
    #     'unit': None,
    #     'factor': 1,
    #     'min': -0.1,
    #     'max': 1.1
    # },

    'a': {
        'desc': 'iA activation gates opening',
        'label': 'a-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'b': {
        'desc': 'iA inactivation gates opening',
        'label': 'b-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'a2b': {
        'desc': 'iA relative conductance',
        'label': 'ab',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'df["a"]**2 * df["b"]'
    },


    'c': {
        'desc': 'iL activation gates opening',
        'label': 'c-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'd1': {
        'desc': 'iL inactivation gates opening',
        'label': 'd1-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'd2': {
        'desc': 'iL Ca2+-gated inactivation gates opening',
        'label': 'd2-gate',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'c2d1d2': {
        'desc': 'iL relative conductance',
        'label': 'c^2d_1d_2',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'df["c"]**2 * df["d1"] * df["d2"]'
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
        'alias': '1 - df["O"] - df["C"]'
    },

    'O + 2OL': {
        'desc': 'iH activation gate relative conductance',
        'label': 'O\ +\ 2O_L',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 2.1,
        'alias': 'df["O"] + 2 * (1 - df["O"] - df["C"])'
    },

    'P0': {
        'desc': 'iH regulating factor activation',
        'label': 'P_0',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1
    },

    'Cai': {
        'desc': 'sumbmembrane Ca2+ concentration',
        'label': '[Ca^{2+}]_i',
        'unit': 'uM',
        'factor': 1e6
    },

    'C_Na': {
        'desc': 'sumbmembrane Na+ concentration',
        'label': '[Na^+]_i',
        'unit': 'uM',
        'factor': 1e6
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
        'label': 'A_{Ca^{2+}}',
        'unit': 'arb',
        'factor': 1
    },

    'VLeak': {
        'constant': 'neuron.VLeak',
        'desc': 'non-specific leakage current resting potential',
        'label': 'V_{leak}',
        'unit': 'mV',
        'factor': 1e0
    },

    'alphaa': {
        'desc': 'iA a-gate activation rate',
        'label': '\\alpha_a',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betaa': {
        'desc': 'iA a-gate inactivation rate',
        'label': '\\beta_a',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'alphab': {
        'desc': 'iA b-gate activation rate',
        'label': '\\alpha_b',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betab': {
        'desc': 'iA b-gate inactivation rate',
        'label': '\\beta_b',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'alphac': {
        'desc': 'iL c-gate activation rate',
        'label': '\\alpha_c',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betac': {
        'desc': 'iL c-gate inactivation rate',
        'label': '\\beta_c',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'alphad1': {
        'desc': 'iL d1-gate activation rate',
        'label': '\\alpha_d1',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betad1': {
        'desc': 'iL d1-gate inactivation rate',
        'label': '\\beta_d1',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'alpham': {
        'desc': 'iNa m-gate activation rate',
        'label': '\\alpha_m',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betam': {
        'desc': 'iNa m-gate inactivation rate',
        'label': '\\beta_m',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'alphah': {
        'desc': 'iNa h-gate activation rate',
        'label': '\\alpha_h',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betah': {
        'desc': 'iNa h-gate inactivation rate',
        'label': '\\beta_h',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'alphan': {
        'desc': 'iK n-gate activation rate',
        'label': '\\alpha_n',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betan': {
        'desc': 'iK n-gate inactivation rate',
        'label': '\\beta_n',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'alphap': {
        'desc': 'iM p-gate activation rate',
        'label': '\\alpha_p',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betap': {
        'desc': 'iM p-gate inactivation rate',
        'label': '\\beta_p',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'alphas': {
        'desc': 'iT s-gate activation rate',
        'label': '\\alpha_s',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betas': {
        'desc': 'iT s-gate inactivation rate',
        'label': '\\beta_s',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'alphau': {
        'desc': 'iT u-gate activation rate',
        'label': '\\alpha_u',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betau': {
        'desc': 'iT u-gate inactivation rate',
        'label': '\\beta_u',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'alphaq': {
        'desc': 'iT q-gate activation rate',
        'label': '\\alpha_q',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betaq': {
        'desc': 'iT q-gate inactivation rate',
        'label': '\\beta_q',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },


    'alphao': {
        'desc': 'iH channels activation rate (between closed and open forms)',
        'label': '\\alpha_O',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    },

    'betao': {
        'desc': 'iH channels inactivation rate (between closed and open forms)',
        'label': '\\beta_O',
        'unit': 'ms^{-1}',
        'factor': 1e-3
    }
}
