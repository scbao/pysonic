# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-21 14:33:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-12-01 08:55:35

''' Dictionary of plotting settings for output variables of the model.  '''


pltvars = {

    't_ms': {
        'desc': 'time',
        'label': 'time',
        'unit': 'ms',
        'factor': 1e3,
        'onset': 3e-3
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
        'factor': 1,
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
        'alias': 'data["m"]**2 * data["h"]'
    },

    'm3h': {
        'desc': 'iNa relative conductance',
        'label': 'm^3h',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'data["m"]**3 * data["h"]'
    },

    'm4h': {
        'desc': 'iNa relative conductance',
        'label': 'm^4h',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'data["m"]**4 * data["h"]'
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

    's2u': {
        'desc': 'iT relative conductance',
        'label': 's^2u',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 1.1,
        'alias': 'data["s"]**2 * data["u"]'
    },

    'w': {
        'desc': 'iKCa activation gates opening',
        'label': 'w-gate',
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

    'O + 2OL': {
        'desc': 'iH activation gate relative conductance',
        'label': 'O\ +\ 2O_L',
        'unit': None,
        'factor': 1,
        'min': -0.1,
        'max': 2.1,
        'alias': 'data["O"] + 2 * (1 - data["O"] - data["C"])'
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

    'C_Na': {
        'desc': 'sumbmembrane Na+ concentration',
        'label': '[Na^+]_i',
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
        'label': 'A_{Ca^{2+}}',
        'unit': 'arb',
        'factor': 1
    },

    'VL': {
        'constant': 'neuron.VL',
        'desc': 'non-specific leakage current resting potential',
        'label': 'V_L',
        'unit': 'mV',
        'factor': 1e0
    },

    'iL': {
        'desc': 'leakage current',
        'label': 'I_L',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currL(data["Vm"])'
    },

    'iNa': {
        'desc': 'Sodium current',
        'label': 'I_{Na}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currNa(data["m"], data["h"], data["Vm"])'
    },

    'iNa2': {
        'desc': 'Sodium current',
        'label': 'I_{Na}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currNa(data["m"], data["h"], data["Vm"], data["C_Na"])'
    },

    'iK': {
        'desc': 'delayed-rectifier Potassium current',
        'label': 'I_K',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currK(data["n"], data["Vm"])'
    },

    'iM': {
        'desc': 'slow non-inactivating Potassium current',
        'label': 'I_M',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currM(data["p"], data["Vm"])'
    },

    'iT': {
        'desc': 'low-threshold Calcium current',
        'label': 'I_T',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currCa(data["s"], data["u"], data["Vm"])'
    },

    'iTs': {
        'desc': 'low-threshold Calcium current',
        'label': 'I_{TS}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currCa(data["s"], data["u"], data["Vm"])'
    },

    'iCa': {
        'desc': 'leech Calcium current',
        'label': 'I_{Ca}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currCa(data["s"], data["Vm"])'
    },

    'iCa2': {
        'desc': 'leech Calcium current',
        'label': 'I_{Ca}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currCa(data["s"], data["Vm"], data["C_Ca"])'
    },

    'iH': {
        'desc': 'hyperpolarization-activated cationic current',
        'label': 'I_h',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currH(data["O"], data["C"], data["Vm"])'
    },

    'iKL': {
        'desc': 'leakage Potassium current',
        'label': 'I_{KL}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currKL(data["Vm"])'
    },

    'iKCa': {
        'desc': 'Calcium-activated Potassium current',
        'label': 'I_{KCa}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currKCa(data["A_Ca"], data["Vm"])'
    },

    'iKCa2': {
        'desc': 'Calcium-activated Potassium current',
        'label': 'I_{KCa}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currKCa(data["w"], data["Vm"])'
    },

    'iPumpNa': {
        'desc': 'Outward current mimicking the activity of the NaK-ATPase pump',
        'label': 'I_{PumpNa}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currPumpNa(data["A_Na"], data["Vm"])'
    },

    'iPumpNa2': {
        'desc': 'Outward current mimicking the activity of the NaK-ATPase pump',
        'label': 'I_{PumpNa}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currPumpNa(data["C_Na"])'
    },

    'iPumpCa2': {
        'desc': 'Outward current describing the removal of Ca2+ from the intracellular space',
        'label': 'I_{PumpCa}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currPumpCa(data["C_Ca"])'
    },

    'iNet': {
        'desc': 'net current',
        'label': 'I_{net}',
        'unit': 'A/m^2',
        'factor': 1e-3,
        'alias': 'neuron.currNet(data["Vm"], neuron_states)'
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
    },

    'alphao': {
        'desc': 'iH channels activation rate (between closed and open forms)',
        'label': '\\alpha_{O,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    },

    'betao': {
        'desc': 'iH channels inactivation rate (between closed and open forms)',
        'label': '\\beta_{O,\ eff}',
        'unit': 'ms^-1',
        'factor': 1e-3
    }
}
