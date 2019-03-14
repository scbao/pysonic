# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-21 14:33:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-14 16:37:23

''' Dictionary of plotting settings for output variables of the model.  '''





tmp = {
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
