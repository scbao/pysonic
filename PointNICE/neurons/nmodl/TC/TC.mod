TITLE TC neuron
:
: Equations governing the behavior of a thalamo-cortical neuron
: during ultrasonic stimulation, according to the NICE model.
:
: Written by Theo Lemaire and Simon Narduzzi, EPFL, 2018
: Contact: theo.lemaire@epfl.ch


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

UNITS {
    :(mA) = (milliamp)
    :(mV) = (millivolt)
    :(uF) = (microfarad)
    :(nC) = (nanocoulomb)

    (molar) = (1/liter)         : moles do not appear in units
    (M)     = (molar)
    (mM)    = (millimolar)
    (um)    = (micron)
    (mA)    = (milliamp)
    (msM)   = (ms mM)
}

NEURON {
    : Definition of NEURON mechanism

    : mechanism name and constituting currents
    SUFFIX TC
    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iKl
    :NONSPECIFIC_CURRENT iH
    NONSPECIFIC_CURRENT iCa
    NONSPECIFIC_CURRENT iLeak

    : simulation parameters (stimulus and dt)
    RANGE duration, PRF, DC, stimon

    : physiological variables
    RANGE Q, Vmeff, gnabar, gkdbar, gkl, gcabar, gleak, ghbar, eleak, ek, ena, eca, eh
    RANGE k1, k2, k3, k4, nca
    RANGE depth, taur, camin
}

CONSTANT {
    FARADAY = 96494     (coul) : moles do not appear in units
}


PARAMETER {
    : Parameters set by the user upon initialization

    : simulation parameters (stimulus and dt)
    duration = 100      (ms)
    offset = 0          (ms)
    PRF = 0             (Hz)
    DC = 0
    dt = 0.05           (ms)

    : membrane properties
    cm = 1              (uF/cm2)
    ena = 50            (mV)
    eca = 120           (mV)
    ek = -90            (mV)
    eh = -40            (mV)
    eleak = -70         (mV)
    gnabar = 0.09       (mho/cm2)
    gkdbar = 0.01       (mho/cm2)
    gkl = 1.38e-5       (mho/cm2)
    gcabar = 0.002      (mho/cm2)
    ghbar = 1.75e-5     (mho/cm2)
    gleak = 1e-5        (mho/cm2)

    : iH Calcium dependence properties
    k1 = 2.5e19         (1/M*M*M*M*ms)    : CB protein Calcium-driven activation rate
    k2 = 0.0004         (1/ms)            : CB protein inactivation rate
    k3 = 0.1            (1/ms)            : CB protein iH channel binding rate
    k4  = 0.001         (1/ms)            : CB protein iH channel unbinding rate
    nca = 4                               : number of Calcium binding sites on CB protein

    : submembrane Calcium evolution properties
    depth = 1e-7        (m)   : depth of shell
    taur = 5            (ms)   : rate of calcium removal
    camin = 5e-8        (M)   : minimal intracellular Calcium concentration

}

STATE {
    : Differential variables other than v

    m  : iNa activation gate
    h  : iNa inactivation gate
    n  : iKd activation gate
    s  : iCa activation gate
    u  : iCa inactivation gate
    C1  : iH channel closed state
    O  : iH channel open state
    P0  : proportion of unbound CB protein
    C_Ca (M) : submembrane Calcium concentration
}

ASSIGNED {
    : Variables computed during the simulation and whose value can be retrieved
    Q        (nC/cm2)
    Vmeff    (mV)
    v        (mV)
    iNa      (mA/cm2)
    iKd      (mA/cm2)
    iKl      (mA/cm2)
    iCa      (mA/cm2)
    iH       (mA/cm2)
    iLeak    (mA/cm2)
    alpha_h  (/ms)
    beta_h   (/ms)
    alpha_m  (/ms)
    beta_m   (/ms)
    alpha_n  (/ms)
    beta_n   (/ms)
    alpha_s  (/ms)
    beta_s   (/ms)
    alpha_u  (/ms)
    beta_u   (/ms)
    alpha_o  (/ms)
    beta_o   (/ms)
    iCadrive (M/ms)
    stimon
    tint    (ms)
}

BREAKPOINT {
    : Main computation block

    : compute Q
    Q = v * cm

    : update stimulation boolean
    stimonoff()

    : integrate states
    SOLVE states METHOD cnexp

    : compute Vmeff
    if(stimon) {Vmeff = veff_on(Q)} else {Vmeff = veff_off(Q)}

    : compute ionic currents
    iNa = gnabar * m * m * m * h * (Vmeff - ena)
    iKd = gkdbar * n * n * n * n * (Vmeff - ek)
    iKl = gkl * (Vmeff - ek)
    iCa = gcabar * s * s * u * (Vmeff - eca)
    iLeak = gleak * (Vmeff - eleak)
    iH = ghbar * (O + 2 * (1 - O - C1)) * (Vmeff - eh)
}

UNITSOFF : turning off units checking for functions not working with standard SI units

DERIVATIVE states {
    : Compute states derivatives

    : compute ON or OFF rate constants accordingly to stimon value
    if(stimon) {interpolate_on(Q)} else {interpolate_off(Q)}

    : compute gating states derivatives
    m' = alpha_m * (1 - m) - beta_m * m
    h' = alpha_h * (1 - h) - beta_h * h
    n' = alpha_n * (1 - n) - beta_n * n
    s' = alpha_s * (1 - s) - beta_s * s
    u' = alpha_u * (1 - u) - beta_u * u

    : compute derivatives of variables for the kinetics system of Ih
    iCadrive = -1e-5 * iCa / (2 * FARADAY * depth)
    if (iCadrive <= 0.) { iCadrive = 0. } : cannot pump inward
    C_Ca' = (camin - C_Ca) / taur + iCadrive
    P0' = k2 * (1 - P0) - k1 * P0 * capow(C_Ca)
    C1' = beta_o * O - alpha_o * C1
    O' = alpha_o * C1 - beta_o * O - k3 * O * (1 - P0) + k4 * (1 - O - C1)
}



INITIAL {
    : Initialize variables and parameters

    : compute Q
    Q = v * cm

    : set initial states values
    m = alpham_off(Q) / (alpham_off(Q) + betam_off(Q))
    h = alphah_off(Q) / (alphah_off(Q) + betah_off(Q))
    n = alphan_off(Q) / (alphan_off(Q) + betan_off(Q))
    s = alphas_off(Q) / (alphas_off(Q) + betas_off(Q))
    u = alphau_off(Q) / (alphau_off(Q) + betau_off(Q))

    : compute steady-state Calcium concentration
    iCa = gcabar * s * s * u * (veff_off(Q) - eca)
    iCadrive = -1e-5 * iCa / (2 * FARADAY * depth)
    C_Ca = camin + taur * iCadrive

    : compute steady values for the kinetics system of Ih
    P0 = k2 / (k2 + k1 * C_Ca^nca)
    O = k4 / (k3 * (1 - P0) + k4 * (1 + betao_off(Q) / alphao_off(Q)))
    C1 = betao_off(Q) / alphao_off(Q) * O

    : initialize tint, set stimulation state and correct PRF
    tint = 0
    stimon = 0
    PRF = PRF / 2 : for some reason PRF must be halved in order to ensure proper on/off switch
}

: Define function tables for variables to be interpolated during ON and OFF periods
FUNCTION_TABLE veff_on(x)       (mV)
FUNCTION_TABLE veff_off(x)      (mV)
FUNCTION_TABLE alpham_on(x)     (/ms)
FUNCTION_TABLE alpham_off(x)    (/ms)
FUNCTION_TABLE betam_on(x)      (/ms)
FUNCTION_TABLE betam_off(x)     (/ms)
FUNCTION_TABLE alphah_on(x)     (/ms)
FUNCTION_TABLE alphah_off(x)    (/ms)
FUNCTION_TABLE betah_on(x)      (/ms)
FUNCTION_TABLE betah_off(x)     (/ms)
FUNCTION_TABLE alphan_on(x)     (/ms)
FUNCTION_TABLE alphan_off(x)    (/ms)
FUNCTION_TABLE betan_on(x)      (/ms)
FUNCTION_TABLE betan_off(x)     (/ms)
FUNCTION_TABLE alphas_on(x)     (/ms)
FUNCTION_TABLE alphas_off(x)    (/ms)
FUNCTION_TABLE betas_on(x)      (/ms)
FUNCTION_TABLE betas_off(x)     (/ms)
FUNCTION_TABLE alphau_on(x)     (/ms)
FUNCTION_TABLE alphau_off(x)    (/ms)
FUNCTION_TABLE betau_on(x)      (/ms)
FUNCTION_TABLE betau_off(x)     (/ms)
FUNCTION_TABLE alphao_on(x)     (/ms)
FUNCTION_TABLE alphao_off(x)    (/ms)
FUNCTION_TABLE betao_on(x)      (/ms)
FUNCTION_TABLE betao_off(x)     (/ms)

PROCEDURE interpolate_on(Q){
    : Interpolate rate constants durong US-ON periods from appropriate tables

    alpha_m = alpham_on(Q)
    beta_m = betam_on(Q)
    alpha_h = alphah_on(Q)
    beta_h = betah_on(Q)
    alpha_n = alphan_on(Q)
    beta_n = betan_on(Q)
    alpha_s = alphas_on(Q)
    beta_s = betas_on(Q)
    alpha_u = alphau_on(Q)
    beta_u = betau_on(Q)
    alpha_o = alphao_on(Q)
    beta_o = betao_on(Q)
}

PROCEDURE interpolate_off(Q){
    : Interpolate rate constants durong US-OFF periods from appropriate tables

    alpha_m = alpham_off(Q)
    beta_m = betam_off(Q)
    alpha_h = alphah_off(Q)
    beta_h = betah_off(Q)
    alpha_n = alphan_off(Q)
    beta_n = betan_off(Q)
    alpha_s = alphas_off(Q)
    beta_s = betas_off(Q)
    alpha_u = alphau_off(Q)
    beta_u = betau_off(Q)
    alpha_o = alphao_off(Q)
    beta_o = betao_off(Q)
}


FUNCTION capow(C_Ca (M)) (M*M*M*M) {
    capow = C_Ca^nca
}


UNITSON : turn back units checking

PROCEDURE stimonoff(){
    : Switch integration to ON or OFF system according to stimulus

    if(t < duration - dt){
        if(tint >= 1000 / PRF){             : reset on at pulse onset
            stimon = 1
            tint = 0
        }else if(tint <= DC * 1000 / PRF){  : ON during TON
            stimon = 1
        }else{                              : OFF during TOFF
            stimon = 0
        }
        tint = tint + dt                    : increment by dt
    }else{                                  : OFF during stimulus offset
        stimon = 0
    }
}