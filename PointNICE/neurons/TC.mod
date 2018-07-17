TITLE TC neuron
:
: Equations governing the behavior of a thalamo-cortical neuron
: during ultrasonic stimulation, according to the NICE model.
:
: Written by Theo Lemaire and Simon Narduzzi, EPFL, 2018
: Contact: theo.lemaire@epfl.ch


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (uF) = (microfarad)
}

NEURON {
    : Definition of NEURON mechanism

    : mechanism name and constituting currents
    SUFFIX TC
    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iKl
    NONSPECIFIC_CURRENT iH
    NONSPECIFIC_CURRENT iCa
    NONSPECIFIC_CURRENT iLeak

    : simulation parameters (stimulus and dt)
    RANGE duration, PRF, DC, stimon

    : physiological variables
    RANGE Q, Vmeff, gnabar, gkdbar, gkl, gcabar, gleak, ghbar, eleak, ek, ena, eca, eh
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
}

STATE {
    : Differential variables other than v, i.e. the ion channels gating states
    m h n p s u
}

ASSIGNED {
    : Variables computed during the simulation and whose value can be retrieved
    Q       (nC/cm2)
    Vmeff   (mV)
    v       (mV)
    iNa     (mA/cm2)
    iKd     (mA/cm2)
    iKl     (mA/cm2)
    iCa     (mA/cm2)
    iH      (mA/cm2)
    iLeak   (mA/cm2)
    alpha_h (/ms)
    beta_h  (/ms)
    alpha_m (/ms)
    beta_m  (/ms)
    alpha_n (/ms)
    beta_n  (/ms)
    alpha_s (/ms)
    beta_s  (/ms)
    alpha_u (/ms)
    beta_u  (/ms)
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
    iH = ghbar * (O + 2 * (1 - O - C)) * (Vmeff - eh)
}

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
}

UNITSOFF : turning off units checking for functions not working with standard SI units

INITIAL {
    : Initialize variables and parameters

    : set initial states values
    m = alpham_off(v) / (alpham_off(v) + betam_off(v))
    h = alphah_off(v) / (alphah_off(v) + betah_off(v))
    n = alphan_off(v) / (alphan_off(v) + betan_off(v))
    s = alphas_off(v) / (alphas_off(v) + betas_off(v))
    u = alphau_off(v) / (alphau_off(v) + betau_off(v))

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
}

PROCEDURE interpolate_off(v){
    : Interpolate rate constants durong US-OFF periods from appropriate tables

    alpha_m = alpham_off(v)
    beta_m = betam_off(v)
    alpha_h = alphah_off(v)
    beta_h = betah_off(v)
    alpha_n = alphan_off(v)
    beta_n = betan_off(v)
    alpha_s = alphas_off(Q)
    beta_s = betas_off(Q)
    alpha_u = alphau_off(Q)
    beta_u = betau_off(Q)
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