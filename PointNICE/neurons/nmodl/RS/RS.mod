TITLE RS neuron
:
: Equations governing the behavior of a regular spiking neuron
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
    SUFFIX RS
    NONSPECIFIC_CURRENT iNa
    NONSPECIFIC_CURRENT iKd
    NONSPECIFIC_CURRENT iM
    NONSPECIFIC_CURRENT iLeak

    : simulation parameters (stimulus and dt)
    RANGE duration, PRF, DC, stimon

    : physiological variables
    RANGE Q, Vmeff, gnabar, gkdbar, gmbar, gleak, eleak, ek, ena
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
    ek = -90            (mV)
    eleak = -70.3       (mV)
    gnabar = 0.056      (mho/cm2)
    gkdbar = 0.006      (mho/cm2)
    gmbar = 7.5e-5      (mho/cm2)
    gleak = 2.05e-5     (mho/cm2)
}

STATE {
    : Differential variables other than v, i.e. the ion channels gating states

    m  : iNa activation gate
    h  : iNa inactivation gate
    n  : iKd activation gate
    p  : iM activation gate
}

ASSIGNED {
    : Variables computed during the simulation and whose value can be retrieved
    Q       (nC/cm2)
    Vmeff   (mV)
    v       (mV)
    iNa     (mA/cm2)
    iKd     (mA/cm2)
    iM      (mA/cm2)
    iLeak   (mA/cm2)
    alpha_h (/ms)
    beta_h  (/ms)
    alpha_m (/ms)
    beta_m  (/ms)
    alpha_n (/ms)
    beta_n  (/ms)
    alpha_p (/ms)
    beta_p  (/ms)
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
    iM = gmbar * p * (Vmeff - ek)
    iLeak = gleak * (Vmeff - eleak)
}

DERIVATIVE states {
    : Compute states derivatives

    : compute ON or OFF rate constants accordingly to stimon value
    if(stimon) {interpolate_on(Q)} else {interpolate_off(Q)}

    : compute gating states derivatives
    m' = alpha_m * (1 - m) - beta_m * m
    h' = alpha_h * (1 - h) - beta_h * h
    n' = alpha_n * (1 - n) - beta_n * n
    p' = alpha_p * (1 - p) - beta_p * p
}

UNITSOFF : turning off units checking for functions not working with standard SI units

INITIAL {
    : Initialize variables and parameters

    : compute Q
    Q = v * cm

    : set initial states values
    m = alpham_off(Q) / (alpham_off(Q) + betam_off(Q))
    h = alphah_off(Q) / (alphah_off(Q) + betah_off(Q))
    n = alphan_off(Q) / (alphan_off(Q) + betan_off(Q))
    p = alphap_off(Q) / (alphap_off(Q) + betap_off(Q))

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
FUNCTION_TABLE alphap_on(x)     (/ms)
FUNCTION_TABLE alphap_off(x)    (/ms)
FUNCTION_TABLE betap_on(x)      (/ms)
FUNCTION_TABLE betap_off(x)     (/ms)

PROCEDURE interpolate_on(Q){
    : Interpolate rate constants durong US-ON periods from appropriate tables

    alpha_m = alpham_on(Q)
    beta_m = betam_on(Q)
    alpha_h = alphah_on(Q)
    beta_h = betah_on(Q)
    alpha_n = alphan_on(Q)
    beta_n = betan_on(Q)
    alpha_p = alphap_on(Q)
    beta_p = betap_on(Q)
}

PROCEDURE interpolate_off(Q){
    : Interpolate rate constants durong US-OFF periods from appropriate tables

    alpha_m = alpham_off(Q)
    beta_m = betam_off(Q)
    alpha_h = alphah_off(Q)
    beta_h = betah_off(Q)
    alpha_n = alphan_off(Q)
    beta_n = betan_off(Q)
    alpha_p = alphap_off(Q)
    beta_p = betap_off(Q)
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