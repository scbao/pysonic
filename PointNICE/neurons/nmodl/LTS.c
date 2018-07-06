/* Created by Language version: 6.2.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__LTS
#define _nrn_initial _nrn_initial__LTS
#define nrn_cur _nrn_cur__LTS
#define _nrn_current _nrn_current__LTS
#define nrn_jacob _nrn_jacob__LTS
#define nrn_state _nrn_state__LTS
#define _net_receive _net_receive__LTS 
#define interpolate_off interpolate_off__LTS 
#define interpolate_on interpolate_on__LTS 
#define states states__LTS 
#define stimonoff stimonoff__LTS 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define duration _p[0]
#define PRF _p[1]
#define DC _p[2]
#define ena _p[3]
#define eca _p[4]
#define ek _p[5]
#define eleak _p[6]
#define gnabar _p[7]
#define gkdbar _p[8]
#define gmbar _p[9]
#define gcabar _p[10]
#define gleak _p[11]
#define Q _p[12]
#define Vmeff _p[13]
#define iNa _p[14]
#define iKd _p[15]
#define iM _p[16]
#define iCa _p[17]
#define iLeak _p[18]
#define stimon _p[19]
#define m _p[20]
#define h _p[21]
#define n _p[22]
#define p _p[23]
#define s _p[24]
#define u _p[25]
#define Dm _p[26]
#define Dh _p[27]
#define Dn _p[28]
#define Dp _p[29]
#define Ds _p[30]
#define Du _p[31]
#define alpha_h _p[32]
#define beta_h _p[33]
#define alpha_m _p[34]
#define beta_m _p[35]
#define alpha_n _p[36]
#define beta_n _p[37]
#define alpha_p _p[38]
#define beta_p _p[39]
#define alpha_s _p[40]
#define beta_s _p[41]
#define alpha_u _p[42]
#define beta_u _p[43]
#define tint _p[44]
#define v _p[45]
#define _g _p[46]
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_alphau_on(void);
 static void _hoc_alphas_on(void);
 static void _hoc_alphap_on(void);
 static void _hoc_alphan_on(void);
 static void _hoc_alphah_on(void);
 static void _hoc_alpham_on(void);
 static void _hoc_alphau_off(void);
 static void _hoc_alphas_off(void);
 static void _hoc_alphap_off(void);
 static void _hoc_alphan_off(void);
 static void _hoc_alphah_off(void);
 static void _hoc_alpham_off(void);
 static void _hoc_betau_on(void);
 static void _hoc_betas_on(void);
 static void _hoc_betap_on(void);
 static void _hoc_betan_on(void);
 static void _hoc_betah_on(void);
 static void _hoc_betam_on(void);
 static void _hoc_betau_off(void);
 static void _hoc_betas_off(void);
 static void _hoc_betap_off(void);
 static void _hoc_betan_off(void);
 static void _hoc_betah_off(void);
 static void _hoc_betam_off(void);
 static void _hoc_interpolate_off(void);
 static void _hoc_interpolate_on(void);
 static void _hoc_stimonoff(void);
 static void _hoc_table_betau_off(void);
 static void _hoc_table_betau_on(void);
 static void _hoc_table_alphau_off(void);
 static void _hoc_table_alphau_on(void);
 static void _hoc_table_betas_off(void);
 static void _hoc_table_betas_on(void);
 static void _hoc_table_alphas_off(void);
 static void _hoc_table_alphas_on(void);
 static void _hoc_table_betap_off(void);
 static void _hoc_table_betap_on(void);
 static void _hoc_table_alphap_off(void);
 static void _hoc_table_alphap_on(void);
 static void _hoc_table_betan_off(void);
 static void _hoc_table_betan_on(void);
 static void _hoc_table_alphan_off(void);
 static void _hoc_table_alphan_on(void);
 static void _hoc_table_betah_off(void);
 static void _hoc_table_betah_on(void);
 static void _hoc_table_alphah_off(void);
 static void _hoc_table_alphah_on(void);
 static void _hoc_table_betam_off(void);
 static void _hoc_table_betam_on(void);
 static void _hoc_table_alpham_off(void);
 static void _hoc_table_alpham_on(void);
 static void _hoc_table_veff_off(void);
 static void _hoc_table_veff_on(void);
 static void _hoc_veff_off(void);
 static void _hoc_veff_on(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_LTS", _hoc_setdata,
 "alphau_on_LTS", _hoc_alphau_on,
 "alphas_on_LTS", _hoc_alphas_on,
 "alphap_on_LTS", _hoc_alphap_on,
 "alphan_on_LTS", _hoc_alphan_on,
 "alphah_on_LTS", _hoc_alphah_on,
 "alpham_on_LTS", _hoc_alpham_on,
 "alphau_off_LTS", _hoc_alphau_off,
 "alphas_off_LTS", _hoc_alphas_off,
 "alphap_off_LTS", _hoc_alphap_off,
 "alphan_off_LTS", _hoc_alphan_off,
 "alphah_off_LTS", _hoc_alphah_off,
 "alpham_off_LTS", _hoc_alpham_off,
 "betau_on_LTS", _hoc_betau_on,
 "betas_on_LTS", _hoc_betas_on,
 "betap_on_LTS", _hoc_betap_on,
 "betan_on_LTS", _hoc_betan_on,
 "betah_on_LTS", _hoc_betah_on,
 "betam_on_LTS", _hoc_betam_on,
 "betau_off_LTS", _hoc_betau_off,
 "betas_off_LTS", _hoc_betas_off,
 "betap_off_LTS", _hoc_betap_off,
 "betan_off_LTS", _hoc_betan_off,
 "betah_off_LTS", _hoc_betah_off,
 "betam_off_LTS", _hoc_betam_off,
 "interpolate_off_LTS", _hoc_interpolate_off,
 "interpolate_on_LTS", _hoc_interpolate_on,
 "stimonoff_LTS", _hoc_stimonoff,
 "table_betau_off_LTS", _hoc_table_betau_off,
 "table_betau_on_LTS", _hoc_table_betau_on,
 "table_alphau_off_LTS", _hoc_table_alphau_off,
 "table_alphau_on_LTS", _hoc_table_alphau_on,
 "table_betas_off_LTS", _hoc_table_betas_off,
 "table_betas_on_LTS", _hoc_table_betas_on,
 "table_alphas_off_LTS", _hoc_table_alphas_off,
 "table_alphas_on_LTS", _hoc_table_alphas_on,
 "table_betap_off_LTS", _hoc_table_betap_off,
 "table_betap_on_LTS", _hoc_table_betap_on,
 "table_alphap_off_LTS", _hoc_table_alphap_off,
 "table_alphap_on_LTS", _hoc_table_alphap_on,
 "table_betan_off_LTS", _hoc_table_betan_off,
 "table_betan_on_LTS", _hoc_table_betan_on,
 "table_alphan_off_LTS", _hoc_table_alphan_off,
 "table_alphan_on_LTS", _hoc_table_alphan_on,
 "table_betah_off_LTS", _hoc_table_betah_off,
 "table_betah_on_LTS", _hoc_table_betah_on,
 "table_alphah_off_LTS", _hoc_table_alphah_off,
 "table_alphah_on_LTS", _hoc_table_alphah_on,
 "table_betam_off_LTS", _hoc_table_betam_off,
 "table_betam_on_LTS", _hoc_table_betam_on,
 "table_alpham_off_LTS", _hoc_table_alpham_off,
 "table_alpham_on_LTS", _hoc_table_alpham_on,
 "table_veff_off_LTS", _hoc_table_veff_off,
 "table_veff_on_LTS", _hoc_table_veff_on,
 "veff_off_LTS", _hoc_veff_off,
 "veff_on_LTS", _hoc_veff_on,
 0, 0
};
#define alphau_on alphau_on_LTS
#define alphas_on alphas_on_LTS
#define alphap_on alphap_on_LTS
#define alphan_on alphan_on_LTS
#define alphah_on alphah_on_LTS
#define alpham_on alpham_on_LTS
#define alphau_off alphau_off_LTS
#define alphas_off alphas_off_LTS
#define alphap_off alphap_off_LTS
#define alphan_off alphan_off_LTS
#define alphah_off alphah_off_LTS
#define alpham_off alpham_off_LTS
#define betau_on betau_on_LTS
#define betas_on betas_on_LTS
#define betap_on betap_on_LTS
#define betan_on betan_on_LTS
#define betah_on betah_on_LTS
#define betam_on betam_on_LTS
#define betau_off betau_off_LTS
#define betas_off betas_off_LTS
#define betap_off betap_off_LTS
#define betan_off betan_off_LTS
#define betah_off betah_off_LTS
#define betam_off betam_off_LTS
#define table_betau_off table_betau_off_LTS
#define table_betau_on table_betau_on_LTS
#define table_alphau_off table_alphau_off_LTS
#define table_alphau_on table_alphau_on_LTS
#define table_betas_off table_betas_off_LTS
#define table_betas_on table_betas_on_LTS
#define table_alphas_off table_alphas_off_LTS
#define table_alphas_on table_alphas_on_LTS
#define table_betap_off table_betap_off_LTS
#define table_betap_on table_betap_on_LTS
#define table_alphap_off table_alphap_off_LTS
#define table_alphap_on table_alphap_on_LTS
#define table_betan_off table_betan_off_LTS
#define table_betan_on table_betan_on_LTS
#define table_alphan_off table_alphan_off_LTS
#define table_alphan_on table_alphan_on_LTS
#define table_betah_off table_betah_off_LTS
#define table_betah_on table_betah_on_LTS
#define table_alphah_off table_alphah_off_LTS
#define table_alphah_on table_alphah_on_LTS
#define table_betam_off table_betam_off_LTS
#define table_betam_on table_betam_on_LTS
#define table_alpham_off table_alpham_off_LTS
#define table_alpham_on table_alpham_on_LTS
#define table_veff_off table_veff_off_LTS
#define table_veff_on table_veff_on_LTS
#define veff_off veff_off_LTS
#define veff_on veff_on_LTS
 extern double alphau_on( _threadargsprotocomma_ double );
 extern double alphas_on( _threadargsprotocomma_ double );
 extern double alphap_on( _threadargsprotocomma_ double );
 extern double alphan_on( _threadargsprotocomma_ double );
 extern double alphah_on( _threadargsprotocomma_ double );
 extern double alpham_on( _threadargsprotocomma_ double );
 extern double alphau_off( _threadargsprotocomma_ double );
 extern double alphas_off( _threadargsprotocomma_ double );
 extern double alphap_off( _threadargsprotocomma_ double );
 extern double alphan_off( _threadargsprotocomma_ double );
 extern double alphah_off( _threadargsprotocomma_ double );
 extern double alpham_off( _threadargsprotocomma_ double );
 extern double betau_on( _threadargsprotocomma_ double );
 extern double betas_on( _threadargsprotocomma_ double );
 extern double betap_on( _threadargsprotocomma_ double );
 extern double betan_on( _threadargsprotocomma_ double );
 extern double betah_on( _threadargsprotocomma_ double );
 extern double betam_on( _threadargsprotocomma_ double );
 extern double betau_off( _threadargsprotocomma_ double );
 extern double betas_off( _threadargsprotocomma_ double );
 extern double betap_off( _threadargsprotocomma_ double );
 extern double betan_off( _threadargsprotocomma_ double );
 extern double betah_off( _threadargsprotocomma_ double );
 extern double betam_off( _threadargsprotocomma_ double );
 extern double table_betau_off( );
 extern double table_betau_on( );
 extern double table_alphau_off( );
 extern double table_alphau_on( );
 extern double table_betas_off( );
 extern double table_betas_on( );
 extern double table_alphas_off( );
 extern double table_alphas_on( );
 extern double table_betap_off( );
 extern double table_betap_on( );
 extern double table_alphap_off( );
 extern double table_alphap_on( );
 extern double table_betan_off( );
 extern double table_betan_on( );
 extern double table_alphan_off( );
 extern double table_alphan_on( );
 extern double table_betah_off( );
 extern double table_betah_on( );
 extern double table_alphah_off( );
 extern double table_alphah_on( );
 extern double table_betam_off( );
 extern double table_betam_on( );
 extern double table_alpham_off( );
 extern double table_alpham_on( );
 extern double table_veff_off( );
 extern double table_veff_on( );
 extern double veff_off( _threadargsprotocomma_ double );
 extern double veff_on( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define cm cm_LTS
 double cm = 1;
#define offset offset_LTS
 double offset = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "offset_LTS", "ms",
 "cm_LTS", "uF/cm2",
 "duration_LTS", "ms",
 "PRF_LTS", "Hz",
 "ena_LTS", "mV",
 "eca_LTS", "mV",
 "ek_LTS", "mV",
 "eleak_LTS", "mV",
 "gnabar_LTS", "mho/cm2",
 "gkdbar_LTS", "mho/cm2",
 "gmbar_LTS", "mho/cm2",
 "gcabar_LTS", "mho/cm2",
 "gleak_LTS", "mho/cm2",
 "Q_LTS", "nC/cm2",
 "Vmeff_LTS", "mV",
 "iNa_LTS", "mA/cm2",
 "iKd_LTS", "mA/cm2",
 "iM_LTS", "mA/cm2",
 "iCa_LTS", "mA/cm2",
 "iLeak_LTS", "mA/cm2",
 0,0
};
 static double delta_t = 1;
 static double h0 = 0;
 static double m0 = 0;
 static double n0 = 0;
 static double p0 = 0;
 static double s0 = 0;
 static double u0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "offset_LTS", &offset_LTS,
 "cm_LTS", &cm_LTS,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[0]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "6.2.0",
"LTS",
 "duration_LTS",
 "PRF_LTS",
 "DC_LTS",
 "ena_LTS",
 "eca_LTS",
 "ek_LTS",
 "eleak_LTS",
 "gnabar_LTS",
 "gkdbar_LTS",
 "gmbar_LTS",
 "gcabar_LTS",
 "gleak_LTS",
 0,
 "Q_LTS",
 "Vmeff_LTS",
 "iNa_LTS",
 "iKd_LTS",
 "iM_LTS",
 "iCa_LTS",
 "iLeak_LTS",
 "stimon_LTS",
 0,
 "m_LTS",
 "h_LTS",
 "n_LTS",
 "p_LTS",
 "s_LTS",
 "u_LTS",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 47, _prop);
 	/*initialize range parameters*/
 	duration = 100;
 	PRF = 0;
 	DC = 0;
 	ena = 50;
 	eca = 120;
 	ek = -90;
 	eleak = -50;
 	gnabar = 0.05;
 	gkdbar = 0.004;
 	gmbar = 2.8e-005;
 	gcabar = 0.0004;
 	gleak = 1.9e-005;
 	_prop->param = _p;
 	_prop->param_size = 47;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _LTS_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
  hoc_register_prop_size(_mechtype, 47, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 LTS C:/Users/Theo/Documents/PointNICE/PointNICE/neurons/nmodl/LTS.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "LTS neuron";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int interpolate_off(_threadargsprotocomma_ double);
static int interpolate_on(_threadargsprotocomma_ double);
static int stimonoff(_threadargsproto_);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 
static void* _ptable_betau_off = (void*)0;
 
static void* _ptable_betau_on = (void*)0;
 
static void* _ptable_alphau_off = (void*)0;
 
static void* _ptable_alphau_on = (void*)0;
 
static void* _ptable_betas_off = (void*)0;
 
static void* _ptable_betas_on = (void*)0;
 
static void* _ptable_alphas_off = (void*)0;
 
static void* _ptable_alphas_on = (void*)0;
 
static void* _ptable_betap_off = (void*)0;
 
static void* _ptable_betap_on = (void*)0;
 
static void* _ptable_alphap_off = (void*)0;
 
static void* _ptable_alphap_on = (void*)0;
 
static void* _ptable_betan_off = (void*)0;
 
static void* _ptable_betan_on = (void*)0;
 
static void* _ptable_alphan_off = (void*)0;
 
static void* _ptable_alphan_on = (void*)0;
 
static void* _ptable_betah_off = (void*)0;
 
static void* _ptable_betah_on = (void*)0;
 
static void* _ptable_alphah_off = (void*)0;
 
static void* _ptable_alphah_on = (void*)0;
 
static void* _ptable_betam_off = (void*)0;
 
static void* _ptable_betam_on = (void*)0;
 
static void* _ptable_alpham_off = (void*)0;
 
static void* _ptable_alpham_on = (void*)0;
 
static void* _ptable_veff_off = (void*)0;
 
static void* _ptable_veff_on = (void*)0;
 static int _slist1[6], _dlist1[6];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   if ( stimon ) {
     interpolate_on ( _threadargscomma_ Q ) ;
     }
   else {
     interpolate_off ( _threadargscomma_ Q ) ;
     }
   Dm = alpha_m * ( 1.0 - m ) - beta_m * m ;
   Dh = alpha_h * ( 1.0 - h ) - beta_h * h ;
   Dn = alpha_n * ( 1.0 - n ) - beta_n * n ;
   Dp = alpha_p * ( 1.0 - p ) - beta_p * p ;
   Ds = alpha_s * ( 1.0 - s ) - beta_s * s ;
   Du = alpha_u * ( 1.0 - u ) - beta_u * u ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 if ( stimon ) {
   interpolate_on ( _threadargscomma_ Q ) ;
   }
 else {
   interpolate_off ( _threadargscomma_ Q ) ;
   }
 Dm = Dm  / (1. - dt*( ( alpha_m )*( ( ( - 1.0 ) ) ) - ( beta_m )*( 1.0 ) )) ;
 Dh = Dh  / (1. - dt*( ( alpha_h )*( ( ( - 1.0 ) ) ) - ( beta_h )*( 1.0 ) )) ;
 Dn = Dn  / (1. - dt*( ( alpha_n )*( ( ( - 1.0 ) ) ) - ( beta_n )*( 1.0 ) )) ;
 Dp = Dp  / (1. - dt*( ( alpha_p )*( ( ( - 1.0 ) ) ) - ( beta_p )*( 1.0 ) )) ;
 Ds = Ds  / (1. - dt*( ( alpha_s )*( ( ( - 1.0 ) ) ) - ( beta_s )*( 1.0 ) )) ;
 Du = Du  / (1. - dt*( ( alpha_u )*( ( ( - 1.0 ) ) ) - ( beta_u )*( 1.0 ) )) ;
 return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
   if ( stimon ) {
     interpolate_on ( _threadargscomma_ Q ) ;
     }
   else {
     interpolate_off ( _threadargscomma_ Q ) ;
     }
    m = m + (1. - exp(dt*(( alpha_m )*( ( ( - 1.0 ) ) ) - ( beta_m )*( 1.0 ))))*(- ( ( alpha_m )*( ( 1.0 ) ) ) / ( ( alpha_m )*( ( ( - 1.0 ) ) ) - ( beta_m )*( 1.0 ) ) - m) ;
    h = h + (1. - exp(dt*(( alpha_h )*( ( ( - 1.0 ) ) ) - ( beta_h )*( 1.0 ))))*(- ( ( alpha_h )*( ( 1.0 ) ) ) / ( ( alpha_h )*( ( ( - 1.0 ) ) ) - ( beta_h )*( 1.0 ) ) - h) ;
    n = n + (1. - exp(dt*(( alpha_n )*( ( ( - 1.0 ) ) ) - ( beta_n )*( 1.0 ))))*(- ( ( alpha_n )*( ( 1.0 ) ) ) / ( ( alpha_n )*( ( ( - 1.0 ) ) ) - ( beta_n )*( 1.0 ) ) - n) ;
    p = p + (1. - exp(dt*(( alpha_p )*( ( ( - 1.0 ) ) ) - ( beta_p )*( 1.0 ))))*(- ( ( alpha_p )*( ( 1.0 ) ) ) / ( ( alpha_p )*( ( ( - 1.0 ) ) ) - ( beta_p )*( 1.0 ) ) - p) ;
    s = s + (1. - exp(dt*(( alpha_s )*( ( ( - 1.0 ) ) ) - ( beta_s )*( 1.0 ))))*(- ( ( alpha_s )*( ( 1.0 ) ) ) / ( ( alpha_s )*( ( ( - 1.0 ) ) ) - ( beta_s )*( 1.0 ) ) - s) ;
    u = u + (1. - exp(dt*(( alpha_u )*( ( ( - 1.0 ) ) ) - ( beta_u )*( 1.0 ))))*(- ( ( alpha_u )*( ( 1.0 ) ) ) / ( ( alpha_u )*( ( ( - 1.0 ) ) ) - ( beta_u )*( 1.0 ) ) - u) ;
   }
  return 0;
}
 
double veff_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_veff_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_veff_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  veff_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_veff_on ( ) {
	hoc_spec_table(&_ptable_veff_on, 1);
	return 0.;
}
 
static void _hoc_table_veff_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_veff_on (  );
 hoc_retpushx(_r);
}
 
double veff_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_veff_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_veff_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  veff_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_veff_off ( ) {
	hoc_spec_table(&_ptable_veff_off, 1);
	return 0.;
}
 
static void _hoc_table_veff_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_veff_off (  );
 hoc_retpushx(_r);
}
 
double alpham_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alpham_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alpham_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alpham_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alpham_on ( ) {
	hoc_spec_table(&_ptable_alpham_on, 1);
	return 0.;
}
 
static void _hoc_table_alpham_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alpham_on (  );
 hoc_retpushx(_r);
}
 
double alpham_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alpham_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alpham_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alpham_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alpham_off ( ) {
	hoc_spec_table(&_ptable_alpham_off, 1);
	return 0.;
}
 
static void _hoc_table_alpham_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alpham_off (  );
 hoc_retpushx(_r);
}
 
double betam_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betam_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betam_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betam_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betam_on ( ) {
	hoc_spec_table(&_ptable_betam_on, 1);
	return 0.;
}
 
static void _hoc_table_betam_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betam_on (  );
 hoc_retpushx(_r);
}
 
double betam_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betam_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betam_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betam_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betam_off ( ) {
	hoc_spec_table(&_ptable_betam_off, 1);
	return 0.;
}
 
static void _hoc_table_betam_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betam_off (  );
 hoc_retpushx(_r);
}
 
double alphah_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alphah_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alphah_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphah_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alphah_on ( ) {
	hoc_spec_table(&_ptable_alphah_on, 1);
	return 0.;
}
 
static void _hoc_table_alphah_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alphah_on (  );
 hoc_retpushx(_r);
}
 
double alphah_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alphah_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alphah_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphah_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alphah_off ( ) {
	hoc_spec_table(&_ptable_alphah_off, 1);
	return 0.;
}
 
static void _hoc_table_alphah_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alphah_off (  );
 hoc_retpushx(_r);
}
 
double betah_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betah_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betah_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betah_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betah_on ( ) {
	hoc_spec_table(&_ptable_betah_on, 1);
	return 0.;
}
 
static void _hoc_table_betah_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betah_on (  );
 hoc_retpushx(_r);
}
 
double betah_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betah_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betah_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betah_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betah_off ( ) {
	hoc_spec_table(&_ptable_betah_off, 1);
	return 0.;
}
 
static void _hoc_table_betah_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betah_off (  );
 hoc_retpushx(_r);
}
 
double alphan_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alphan_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alphan_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphan_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alphan_on ( ) {
	hoc_spec_table(&_ptable_alphan_on, 1);
	return 0.;
}
 
static void _hoc_table_alphan_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alphan_on (  );
 hoc_retpushx(_r);
}
 
double alphan_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alphan_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alphan_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphan_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alphan_off ( ) {
	hoc_spec_table(&_ptable_alphan_off, 1);
	return 0.;
}
 
static void _hoc_table_alphan_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alphan_off (  );
 hoc_retpushx(_r);
}
 
double betan_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betan_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betan_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betan_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betan_on ( ) {
	hoc_spec_table(&_ptable_betan_on, 1);
	return 0.;
}
 
static void _hoc_table_betan_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betan_on (  );
 hoc_retpushx(_r);
}
 
double betan_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betan_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betan_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betan_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betan_off ( ) {
	hoc_spec_table(&_ptable_betan_off, 1);
	return 0.;
}
 
static void _hoc_table_betan_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betan_off (  );
 hoc_retpushx(_r);
}
 
double alphap_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alphap_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alphap_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphap_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alphap_on ( ) {
	hoc_spec_table(&_ptable_alphap_on, 1);
	return 0.;
}
 
static void _hoc_table_alphap_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alphap_on (  );
 hoc_retpushx(_r);
}
 
double alphap_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alphap_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alphap_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphap_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alphap_off ( ) {
	hoc_spec_table(&_ptable_alphap_off, 1);
	return 0.;
}
 
static void _hoc_table_alphap_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alphap_off (  );
 hoc_retpushx(_r);
}
 
double betap_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betap_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betap_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betap_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betap_on ( ) {
	hoc_spec_table(&_ptable_betap_on, 1);
	return 0.;
}
 
static void _hoc_table_betap_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betap_on (  );
 hoc_retpushx(_r);
}
 
double betap_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betap_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betap_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betap_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betap_off ( ) {
	hoc_spec_table(&_ptable_betap_off, 1);
	return 0.;
}
 
static void _hoc_table_betap_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betap_off (  );
 hoc_retpushx(_r);
}
 
double alphas_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alphas_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alphas_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphas_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alphas_on ( ) {
	hoc_spec_table(&_ptable_alphas_on, 1);
	return 0.;
}
 
static void _hoc_table_alphas_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alphas_on (  );
 hoc_retpushx(_r);
}
 
double alphas_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alphas_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alphas_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphas_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alphas_off ( ) {
	hoc_spec_table(&_ptable_alphas_off, 1);
	return 0.;
}
 
static void _hoc_table_alphas_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alphas_off (  );
 hoc_retpushx(_r);
}
 
double betas_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betas_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betas_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betas_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betas_on ( ) {
	hoc_spec_table(&_ptable_betas_on, 1);
	return 0.;
}
 
static void _hoc_table_betas_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betas_on (  );
 hoc_retpushx(_r);
}
 
double betas_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betas_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betas_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betas_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betas_off ( ) {
	hoc_spec_table(&_ptable_betas_off, 1);
	return 0.;
}
 
static void _hoc_table_betas_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betas_off (  );
 hoc_retpushx(_r);
}
 
double alphau_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alphau_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alphau_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphau_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alphau_on ( ) {
	hoc_spec_table(&_ptable_alphau_on, 1);
	return 0.;
}
 
static void _hoc_table_alphau_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alphau_on (  );
 hoc_retpushx(_r);
}
 
double alphau_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_alphau_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_alphau_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alphau_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_alphau_off ( ) {
	hoc_spec_table(&_ptable_alphau_off, 1);
	return 0.;
}
 
static void _hoc_table_alphau_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_alphau_off (  );
 hoc_retpushx(_r);
}
 
double betau_on ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betau_on, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betau_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betau_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betau_on ( ) {
	hoc_spec_table(&_ptable_betau_on, 1);
	return 0.;
}
 
static void _hoc_table_betau_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betau_on (  );
 hoc_retpushx(_r);
}
 
double betau_off ( _threadargsprotocomma_ double _lx ) {
 double _arg[1];
 _arg[0] = _lx;
 return hoc_func_table(_ptable_betau_off, 1, _arg);
 }
/*  }
  */
 
static void _hoc_betau_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betau_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 double table_betau_off ( ) {
	hoc_spec_table(&_ptable_betau_off, 1);
	return 0.;
}
 
static void _hoc_table_betau_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  table_betau_off (  );
 hoc_retpushx(_r);
}
 
static int  interpolate_on ( _threadargsprotocomma_ double _lQ ) {
   alpha_m = alpham_on ( _threadargscomma_ _lQ ) ;
   beta_m = betam_on ( _threadargscomma_ _lQ ) ;
   alpha_h = alphah_on ( _threadargscomma_ _lQ ) ;
   beta_h = betah_on ( _threadargscomma_ _lQ ) ;
   alpha_n = alphan_on ( _threadargscomma_ _lQ ) ;
   beta_n = betan_on ( _threadargscomma_ _lQ ) ;
   alpha_p = alphap_on ( _threadargscomma_ _lQ ) ;
   beta_p = betap_on ( _threadargscomma_ _lQ ) ;
   alpha_s = alphas_on ( _threadargscomma_ _lQ ) ;
   beta_s = betas_on ( _threadargscomma_ _lQ ) ;
   alpha_u = alphau_on ( _threadargscomma_ _lQ ) ;
   beta_u = betau_on ( _threadargscomma_ _lQ ) ;
    return 0; }
 
static void _hoc_interpolate_on(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 interpolate_on ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int  interpolate_off ( _threadargsprotocomma_ double _lv ) {
   alpha_m = alpham_off ( _threadargscomma_ _lv ) ;
   beta_m = betam_off ( _threadargscomma_ _lv ) ;
   alpha_h = alphah_off ( _threadargscomma_ _lv ) ;
   beta_h = betah_off ( _threadargscomma_ _lv ) ;
   alpha_n = alphan_off ( _threadargscomma_ _lv ) ;
   beta_n = betan_off ( _threadargscomma_ _lv ) ;
   alpha_p = alphap_off ( _threadargscomma_ Q ) ;
   beta_p = betap_off ( _threadargscomma_ Q ) ;
   alpha_s = alphas_off ( _threadargscomma_ Q ) ;
   beta_s = betas_off ( _threadargscomma_ Q ) ;
   alpha_u = alphau_off ( _threadargscomma_ Q ) ;
   beta_u = betau_off ( _threadargscomma_ Q ) ;
    return 0; }
 
static void _hoc_interpolate_off(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 interpolate_off ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int  stimonoff ( _threadargsproto_ ) {
   if ( t < duration - dt ) {
     if ( tint >= 1000.0 / PRF ) {
       stimon = 1.0 ;
       tint = 0.0 ;
       }
     else if ( tint <= DC * 1000.0 / PRF ) {
       stimon = 1.0 ;
       }
     else {
       stimon = 0.0 ;
       }
     tint = tint + dt ;
     }
   else {
     stimon = 0.0 ;
     }
    return 0; }
 
static void _hoc_stimonoff(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 stimonoff ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 6;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 6; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
  n = n0;
  p = p0;
  s = s0;
  u = u0;
 {
   m = alpham_off ( _threadargscomma_ v ) / ( alpham_off ( _threadargscomma_ v ) + betam_off ( _threadargscomma_ v ) ) ;
   h = alphah_off ( _threadargscomma_ v ) / ( alphah_off ( _threadargscomma_ v ) + betah_off ( _threadargscomma_ v ) ) ;
   n = alphan_off ( _threadargscomma_ v ) / ( alphan_off ( _threadargscomma_ v ) + betan_off ( _threadargscomma_ v ) ) ;
   p = alphap_off ( _threadargscomma_ v ) / ( alphap_off ( _threadargscomma_ v ) + betap_off ( _threadargscomma_ v ) ) ;
   s = alphas_off ( _threadargscomma_ v ) / ( alphas_off ( _threadargscomma_ v ) + betas_off ( _threadargscomma_ v ) ) ;
   u = alphau_off ( _threadargscomma_ v ) / ( alphau_off ( _threadargscomma_ v ) + betau_off ( _threadargscomma_ v ) ) ;
   tint = 0.0 ;
   stimon = 0.0 ;
   PRF = PRF / 2.0 ;
   }
 
}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   Q = v * cm ;
   stimonoff ( _threadargs_ ) ;
   if ( stimon ) {
     Vmeff = veff_on ( _threadargscomma_ Q ) ;
     }
   else {
     Vmeff = veff_off ( _threadargscomma_ Q ) ;
     }
   iNa = gnabar * m * m * m * h * ( Vmeff - ena ) ;
   iKd = gkdbar * n * n * n * n * ( Vmeff - ek ) ;
   iM = gmbar * p * ( Vmeff - ek ) ;
   iCa = gcabar * s * s * u * ( Vmeff - eca ) ;
   iLeak = gleak * ( Vmeff - eleak ) ;
   }
 _current += iNa;
 _current += iKd;
 _current += iM;
 _current += iCa;
 _current += iLeak;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {   states(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(m) - _p;  _dlist1[0] = &(Dm) - _p;
 _slist1[1] = &(h) - _p;  _dlist1[1] = &(Dh) - _p;
 _slist1[2] = &(n) - _p;  _dlist1[2] = &(Dn) - _p;
 _slist1[3] = &(p) - _p;  _dlist1[3] = &(Dp) - _p;
 _slist1[4] = &(s) - _p;  _dlist1[4] = &(Ds) - _p;
 _slist1[5] = &(u) - _p;  _dlist1[5] = &(Du) - _p;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif
