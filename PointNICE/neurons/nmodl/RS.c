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
 
#define nrn_init _nrn_init__RS
#define _nrn_initial _nrn_initial__RS
#define nrn_cur _nrn_cur__RS
#define _nrn_current _nrn_current__RS
#define nrn_jacob _nrn_jacob__RS
#define nrn_state _nrn_state__RS
#define _net_receive _net_receive__RS 
#define interpolate_off interpolate_off__RS 
#define interpolate_on interpolate_on__RS 
#define states states__RS 
#define stimonoff stimonoff__RS 
 
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
#define ek _p[4]
#define eleak _p[5]
#define gnabar _p[6]
#define gkdbar _p[7]
#define gmbar _p[8]
#define gleak _p[9]
#define Q _p[10]
#define Vmeff _p[11]
#define iNa _p[12]
#define iKd _p[13]
#define iM _p[14]
#define iLeak _p[15]
#define stimon _p[16]
#define m _p[17]
#define h _p[18]
#define n _p[19]
#define p _p[20]
#define Dm _p[21]
#define Dh _p[22]
#define Dn _p[23]
#define Dp _p[24]
#define alpha_h _p[25]
#define beta_h _p[26]
#define alpha_m _p[27]
#define beta_m _p[28]
#define alpha_n _p[29]
#define beta_n _p[30]
#define alpha_p _p[31]
#define beta_p _p[32]
#define tint _p[33]
#define v _p[34]
#define _g _p[35]
 
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
 static void _hoc_alphap_on(void);
 static void _hoc_alphan_on(void);
 static void _hoc_alphah_on(void);
 static void _hoc_alpham_on(void);
 static void _hoc_alphap_off(void);
 static void _hoc_alphan_off(void);
 static void _hoc_alphah_off(void);
 static void _hoc_alpham_off(void);
 static void _hoc_betap_on(void);
 static void _hoc_betan_on(void);
 static void _hoc_betah_on(void);
 static void _hoc_betam_on(void);
 static void _hoc_betap_off(void);
 static void _hoc_betan_off(void);
 static void _hoc_betah_off(void);
 static void _hoc_betam_off(void);
 static void _hoc_interpolate_off(void);
 static void _hoc_interpolate_on(void);
 static void _hoc_stimonoff(void);
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
 "setdata_RS", _hoc_setdata,
 "alphap_on_RS", _hoc_alphap_on,
 "alphan_on_RS", _hoc_alphan_on,
 "alphah_on_RS", _hoc_alphah_on,
 "alpham_on_RS", _hoc_alpham_on,
 "alphap_off_RS", _hoc_alphap_off,
 "alphan_off_RS", _hoc_alphan_off,
 "alphah_off_RS", _hoc_alphah_off,
 "alpham_off_RS", _hoc_alpham_off,
 "betap_on_RS", _hoc_betap_on,
 "betan_on_RS", _hoc_betan_on,
 "betah_on_RS", _hoc_betah_on,
 "betam_on_RS", _hoc_betam_on,
 "betap_off_RS", _hoc_betap_off,
 "betan_off_RS", _hoc_betan_off,
 "betah_off_RS", _hoc_betah_off,
 "betam_off_RS", _hoc_betam_off,
 "interpolate_off_RS", _hoc_interpolate_off,
 "interpolate_on_RS", _hoc_interpolate_on,
 "stimonoff_RS", _hoc_stimonoff,
 "table_betap_off_RS", _hoc_table_betap_off,
 "table_betap_on_RS", _hoc_table_betap_on,
 "table_alphap_off_RS", _hoc_table_alphap_off,
 "table_alphap_on_RS", _hoc_table_alphap_on,
 "table_betan_off_RS", _hoc_table_betan_off,
 "table_betan_on_RS", _hoc_table_betan_on,
 "table_alphan_off_RS", _hoc_table_alphan_off,
 "table_alphan_on_RS", _hoc_table_alphan_on,
 "table_betah_off_RS", _hoc_table_betah_off,
 "table_betah_on_RS", _hoc_table_betah_on,
 "table_alphah_off_RS", _hoc_table_alphah_off,
 "table_alphah_on_RS", _hoc_table_alphah_on,
 "table_betam_off_RS", _hoc_table_betam_off,
 "table_betam_on_RS", _hoc_table_betam_on,
 "table_alpham_off_RS", _hoc_table_alpham_off,
 "table_alpham_on_RS", _hoc_table_alpham_on,
 "table_veff_off_RS", _hoc_table_veff_off,
 "table_veff_on_RS", _hoc_table_veff_on,
 "veff_off_RS", _hoc_veff_off,
 "veff_on_RS", _hoc_veff_on,
 0, 0
};
#define alphap_on alphap_on_RS
#define alphan_on alphan_on_RS
#define alphah_on alphah_on_RS
#define alpham_on alpham_on_RS
#define alphap_off alphap_off_RS
#define alphan_off alphan_off_RS
#define alphah_off alphah_off_RS
#define alpham_off alpham_off_RS
#define betap_on betap_on_RS
#define betan_on betan_on_RS
#define betah_on betah_on_RS
#define betam_on betam_on_RS
#define betap_off betap_off_RS
#define betan_off betan_off_RS
#define betah_off betah_off_RS
#define betam_off betam_off_RS
#define table_betap_off table_betap_off_RS
#define table_betap_on table_betap_on_RS
#define table_alphap_off table_alphap_off_RS
#define table_alphap_on table_alphap_on_RS
#define table_betan_off table_betan_off_RS
#define table_betan_on table_betan_on_RS
#define table_alphan_off table_alphan_off_RS
#define table_alphan_on table_alphan_on_RS
#define table_betah_off table_betah_off_RS
#define table_betah_on table_betah_on_RS
#define table_alphah_off table_alphah_off_RS
#define table_alphah_on table_alphah_on_RS
#define table_betam_off table_betam_off_RS
#define table_betam_on table_betam_on_RS
#define table_alpham_off table_alpham_off_RS
#define table_alpham_on table_alpham_on_RS
#define table_veff_off table_veff_off_RS
#define table_veff_on table_veff_on_RS
#define veff_off veff_off_RS
#define veff_on veff_on_RS
 extern double alphap_on( _threadargsprotocomma_ double );
 extern double alphan_on( _threadargsprotocomma_ double );
 extern double alphah_on( _threadargsprotocomma_ double );
 extern double alpham_on( _threadargsprotocomma_ double );
 extern double alphap_off( _threadargsprotocomma_ double );
 extern double alphan_off( _threadargsprotocomma_ double );
 extern double alphah_off( _threadargsprotocomma_ double );
 extern double alpham_off( _threadargsprotocomma_ double );
 extern double betap_on( _threadargsprotocomma_ double );
 extern double betan_on( _threadargsprotocomma_ double );
 extern double betah_on( _threadargsprotocomma_ double );
 extern double betam_on( _threadargsprotocomma_ double );
 extern double betap_off( _threadargsprotocomma_ double );
 extern double betan_off( _threadargsprotocomma_ double );
 extern double betah_off( _threadargsprotocomma_ double );
 extern double betam_off( _threadargsprotocomma_ double );
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
#define cm cm_RS
 double cm = 1;
#define offset offset_RS
 double offset = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "offset_RS", "ms",
 "cm_RS", "uF/cm2",
 "duration_RS", "ms",
 "PRF_RS", "Hz",
 "ena_RS", "mV",
 "ek_RS", "mV",
 "eleak_RS", "mV",
 "gnabar_RS", "mho/cm2",
 "gkdbar_RS", "mho/cm2",
 "gmbar_RS", "mho/cm2",
 "gleak_RS", "mho/cm2",
 "Q_RS", "nC/cm2",
 "Vmeff_RS", "mV",
 "iNa_RS", "mA/cm2",
 "iKd_RS", "mA/cm2",
 "iM_RS", "mA/cm2",
 "iLeak_RS", "mA/cm2",
 0,0
};
 static double delta_t = 1;
 static double h0 = 0;
 static double m0 = 0;
 static double n0 = 0;
 static double p0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "offset_RS", &offset_RS,
 "cm_RS", &cm_RS,
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
"RS",
 "duration_RS",
 "PRF_RS",
 "DC_RS",
 "ena_RS",
 "ek_RS",
 "eleak_RS",
 "gnabar_RS",
 "gkdbar_RS",
 "gmbar_RS",
 "gleak_RS",
 0,
 "Q_RS",
 "Vmeff_RS",
 "iNa_RS",
 "iKd_RS",
 "iM_RS",
 "iLeak_RS",
 "stimon_RS",
 0,
 "m_RS",
 "h_RS",
 "n_RS",
 "p_RS",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 36, _prop);
 	/*initialize range parameters*/
 	duration = 100;
 	PRF = 0;
 	DC = 0;
 	ena = 50;
 	ek = -90;
 	eleak = -70.3;
 	gnabar = 0.056;
 	gkdbar = 0.006;
 	gmbar = 7.5e-005;
 	gleak = 2.05e-005;
 	_prop->param = _p;
 	_prop->param_size = 36;
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

 void _RS_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
  hoc_register_prop_size(_mechtype, 36, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 RS C:/Users/Theo/Documents/PointNICE/PointNICE/neurons/nmodl/RS.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "RS neuron";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int interpolate_off(_threadargsprotocomma_ double);
static int interpolate_on(_threadargsprotocomma_ double);
static int stimonoff(_threadargsproto_);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 
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
 static int _slist1[4], _dlist1[4];
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
 
static int  interpolate_on ( _threadargsprotocomma_ double _lQ ) {
   alpha_m = alpham_on ( _threadargscomma_ _lQ ) ;
   beta_m = betam_on ( _threadargscomma_ _lQ ) ;
   alpha_h = alphah_on ( _threadargscomma_ _lQ ) ;
   beta_h = betah_on ( _threadargscomma_ _lQ ) ;
   alpha_n = alphan_on ( _threadargscomma_ _lQ ) ;
   beta_n = betan_on ( _threadargscomma_ _lQ ) ;
   alpha_p = alphap_on ( _threadargscomma_ _lQ ) ;
   beta_p = betap_on ( _threadargscomma_ _lQ ) ;
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
 
static int _ode_count(int _type){ return 4;}
 
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
	for (_i=0; _i < 4; ++_i) {
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
 {
   m = alpham_off ( _threadargscomma_ v ) / ( alpham_off ( _threadargscomma_ v ) + betam_off ( _threadargscomma_ v ) ) ;
   h = alphah_off ( _threadargscomma_ v ) / ( alphah_off ( _threadargscomma_ v ) + betah_off ( _threadargscomma_ v ) ) ;
   n = alphan_off ( _threadargscomma_ v ) / ( alphan_off ( _threadargscomma_ v ) + betan_off ( _threadargscomma_ v ) ) ;
   p = alphap_off ( _threadargscomma_ v ) / ( alphap_off ( _threadargscomma_ v ) + betap_off ( _threadargscomma_ v ) ) ;
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
   iLeak = gleak * ( Vmeff - eleak ) ;
   }
 _current += iNa;
 _current += iKd;
 _current += iM;
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
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif
