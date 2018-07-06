#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _FS_reg();
extern void _LTS_reg();
extern void _RE_reg();
extern void _RS_reg();

modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," FS.mod");
fprintf(stderr," LTS.mod");
fprintf(stderr," RE.mod");
fprintf(stderr," RS.mod");
fprintf(stderr, "\n");
    }
_FS_reg();
_LTS_reg();
_RE_reg();
_RS_reg();
}
