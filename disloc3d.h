#ifndef DISLOC3D_H_
#define DISLOC3D_H_

#ifdef __cplusplus
extern "C"
#endif

#define STARS "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

void disloc3d(double *models,         int nmodel,
              double *observations,   int nobs,
	      double mu,              double nu,
	      double *U,              double *D,           double *S,
	      int *flags );


#endif

