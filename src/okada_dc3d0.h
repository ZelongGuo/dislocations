#ifndef DC3D0_H_
#define DC3D0_H_

#ifdef __cplusplus
extern "C"
#endif

#define STARS "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

int dc3d0_(double *alpha,
	   double *x,     double *y,    double *z__,
	   double *depth, double *dip,
	   double *pot1,  double *pot2, double *pot3, double *pot4, 
	   double *ux,    double *uy,   double *uz,
	   double *uxx,   double *uyx,  double *uzx,
	   double *uxy,   double *uyy,  double *uzy,
	   double *uxz,   double *uyz,  double *uzz,  int *iret);

#endif
