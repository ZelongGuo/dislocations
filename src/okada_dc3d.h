#ifndef DC3D_H_
#define DC3D_H_

#ifdef __cplusplus
extern "C"
#endif

#define STARS "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

int dc3d_(double *alpha,
	  double *x,     double *y,     double *z__,
	  double *depth, double *dip,
	  double *al1,   double *al2,   double *aw1, double *aw2,
	  double *disl1, double *disl2, double *disl3,
	  double *ux,    double *uy,    double *uz,
	  double *uxx,   double *uyx,   double *uzx,
	  double *uxy,   double *uyy,   double *uzy,
	  double *uxz,   double *uyz,   double *uzz, int *iret);


#endif
