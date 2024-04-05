#ifndef DISLOC3D_H_
#define DISLOC3D_H_

#ifdef __cplusplus
extern "C"
#endif

#define STARS "+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

// void disloc3d(double (*models)[10], int nmodel, double (*obs)[3], int nobs, double mu, double nu, double (*U)[3], double (*D)[9], double (*S)[9], int flags[nobs][nmodel]);

void disloc3d(double *models, int nmodel, double *obss, int nobs, double mu, double nu, double *U, double *D, double *S, int *flags);




#endif

