#ifndef OKADA_H_
#define OKADA_H_

int dc3d0_(double *alpha, double *x, double *y, double *z__, double *depth,
           double *dip, double *pot1, double *pot2, double *pot3, double *pot4,
           double *ux, double *uy, double *uz, double *uxx, double *uyx,
           double *uzx, double *uxy, double *uyy, double *uzy, double *uxz,
           double *uyz, double *uzz, int *iret);

int dc3d_(double *alpha, double *x, double *y, double *z__, double *depth,
          double *dip, double *al1, double *al2, double *aw1, double *aw2,
          double *disl1, double *disl2, double *disl3, double *ux, double *uy,
          double *uz, double *uxx, double *uyx, double *uzx, double *uxy,
          double *uyy, double *uzy, double *uxz, double *uyz, double *uzz,
          int *iret);

// void disloc3d(double (*models)[10], int nmodel, double (*obs)[3], int nobs,
// double mu, double nu, double (*U)[3], double (*D)[9], double (*S)[9], int
// flags[nobs][nmodel]);

void okada_disloc3d(double *models, int nmodel, double *obss, int nobs, double mu,
              double nu, double *U, double *D, double *S, double *E,
              int *flags);
#endif
