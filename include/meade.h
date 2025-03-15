// #include <random>
#ifndef MEADE_H_
#define MEADE_H_

// IEEE 754 double-precision floating point storage for doubel comparison
#define EPSILON 1e-12  // Error Threshold

/* ------------------------------------------------------------ */
/* meade_advs.c */
void advs(double b_y1, double y2, double y3, double a, double b, double nu,
         double B1, double B2, double B3, double *e11, double *e22, double *e33,
         double *e12, double *e13, double *e23);

/* ------------------------------------------------------------ */
/* meade_dc3d.c */

void CalTriDisps(const double sx, const double sy, const double sz, double *x,
                 double *y, double *z, const double pr, const double ss,
                 const double ts, const double ds, double *U);

void CalTriStrains(const double sx, const double sy, const double sz, double *x,
                   double *y, double *z, const double pr, const double ss,
                   const double ts, const double ds, double *E);

void CalTriStress(double *E, double lamda, double mu, double *S);

/* ------------------------------------------------------------ */
/* meade_disloc3d.c */
void meade_disloc3d(double *models, int nmodel, double *obss, int nobs,
                    double mu, double nu, double *U, double *S, double *E);

#endif
