#include <random>
#ifdef __cplusplus
extern "C"
#endif

#define STARS                                                                  \
  "+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+" \
  "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"

void advs(double b_y1, double y2, double y3, double a, double b, double nu,
         double B1, double B2, double B3, double *e11, double *e22, double *e33,
         double *e12, double *e13, double *e23);

void CalTriDisps(const double sx, const double sy, const double sz, double *x,
                 double *y, double *z, const double pr, const double ss,
                 const double ts, const double ds, double *U);

void CalTriStrains(const double sx, const double sy, const double sz, double *x,
                   double *y, double *z, const double pr, const double ss,
                   const double ts, const double ds, double *E);

void CalTriStress(double *E, double lambda, double mu, double *S);
