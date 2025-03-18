#ifndef MEHDI_H_
#define MEHDI_H_

// Define PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// IEEE 754 double-precision floating point storage for doubel comparison
#define EPSILON 1e-12  // Error Threshold

/* ------------------    From TDUtls.c for all    ------------------ */
void CoordTrans(double x1, double x2, double x3, double A[9],
                double *X1, double *X2, double *X3);

void trimodefinder(double x, double y, double z, double p1[2], double p2[2], double p3[2], int *trimode);

/* ---------------- From TDdisp_utls.c for TDdispFS.c -------------- */
void AngDisDisp(double x, double y, double z, double alpha,
                double bx, double by, double bz, double nu,
                double *u, double *v, double *w);

void TDSetupD(double x, double y, double z, double alpha,
              double bx, double by, double bz, double nu,
              double TriVertex[3], double SideVec[3],
              double *u, double *v, double *w);

void TDdispFS(double X, double Y, double Z, double P1[3], double P2[3], double P3[3],
              double Ss, double Ds, double Ts, double nu,
              double *ue, double *un, double *uv);

/* -------------- From TDdispHS_utls.c for TDdispHS.c -------------- */
void TDdispFS4HS(double X, double Y, double Z, double P1[3], double P2[3], double P3[3],
              double Ss, double Ds, double Ts, double nu,
              double *ue, double *un, double *uv);

void AngDisDispFSC(double y1, double y2, double y3, double beta,
                   double b1, double b2, double b3, double nu, double a,
                   double *v1, double *v2, double *v3);

void AngSetupFSC(double X, double Y, double Z, double bX, double bY, double bZ,
                 double PA[3], double PB[3], double nu, double *ue, double *un, double *uv);

void TDdisp_HarFunc(double X, double Y, double Z, double P1[3], double P2[3], double P3[3], 
                    double Ss, double Ds, double Ts, double nu, double *ue, double *un, double *uv);

void TDdispHS(double X, double Y, double Z, double P1[3], double P2[3], double P3[3],
              double Ss, double Ds, double Ts, double nu, double *ue, double *un, double *uv);

/* ---------------- From TDstress_utls.c for TDstressFS.c -------------- */
void AngDisStrain(double x, double y, double z, double alpha, double bx, double by, double bz, double nu,
                  double *Exx, double *Eyy, double *Ezz, double *Exy, double *Exz, double *Eyz);

void TDSetupS(double x, double y, double z, double alpha, double bx, double by, double bz, double nu,
              double TriVertex[3], double SideVec[3], double *exx, double *eyy, double *ezz, double *exy,
              double *exz, double *eyz);

void TensTrans(double Txx1, double Tyy1, double Tzz1, double Txy1, double Txz1, double Tyz1,
               double A[9], double *Txx2, double *Tyy2, double *Tzz2, double *Txy2, double *Txz2, double *Tyz2);

void TDstressFS(double X, double Y, double Z, double P1[3], double P2[3], double P3[3],
                double Ss, double Ds, double Ts, double mu, double lambda,
                double Stress[6], double Strain[6]);

/* -------------- From TDstressHS_utls.c for TDstressHS.c -------------- */
void TDstressFS4HS(double X, double Y, double Z, double P1[3], double P2[3],
                double P3[3], double Ss, double Ds, double Ts, double mu,
                double lambda, double Stress[6], double Strain[6]);

void AngDisStrainFSC(double y1, double y2, double y3, double beta, double b1, double b2, double b3, 
                     double nu, double a, double *v11, double *v22, double *v33, double *v12,
                     double *v13, double *v23);

void AngSetupFSC_S(double X, double Y, double Z, double bX, double bY, double bZ,
                   double PA[3], double PB[3], double mu, double lambda,
                   double Stress[6], double Strain[6]);

void TDstress_HarFunc(double X, double Y, double Z, double P1[3], double P2[3], double P3[3],
                      double Ss, double Ds, double Ts, double mu, double lambda,
                      double Stress[6], double Strain[6]);

void TDstressHS(double X, double Y, double Z, double P1[3], double P2[3], double P3[3],
                double Ss, double Ds, double Ts, double mu, double lambda,
                double Stress[6], double Strain[6]);
#endif
