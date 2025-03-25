#include "mehdi.h"
#include <math.h>
#include <stdlib.h>

/*---------------------------------------------------------
 *  Main function used for calculating stress in full space.
 *
 *  Referring to the Matlab codes of Mehdi
 *
 *  Author: Zelong Guo
 *  03.2025, @ Potsdam, Germany
 *  zelong.guo@outlook.com
 *
 * ---------------------------------------------------------*/

void TDstressFS(double X, double Y, double Z, double P1[3], double P2[3],
                double P3[3], double Ss, double Ds, double Ts, double mu,
                double lambda, double Stress[6], double Strain[6]) {
    // Calculate Poisson's ratio
    double nu = 1.0 / (1.0 + lambda / mu) / 2.0;

    // Slip components
    double bx = Ts; // Tensile-slip
    double by = Ss; // Strike-slip
    double bz = Ds; // Dip-slip

    // Calculate unit normal, strike, and dip vectors
    double Vnorm[3], Vstrike[3], Vdip[3];
    Vnorm[0] =
        (P2[1] - P1[1]) * (P3[2] - P1[2]) - (P2[2] - P1[2]) * (P3[1] - P1[1]);
    Vnorm[1] =
        (P2[2] - P1[2]) * (P3[0] - P1[0]) - (P2[0] - P1[0]) * (P3[2] - P1[2]);
    Vnorm[2] =
        (P2[0] - P1[0]) * (P3[1] - P1[1]) - (P2[1] - P1[1]) * (P3[0] - P1[0]);
    double norm_Vnorm =
        sqrt(Vnorm[0] * Vnorm[0] + Vnorm[1] * Vnorm[1] + Vnorm[2] * Vnorm[2]);
    Vnorm[0] /= norm_Vnorm;
    Vnorm[1] /= norm_Vnorm;
    Vnorm[2] /= norm_Vnorm;

    /* Enforce normVec z component upward for normal situations */
    if (Vnorm[2] < 0) {
        for (int i = 0; i < 3; i++) {
            Vnorm[i] = -Vnorm[i];
            double temp = P2[i];
            P2[i] = P3[i];
            P3[i] = temp;
        }
    }

    double eY[3] = {0.0, 1.0, 0.0};
    double eZ[3] = {0.0, 0.0, 1.0};
    Vstrike[0] = eZ[1] * Vnorm[2] - eZ[2] * Vnorm[1];
    Vstrike[1] = eZ[2] * Vnorm[0] - eZ[0] * Vnorm[2];
    Vstrike[2] = eZ[0] * Vnorm[1] - eZ[1] * Vnorm[0];
    double norm_Vstrike =
        sqrt(Vstrike[0] * Vstrike[0] + Vstrike[1] * Vstrike[1] +
             Vstrike[2] * Vstrike[2]);
    if (norm_Vstrike == 0.0) {
        Vstrike[0] = eY[0] * Vnorm[2];
        Vstrike[1] = eY[1] * Vnorm[2];
        Vstrike[2] = eY[2] * Vnorm[2];
        norm_Vstrike = sqrt(Vstrike[0] * Vstrike[0] + Vstrike[1] * Vstrike[1] +
                            Vstrike[2] * Vstrike[2]);
    }
    Vstrike[0] /= norm_Vstrike;
    Vstrike[1] /= norm_Vstrike;
    Vstrike[2] /= norm_Vstrike;

    Vdip[0] = Vnorm[1] * Vstrike[2] - Vnorm[2] * Vstrike[1];
    Vdip[1] = Vnorm[2] * Vstrike[0] - Vnorm[0] * Vstrike[2];
    Vdip[2] = Vnorm[0] * Vstrike[1] - Vnorm[1] * Vstrike[0];

    // Transformation matrix A (3x3 matrix stored as 1D array in row-major order)
    double A[9] = {
        Vnorm[0],   Vnorm[1],   Vnorm[2],
        Vstrike[0], Vstrike[1], Vstrike[2],
        Vdip[0],    Vdip[1],    Vdip[2]
    };

    // Transform coordinates from EFCS to TDCS
    double x, y, z;
    CoordTrans(X - P2[0], Y - P2[1], Z - P2[2], A, &x, &y, &z);

    double p1[3], p2[3] = {0.0, 0.0, 0.0}, p3[3];
    CoordTrans(P1[0] - P2[0], P1[1] - P2[1], P1[2] - P2[2], A, &p1[0], &p1[1],
               &p1[2]);
    CoordTrans(P3[0] - P2[0], P3[1] - P2[1], P3[2] - P2[2], A, &p3[0], &p3[1],
               &p3[2]);

    // Calculate unit vectors along TD sides in TDCS
    double e12[3], e13[3], e23[3];
    for (int i = 0; i < 3; i++) {
        e12[i] = p2[i] - p1[i];
        e13[i] = p3[i] - p1[i];
        e23[i] = p3[i] - p2[i];
    }
    double norm_e12 = sqrt(e12[0] * e12[0] + e12[1] * e12[1] + e12[2] * e12[2]);
    double norm_e13 = sqrt(e13[0] * e13[0] + e13[1] * e13[1] + e13[2] * e13[2]);
    double norm_e23 = sqrt(e23[0] * e23[0] + e23[1] * e23[1] + e23[2] * e23[2]);
    for (int i = 0; i < 3; i++) {
        e12[i] /= norm_e12;
        e13[i] /= norm_e13;
        e23[i] /= norm_e23;
    }

    // Calculate TD angles
    double A_angle = acos(e12[0] * e13[0] + e12[1] * e13[1] + e12[2] * e13[2]);
    double B = acos(-(e12[0] * e23[0] + e12[1] * e23[1] + e12[2] * e23[2]));
    double C = acos(e23[0] * e13[0] + e23[1] * e13[1] + e23[2] * e13[2]);

    // Determine artefact-free configuration
    int Trimode;
    double p1_23[2] = {p1[1], p1[2]};
    double p2_23[2] = {p2[1], p2[2]};
    double p3_23[2] = {p3[1], p3[2]};
    trimodefinder(y, z, x, p1_23, p2_23, p3_23, &Trimode);

    // Strain components in TDCS
    double exx = 0.0, eyy = 0.0, ezz = 0.0, exy = 0.0, exz = 0.0, eyz = 0.0;

    // Configuration I
    if (Trimode == 1) {
        double Exx1Tp, Eyy1Tp, Ezz1Tp, Exy1Tp, Exz1Tp, Eyz1Tp;
        double Exx2Tp, Eyy2Tp, Ezz2Tp, Exy2Tp, Exz2Tp, Eyz2Tp;
        double Exx3Tp, Eyy3Tp, Ezz3Tp, Exy3Tp, Exz3Tp, Eyz3Tp;
        double neg_e13[3] = {-e13[0], -e13[1], -e13[2]};

        TDSetupS(x, y, z, A_angle, bx, by, bz, nu, p1, neg_e13, &Exx1Tp,
                 &Eyy1Tp, &Ezz1Tp, &Exy1Tp, &Exz1Tp, &Eyz1Tp);
        TDSetupS(x, y, z, B, bx, by, bz, nu, p2, e12, &Exx2Tp, &Eyy2Tp, &Ezz2Tp,
                 &Exy2Tp, &Exz2Tp, &Eyz2Tp);
        TDSetupS(x, y, z, C, bx, by, bz, nu, p3, e23, &Exx3Tp, &Eyy3Tp, &Ezz3Tp,
                 &Exy3Tp, &Exz3Tp, &Eyz3Tp);

        exx = Exx1Tp + Exx2Tp + Exx3Tp;
        eyy = Eyy1Tp + Eyy2Tp + Eyy3Tp;
        ezz = Ezz1Tp + Ezz2Tp + Ezz3Tp;
        exy = Exy1Tp + Exy2Tp + Exy3Tp;
        exz = Exz1Tp + Exz2Tp + Exz3Tp;
        eyz = Eyz1Tp + Eyz2Tp + Eyz3Tp;
    }
    // Configuration II
    else if (Trimode == -1) {
        double Exx1Tn, Eyy1Tn, Ezz1Tn, Exy1Tn, Exz1Tn, Eyz1Tn;
        double Exx2Tn, Eyy2Tn, Ezz2Tn, Exy2Tn, Exz2Tn, Eyz2Tn;
        double Exx3Tn, Eyy3Tn, Ezz3Tn, Exy3Tn, Exz3Tn, Eyz3Tn;
        double neg_e12[3] = {-e12[0], -e12[1], -e12[2]};
        double neg_e23[3] = {-e23[0], -e23[1], -e23[2]};

        TDSetupS(x, y, z, A_angle, bx, by, bz, nu, p1, e13, &Exx1Tn, &Eyy1Tn,
                 &Ezz1Tn, &Exy1Tn, &Exz1Tn, &Eyz1Tn);
        TDSetupS(x, y, z, B, bx, by, bz, nu, p2, neg_e12, &Exx2Tn, &Eyy2Tn,
                 &Ezz2Tn, &Exy2Tn, &Exz2Tn, &Eyz2Tn);
        TDSetupS(x, y, z, C, bx, by, bz, nu, p3, neg_e23, &Exx3Tn, &Eyy3Tn,
                 &Ezz3Tn, &Exy3Tn, &Exz3Tn, &Eyz3Tn);

        exx = Exx1Tn + Exx2Tn + Exx3Tn;
        eyy = Eyy1Tn + Eyy2Tn + Eyy3Tn;
        ezz = Ezz1Tn + Ezz2Tn + Ezz3Tn;
        exy = Exy1Tn + Exy2Tn + Exy3Tn;
        exz = Exz1Tn + Exz2Tn + Exz3Tn;
        eyz = Eyz1Tn + Eyz2Tn + Eyz3Tn;
    }
    // Zero case
    else if (Trimode == 0) {
        exx = NAN;
        eyy = NAN;
        ezz = NAN;
        exy = NAN;
        exz = NAN;
        eyz = NAN;
    }

    // Transform strain tensor from TDCS to EFCS
    double Exx, Eyy, Ezz, Exy, Exz, Eyz;
    double At[9] = {
        Vnorm[0], Vstrike[0], Vdip[0],
        Vnorm[1], Vstrike[1], Vdip[1],
        Vnorm[2], Vstrike[2], Vdip[2]
    };
    TensTrans(exx, eyy, ezz, exy, exz, eyz, At, &Exx, &Eyy, &Ezz, &Exy, &Exz,
              &Eyz);

    // Calculate stress tensor components in EFCS
    double Sxx = 2.0 * mu * Exx + lambda * (Exx + Eyy + Ezz);
    double Syy = 2.0 * mu * Eyy + lambda * (Exx + Eyy + Ezz);
    double Szz = 2.0 * mu * Ezz + lambda * (Exx + Eyy + Ezz);
    double Sxy = 2.0 * mu * Exy;
    double Sxz = 2.0 * mu * Exz;
    double Syz = 2.0 * mu * Eyz;

    // Assign outputs
    Stress[0] = Sxx;
    Stress[1] = Syy;
    Stress[2] = Szz;
    Stress[3] = Sxy;
    Stress[4] = Sxz;
    Stress[5] = Syz;

    Strain[0] = Exx;
    Strain[1] = Eyy;
    Strain[2] = Ezz;
    Strain[3] = Exy;
    Strain[4] = Exz;
    Strain[5] = Eyz;
}
