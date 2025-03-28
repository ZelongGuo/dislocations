#include "mehdi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*---------------------------------------------------------
 *  Functions used for TDstressHS.c
 *
 *  Referring to the Matlab codes of Mehdi
 *
 *  Author: Zelong Guo
 *  03.2025, @ Potsdam, Germany
 *  zelong.guo@outlook.com
 *
 * ---------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/

void TDstressFS4HS(double X, double Y, double Z, double P1[3], double P2[3],
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

    double eX[3] = {1.0, 0.0, 0.0};
    double eY[3] = {0.0, 1.0, 0.0};
    double eZ[3] = {0.0, 0.0, 1.0};
    Vstrike[0] = eZ[1] * Vnorm[2] - eZ[2] * Vnorm[1];
    Vstrike[1] = eZ[2] * Vnorm[0] - eZ[0] * Vnorm[2];
    Vstrike[2] = eZ[0] * Vnorm[1] - eZ[1] * Vnorm[0];
    double norm_Vstrike =
        sqrt(Vstrike[0] * Vstrike[0] + Vstrike[1] * Vstrike[1] +
             Vstrike[2] * Vstrike[2]);
    if (norm_Vstrike == 0.0) {
        Vstrike[0] = eX[0] * Vnorm[2];
        Vstrike[1] = eX[1] * Vnorm[2];
        Vstrike[2] = eX[2] * Vnorm[2];
        norm_Vstrike = sqrt(Vstrike[0] * Vstrike[0] + Vstrike[1] * Vstrike[1] +
                            Vstrike[2] * Vstrike[2]);
    // For horizontal elements in case of half-space calculation!!!
    // Correct the strike vector of image dislocation only
        if (P1[2] > 0) {
            Vstrike[0] = -Vstrike[0];
            Vstrike[1] = -Vstrike[1];
            Vstrike[2] = -Vstrike[2];
        }
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

/*-----------------------------------------------------------------------------------------------*/
void AngDisStrainFSC(double y1, double y2, double y3, double beta, double b1, double b2, double b3,
                     double nu, double a, double *v11, double *v22, double *v33, double *v12,
                     double *v13, double *v23) {
    // Precompute trigonometric functions and common terms
    double sinB = sin(beta);
    double cosB = cos(beta);
    double cotB = 1.0 / tan(beta); // cot(beta) = cos(beta)/sin(beta)
    double y3b = y3 + 2.0 * a;
    double z1b = y1 * cosB + y3b * sinB;
    double z3b = -y1 * sinB + y3b * cosB;
    double rb2 = y1 * y1 + y2 * y2 + y3b * y3b;
    double rb = sqrt(rb2);

    // Intermediate variables W1 to W9
    double W1 = rb * cosB + y3b;
    double W2 = cosB + a / rb;
    double W3 = cosB + y3b / rb;
    double W4 = nu + a / rb;
    double W5 = 2.0 * nu + a / rb;
    double W6 = rb + y3b;
    double W7 = rb + z3b;
    double W8 = y3 + a;
    double W9 = 1.0 + a / rb / cosB;

    double N1 = 1.0 - 2.0 * nu;

    // Partial derivatives of the Burgers' function
    double rFib_ry2 = z1b / rb / (rb + z3b) - y1 / rb / (rb + y3b);
    double rFib_ry1 = y2 / rb / (rb + y3b) - cosB * y2 / rb / (rb + z3b);
    double rFib_ry3 = -sinB * y2 / rb / (rb + z3b);

    // Compute strain components
    // v11
    *v11 =
        b1 * (0.25 *
              ((-2.0 + 2.0 * nu) * N1 * rFib_ry1 * cotB * cotB -
               N1 * y2 / (W6 * W6) * ((1.0 - W5) * cotB - y1 / W6 * W4) / rb * y1 +
               N1 * y2 / W6 *
                   (a / (rb * rb * rb) * y1 * cotB - 1.0 / W6 * W4 + y1 * y1 / (W6 * W6) * W4 / rb +
                    y1 * y1 / W6 * a / (rb * rb * rb)) -
               N1 * y2 * cosB * cotB / (W7 * W7) * W2 * (y1 / rb - sinB) -
               N1 * y2 * cosB * cotB / W7 * a / (rb * rb * rb) * y1 -
               3.0 * a * y2 * W8 * cotB / (rb * rb * rb * rb * rb) * y1 -
               y2 * W8 / (rb * rb * rb) / W6 * (-N1 * cotB + y1 / W6 * W5 + a * y1 / rb2) * y1 -
               y2 * W8 / rb2 / (W6 * W6) * (-N1 * cotB + y1 / W6 * W5 + a * y1 / rb2) * y1 +
               y2 * W8 / rb / W6 *
                   (1.0 / W6 * W5 - y1 * y1 / (W6 * W6) * W5 / rb -
                    y1 * y1 / W6 * a / (rb * rb * rb) + a / rb2 - 2.0 * a * y1 * y1 / (rb2 * rb2)) -
               y2 * W8 / (rb * rb * rb) / W7 *
                   (cosB / W7 *
                        (W1 * (N1 * cosB - a / rb) * cotB +
                         (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) -
                    a * y3b * cosB * cotB / rb2) *
                   y1 -
               y2 * W8 / rb / (W7 * W7) *
                   (cosB / W7 *
                        (W1 * (N1 * cosB - a / rb) * cotB +
                         (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) -
                    a * y3b * cosB * cotB / rb2) *
                   (y1 / rb - sinB) +
               y2 * W8 / rb / W7 *
                   (-cosB / (W7 * W7) *
                        (W1 * (N1 * cosB - a / rb) * cotB +
                         (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) *
                        (y1 / rb - sinB) +
                    cosB / W7 *
                        (1.0 / rb * cosB * y1 * (N1 * cosB - a / rb) * cotB +
                         W1 * a / (rb * rb * rb) * y1 * cotB +
                         (2.0 - 2.0 * nu) * (1.0 / rb * sinB * y1 - 1.0) * cosB) +
                    2.0 * a * y3b * cosB * cotB / (rb2 * rb2) * y1)) /
              (M_PI * (1.0 - nu))) +
        b2 * (0.25 *
              (N1 * (((2.0 - 2.0 * nu) * cotB * cotB + nu) / rb * y1 / W6 -
                     ((2.0 - 2.0 * nu) * cotB * cotB + 1.0) * cosB * (y1 / rb - sinB) / W7) -
               N1 / (W6 * W6) *
                   (-N1 * y1 * cotB + nu * y3b - a + a * y1 * cotB / rb + y1 * y1 / W6 * W4) / rb *
                   y1 +
               N1 / W6 *
                   (-N1 * cotB + a * cotB / rb - a * y1 * y1 * cotB / (rb * rb * rb) +
                    2.0 * y1 / W6 * W4 - y1 * y1 * y1 / (W6 * W6) * W4 / rb -
                    y1 * y1 * y1 / W6 * a / (rb * rb * rb)) +
               N1 * cotB / (W7 * W7) * (z1b * cosB - a * (rb * sinB - y1) / rb / cosB) *
                   (y1 / rb - sinB) -
               N1 * cotB / W7 *
                   (cosB * cosB - a * (1.0 / rb * sinB * y1 - 1.0) / rb / cosB +
                    a * (rb * sinB - y1) / (rb * rb * rb) / cosB * y1) -
               a * W8 * cotB / (rb * rb * rb) +
               3.0 * a * y1 * y1 * W8 * cotB / (rb * rb * rb * rb * rb) -
               W8 / (W6 * W6) *
                   (2.0 * nu + 1.0 / rb * (N1 * y1 * cotB + a) - y1 * y1 / rb / W6 * W5 -
                    a * y1 * y1 / (rb * rb * rb)) /
                   rb * y1 +
               W8 / W6 *
                   (-1.0 / (rb * rb * rb) * (N1 * y1 * cotB + a) * y1 + 1.0 / rb * N1 * cotB -
                    2.0 * y1 / rb / W6 * W5 + y1 * y1 * y1 / (rb * rb * rb) / W6 * W5 +
                    y1 * y1 * y1 / rb2 / (W6 * W6) * W5 + y1 * y1 * y1 / (rb2 * rb2) / W6 * a -
                    2.0 * a / (rb * rb * rb) * y1 +
                    3.0 * a * y1 * y1 * y1 / (rb * rb * rb * rb * rb)) -
               W8 * cotB / (W7 * W7) *
                   (-cosB * sinB + a * y1 * y3b / (rb * rb * rb) / cosB +
                    (rb * sinB - y1) / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7 * W9)) *
                   (y1 / rb - sinB) +
               W8 * cotB / W7 *
                   (a * y3b / (rb * rb * rb) / cosB -
                    3.0 * a * y1 * y1 * y3b / (rb * rb * rb * rb * rb) / cosB +
                    (1.0 / rb * sinB * y1 - 1.0) / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7 * W9) -
                    (rb * sinB - y1) / (rb * rb * rb) * ((2.0 - 2.0 * nu) * cosB - W1 / W7 * W9) *
                        y1 +
                    (rb * sinB - y1) / rb *
                        (-1.0 / rb * cosB * y1 / W7 * W9 + W1 / (W7 * W7) * W9 * (y1 / rb - sinB) +
                         W1 / W7 * a / (rb * rb * rb) / cosB * y1))) /
              (M_PI * (1.0 - nu))) +
        b3 *
            (0.25 *
             (N1 * (-y2 / (W6 * W6) * (1.0 + a / rb) / rb * y1 - y2 / W6 * a / (rb * rb * rb) * y1 +
                    y2 * cosB / (W7 * W7) * W2 * (y1 / rb - sinB) +
                    y2 * cosB / W7 * a / (rb * rb * rb) * y1) +
              y2 * W8 / (rb * rb * rb) * (a / rb2 + 1.0 / W6) * y1 -
              y2 * W8 / rb * (-2.0 * a / (rb2 * rb2) * y1 - 1.0 / (W6 * W6) / rb * y1) -
              y2 * W8 * cosB / (rb * rb * rb) / W7 * (W1 / W7 * W2 + a * y3b / rb2) * y1 -
              y2 * W8 * cosB / rb / (W7 * W7) * (W1 / W7 * W2 + a * y3b / rb2) * (y1 / rb - sinB) +
              y2 * W8 * cosB / rb / W7 *
                  (1.0 / rb * cosB * y1 / W7 * W2 - W1 / (W7 * W7) * W2 * (y1 / rb - sinB) -
                   W1 / W7 * a / (rb * rb * rb) * y1 - 2.0 * a * y3b / (rb2 * rb2) * y1)) /
             (M_PI * (1.0 - nu)));

    // v22
    *v22 =
        b1 *
            (0.25 *
             (N1 * (((2.0 - 2.0 * nu) * cotB * cotB - nu) / rb * y2 / W6 -
                    ((2.0 - 2.0 * nu) * cotB * cotB + 1.0 - 2.0 * nu) * cosB / rb * y2 / W7) +
              N1 / (W6 * W6) * (y1 * cotB * (1.0 - W5) + nu * y3b - a + y2 * y2 / W6 * W4) / rb *
                  y2 -
              N1 / W6 *
                  (a * y1 * cotB / (rb * rb * rb) * y2 + 2.0 * y2 / W6 * W4 -
                   y2 * y2 * y2 / (W6 * W6) * W4 / rb - y2 * y2 * y2 / W6 * a / (rb * rb * rb)) +
              N1 * z1b * cotB / (W7 * W7) * W2 / rb * y2 +
              N1 * z1b * cotB / W7 * a / (rb * rb * rb) * y2 +
              3.0 * a * y2 * W8 * cotB / (rb * rb * rb * rb * rb) * y1 -
              W8 / (W6 * W6) *
                  (-2.0 * nu + 1.0 / rb * (N1 * y1 * cotB - a) + y2 * y2 / rb / W6 * W5 +
                   a * y2 * y2 / (rb * rb * rb)) /
                  rb * y2 +
              W8 / W6 *
                  (-1.0 / (rb * rb * rb) * (N1 * y1 * cotB - a) * y2 + 2.0 * y2 / rb / W6 * W5 -
                   y2 * y2 * y2 / (rb * rb * rb) / W6 * W5 - y2 * y2 * y2 / rb2 / (W6 * W6) * W5 -
                   y2 * y2 * y2 / (rb2 * rb2) / W6 * a + 2.0 * a / (rb * rb * rb) * y2 -
                   3.0 * a * y2 * y2 * y2 / (rb * rb * rb * rb * rb)) -
              W8 / (W7 * W7) *
                  (cosB * cosB - 1.0 / rb * (N1 * z1b * cotB + a * cosB) +
                   a * y3b * z1b * cotB / (rb * rb * rb) -
                   1.0 / rb / W7 * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * W1)) /
                  rb * y2 +
              W8 / W7 *
                  (1.0 / (rb * rb * rb) * (N1 * z1b * cotB + a * cosB) * y2 -
                   3.0 * a * y3b * z1b * cotB / (rb * rb * rb * rb * rb) * y2 +
                   1.0 / (rb * rb * rb) / W7 * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * W1) *
                       y2 +
                   1.0 / rb2 / (W7 * W7) * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * W1) * y2 -
                   1.0 / rb / W7 *
                       (2.0 * y2 * cosB * cosB + a * z1b * cotB / (rb * rb * rb) * W1 * y2 -
                        a * z1b * cotB / rb2 * cosB * y2))) /
             (M_PI * (1.0 - nu))) +
        b2 * (0.25 *
              ((2.0 - 2.0 * nu) * N1 * rFib_ry2 * cotB * cotB +
               N1 / W6 * ((W5 - 1.0) * cotB + y1 / W6 * W4) -
               N1 * y2 * y2 / (W6 * W6) * ((W5 - 1.0) * cotB + y1 / W6 * W4) / rb +
               N1 * y2 / W6 *
                   (-a / (rb * rb * rb) * y2 * cotB - y1 / (W6 * W6) * W4 / rb * y2 -
                    y2 / W6 * a / (rb * rb * rb) * y1) -
               N1 * cotB / W7 * W9 + N1 * y2 * y2 * cotB / (W7 * W7) * W9 / rb +
               N1 * y2 * y2 * cotB / W7 * a / (rb * rb * rb) / cosB -
               a * W8 * cotB / (rb * rb * rb) +
               3.0 * a * y2 * y2 * W8 * cotB / (rb * rb * rb * rb * rb) +
               W8 / rb / W6 *
                   (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb * (1.0 / rb + 1.0 / W6)) -
               y2 * y2 * W8 / (rb * rb * rb) / W6 *
                   (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb * (1.0 / rb + 1.0 / W6)) -
               y2 * y2 * W8 / rb2 / (W6 * W6) *
                   (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb * (1.0 / rb + 1.0 / W6)) +
               y2 * W8 / rb / W6 *
                   (2.0 * nu * y1 / (W6 * W6) / rb * y2 +
                    a * y1 / (rb * rb * rb) * (1.0 / rb + 1.0 / W6) * y2 -
                    a * y1 / rb * (-1.0 / (rb * rb * rb) * y2 - 1.0 / (W6 * W6) / rb * y2)) +
               W8 * cotB / rb / W7 *
                   ((-2.0 + 2.0 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB) -
               y2 * y2 * W8 * cotB / (rb * rb * rb) / W7 *
                   ((-2.0 + 2.0 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB) -
               y2 * y2 * W8 * cotB / rb2 / (W7 * W7) *
                   ((-2.0 + 2.0 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB) +
               y2 * W8 * cotB / rb / W7 *
                   (1.0 / rb * cosB * y2 / W7 * W9 - W1 / (W7 * W7) * W9 / rb * y2 -
                    W1 / W7 * a / (rb * rb * rb) / cosB * y2 -
                    2.0 * a * y3b / (rb2 * rb2) / cosB * y2)) /
              (M_PI * (1.0 - nu))) +
        b3 * (0.25 *
              (N1 * (-sinB / rb * y2 / W7 + y2 / (W6 * W6) * (1.0 + a / rb) / rb * y1 +
                     y2 / W6 * a / (rb * rb * rb) * y1 - z1b / (W7 * W7) * W2 / rb * y2 -
                     z1b / W7 * a / (rb * rb * rb) * y2) -
               y2 * W8 / (rb * rb * rb) * (a / rb2 + 1.0 / W6) * y1 +
               y1 * W8 / rb * (-2.0 * a / (rb2 * rb2) * y2 - 1.0 / (W6 * W6) / rb * y2) +
               W8 / (W7 * W7) *
                   (sinB * (cosB - a / rb) + z1b / rb * (1.0 + a * y3b / rb2) -
                    1.0 / rb / W7 * (y2 * y2 * cosB * sinB - a * z1b / rb * W1)) /
                   rb * y2 -
               W8 / W7 *
                   (sinB * a / (rb * rb * rb) * y2 -
                    z1b / (rb * rb * rb) * (1.0 + a * y3b / rb2) * y2 -
                    2.0 * z1b / (rb * rb * rb * rb * rb) * a * y3b * y2 +
                    1.0 / (rb * rb * rb) / W7 * (y2 * y2 * cosB * sinB - a * z1b / rb * W1) * y2 +
                    1.0 / rb2 / (W7 * W7) * (y2 * y2 * cosB * sinB - a * z1b / rb * W1) * y2 -
                    1.0 / rb / W7 *
                        (2.0 * y2 * cosB * sinB + a * z1b / (rb * rb * rb) * W1 * y2 -
                         a * z1b / rb2 * cosB * y2))) /
              (M_PI * (1.0 - nu)));

    // v33
    *v33 =
        b1 * (0.25 *
              ((2.0 - 2.0 * nu) * (N1 * rFib_ry3 * cotB - y2 / (W6 * W6) * W5 * (y3b / rb + 1.0) -
                                   0.5 * y2 / W6 * a / (rb * rb * rb) * 2.0 * y3b +
                                   y2 * cosB / (W7 * W7) * W2 * W3 +
                                   0.5 * y2 * cosB / W7 * a / (rb * rb * rb) * 2.0 * y3b) +
               y2 / rb * (2.0 * nu / W6 + a / rb2) -
               0.5 * y2 * W8 / (rb * rb * rb) * (2.0 * nu / W6 + a / rb2) * 2.0 * y3b +
               y2 * W8 / rb *
                   (-2.0 * nu / (W6 * W6) * (y3b / rb + 1.0) - a / (rb2 * rb2) * 2.0 * y3b) +
               y2 * cosB / rb / W7 * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3b / rb2) -
               0.5 * y2 * W8 * cosB / (rb * rb * rb) / W7 *
                   (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3b / rb2) * 2.0 * y3b -
               y2 * W8 * cosB / rb / (W7 * W7) * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3b / rb2) *
                   W3 +
               y2 * W8 * cosB / rb / W7 *
                   (-(cosB * y3b / rb + 1.0) / W7 * W2 + W1 / (W7 * W7) * W2 * W3 +
                    0.5 * W1 / W7 * a / (rb * rb * rb) * 2.0 * y3b - a / rb2 +
                    a * y3b / (rb2 * rb2) * 2.0 * y3b)) /
              (M_PI * (1.0 - nu))) +
        b2 * (0.25 *
              ((-2.0 + 2.0 * nu) * N1 * cotB * ((y3b / rb + 1.0) / W6 - cosB * W3 / W7) +
               (2.0 - 2.0 * nu) * y1 / (W6 * W6) * W5 * (y3b / rb + 1.0) +
               0.5 * (2.0 - 2.0 * nu) * y1 / W6 * a / (rb * rb * rb) * 2.0 * y3b +
               (2.0 - 2.0 * nu) * sinB / W7 * W2 - (2.0 - 2.0 * nu) * z1b / (W7 * W7) * W2 * W3 -
               0.5 * (2.0 - 2.0 * nu) * z1b / W7 * a / (rb * rb * rb) * 2.0 * y3b +
               1.0 / rb * (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb2) -
               0.5 * W8 / (rb * rb * rb) * (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb2) * 2.0 *
                   y3b +
               W8 / rb *
                   (2.0 * nu * y1 / (W6 * W6) * (y3b / rb + 1.0) +
                    a * y1 / (rb2 * rb2) * 2.0 * y3b) -
               1.0 / W7 *
                   (cosB * sinB + W1 * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) +
                    a / rb * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7)) +
               W8 / (W7 * W7) *
                   (cosB * sinB + W1 * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) +
                    a / rb * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7)) *
                   W3 -
               W8 / W7 *
                   ((cosB * y3b / rb + 1.0) * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) -
                    0.5 * W1 * cotB / (rb * rb * rb) * ((2.0 - 2.0 * nu) * cosB - W1 / W7) * 2.0 *
                        y3b +
                    W1 * cotB / rb * (-(cosB * y3b / rb + 1.0) / W7 + W1 / (W7 * W7) * W3) -
                    0.5 * a / (rb * rb * rb) * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7) * 2.0 *
                        y3b +
                    a / rb *
                        (-z1b / rb2 - y3b * sinB / rb2 + y3b * z1b / (rb2 * rb2) * 2.0 * y3b -
                         sinB * W1 / rb / W7 - z1b * (cosB * y3b / rb + 1.0) / rb / W7 +
                         0.5 * z1b * W1 / (rb * rb * rb) / W7 * 2.0 * y3b +
                         z1b * W1 / rb / (W7 * W7) * W3))) /
              (M_PI * (1.0 - nu))) +
        b3 * (0.25 *
              ((2.0 - 2.0 * nu) * rFib_ry3 - (2.0 - 2.0 * nu) * y2 * sinB / (W7 * W7) * W2 * W3 -
               0.5 * (2.0 - 2.0 * nu) * y2 * sinB / W7 * a / (rb * rb * rb) * 2.0 * y3b +
               y2 * sinB / rb / W7 * (1.0 + W1 / W7 * W2 + a * y3b / rb2) -
               0.5 * y2 * W8 * sinB / (rb * rb * rb) / W7 * (1.0 + W1 / W7 * W2 + a * y3b / rb2) *
                   2.0 * y3b -
               y2 * W8 * sinB / rb / (W7 * W7) * (1.0 + W1 / W7 * W2 + a * y3b / rb2) * W3 +
               y2 * W8 * sinB / rb / W7 *
                   ((cosB * y3b / rb + 1.0) / W7 * W2 - W1 / (W7 * W7) * W2 * W3 -
                    0.5 * W1 / W7 * a / (rb * rb * rb) * 2.0 * y3b + a / rb2 -
                    a * y3b / (rb2 * rb2) * 2.0 * y3b)) /
              (M_PI * (1.0 - nu)));

    // v12
    *v12 =
        b1 / 2.0 *
            (0.25 *
             ((-2.0 + 2.0 * nu) * N1 * rFib_ry2 * cotB * cotB +
              N1 / W6 * ((1.0 - W5) * cotB - y1 / W6 * W4) -
              N1 * y2 * y2 / (W6 * W6) * ((1.0 - W5) * cotB - y1 / W6 * W4) / rb +
              N1 * y2 / W6 *
                  (a / (rb * rb * rb) * y2 * cotB + y1 / (W6 * W6) * W4 / rb * y2 +
                   y2 / W6 * a / (rb * rb * rb) * y1) +
              N1 * cosB * cotB / W7 * W2 - N1 * y2 * y2 * cosB * cotB / (W7 * W7) * W2 / rb -
              N1 * y2 * y2 * cosB * cotB / W7 * a / (rb * rb * rb) +
              a * W8 * cotB / (rb * rb * rb) -
              3.0 * a * y2 * y2 * W8 * cotB / (rb * rb * rb * rb * rb) +
              W8 / rb / W6 * (-N1 * cotB + y1 / W6 * W5 + a * y1 / rb2) -
              y2 * y2 * W8 / (rb * rb * rb) / W6 * (-N1 * cotB + y1 / W6 * W5 + a * y1 / rb2) -
              y2 * y2 * W8 / rb2 / (W6 * W6) * (-N1 * cotB + y1 / W6 * W5 + a * y1 / rb2) +
              y2 * W8 / rb / W6 *
                  (-y1 / (W6 * W6) * W5 / rb * y2 - y2 / W6 * a / (rb * rb * rb) * y1 -
                   2.0 * a * y1 / (rb2 * rb2) * y2) +
              W8 / rb / W7 *
                  (cosB / W7 *
                       (W1 * (N1 * cosB - a / rb) * cotB +
                        (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) -
                   a * y3b * cosB * cotB / rb2) -
              y2 * y2 * W8 / (rb * rb * rb) / W7 *
                  (cosB / W7 *
                       (W1 * (N1 * cosB - a / rb) * cotB +
                        (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) -
                   a * y3b * cosB * cotB / rb2) -
              y2 * y2 * W8 / rb2 / (W7 * W7) *
                  (cosB / W7 *
                       (W1 * (N1 * cosB - a / rb) * cotB +
                        (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) -
                   a * y3b * cosB * cotB / rb2) +
              y2 * W8 / rb / W7 *
                  (-cosB / (W7 * W7) *
                       (W1 * (N1 * cosB - a / rb) * cotB +
                        (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) /
                       rb * y2 +
                   cosB / W7 *
                       (1.0 / rb * cosB * y2 * (N1 * cosB - a / rb) * cotB +
                        W1 * a / (rb * rb * rb) * y2 * cotB +
                        (2.0 - 2.0 * nu) / rb * sinB * y2 * cosB) +
                   2.0 * a * y3b * cosB * cotB / (rb2 * rb2) * y2)) /
             (M_PI * (1.0 - nu))) +
        b2 / 2.0 *
            (0.25 *
             (N1 * (((2.0 - 2.0 * nu) * cotB * cotB + nu) / rb * y2 / W6 -
                    ((2.0 - 2.0 * nu) * cotB * cotB + 1.0) * cosB / rb * y2 / W7) -
              N1 / (W6 * W6) *
                  (-N1 * y1 * cotB + nu * y3b - a + a * y1 * cotB / rb + y1 * y1 / W6 * W4) / rb *
                  y2 +
              N1 / W6 *
                  (-a * y1 * cotB / (rb * rb * rb) * y2 - y1 * y1 / (W6 * W6) * W4 / rb * y2 -
                   y1 * y1 / W6 * a / (rb * rb * rb) * y2) +
              N1 * cotB / (W7 * W7) * (z1b * cosB - a * (rb * sinB - y1) / rb / cosB) / rb * y2 -
              N1 * cotB / W7 *
                  (-a / rb2 * sinB * y2 / cosB +
                   a * (rb * sinB - y1) / (rb * rb * rb) / cosB * y2) +
              3.0 * a * y2 * W8 * cotB / (rb * rb * rb * rb * rb) * y1 -
              W8 / (W6 * W6) *
                  (2.0 * nu + 1.0 / rb * (N1 * y1 * cotB + a) - y1 * y1 / rb / W6 * W5 -
                   a * y1 * y1 / (rb * rb * rb)) /
                  rb * y2 +
              W8 / W6 *
                  (-1.0 / (rb * rb * rb) * (N1 * y1 * cotB + a) * y2 +
                   y1 * y1 / (rb * rb * rb) / W6 * W5 * y2 + y1 * y1 / rb2 / (W6 * W6) * W5 * y2 +
                   y1 * y1 / (rb2 * rb2) / W6 * a * y2 +
                   3.0 * a * y1 * y1 / (rb * rb * rb * rb * rb) * y2) -
              W8 * cotB / (W7 * W7) *
                  (-cosB * sinB + a * y1 * y3b / (rb * rb * rb) / cosB +
                   (rb * sinB - y1) / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7 * W9)) /
                  rb * y2 +
              W8 * cotB / W7 *
                  (-3.0 * a * y1 * y3b / (rb * rb * rb * rb * rb) / cosB * y2 +
                   1.0 / rb2 * sinB * y2 * ((2.0 - 2.0 * nu) * cosB - W1 / W7 * W9) -
                   (rb * sinB - y1) / (rb * rb * rb) * ((2.0 - 2.0 * nu) * cosB - W1 / W7 * W9) *
                       y2 +
                   (rb * sinB - y1) / rb *
                       (-1.0 / rb * cosB * y2 / W7 * W9 + W1 / (W7 * W7) * W9 / rb * y2 +
                        W1 / W7 * a / (rb * rb * rb) / cosB * y2))) /
             (M_PI * (1.0 - nu))) +
        b3 / 2.0 *
            (0.25 *
             (N1 * (1.0 / W6 * (1.0 + a / rb) - y2 * y2 / (W6 * W6) * (1.0 + a / rb) / rb -
                    y2 * y2 / W6 * a / (rb * rb * rb) - cosB / W7 * W2 +
                    y2 * y2 * cosB / (W7 * W7) * W2 / rb +
                    y2 * y2 * cosB / W7 * a / (rb * rb * rb)) -
              W8 / rb * (a / rb2 + 1.0 / W6) +
              y2 * y2 * W8 / (rb * rb * rb) * (a / rb2 + 1.0 / W6) -
              y2 * W8 / rb * (-2.0 * a / (rb2 * rb2) * y2 - 1.0 / (W6 * W6) / rb * y2) +
              W8 * cosB / rb / W7 * (W1 / W7 * W2 + a * y3b / rb2) -
              y2 * y2 * W8 * cosB / (rb * rb * rb) / W7 * (W1 / W7 * W2 + a * y3b / rb2) -
              y2 * y2 * W8 * cosB / rb2 / (W7 * W7) * (W1 / W7 * W2 + a * y3b / rb2) +
              y2 * W8 * cosB / rb / W7 *
                  (1.0 / rb * cosB * y2 / W7 * W2 - W1 / (W7 * W7) * W2 / rb * y2 -
                   W1 / W7 * a / (rb * rb * rb) * y2 - 2.0 * a * y3b / (rb2 * rb2) * y2)) /
             (M_PI * (1.0 - nu))) +
        b1 / 2.0 *
            (0.25 *
             (N1 * (((2.0 - 2.0 * nu) * cotB * cotB - nu) / rb * y1 / W6 -
                    ((2.0 - 2.0 * nu) * cotB * cotB + 1.0 - 2.0 * nu) * cosB * (y1 / rb - sinB) /
                        W7) +
              N1 / (W6 * W6) * (y1 * cotB * (1.0 - W5) + nu * y3b - a + y2 * y2 / W6 * W4) / rb *
                  y1 -
              N1 / W6 *
                  ((1.0 - W5) * cotB + a * y1 * y1 * cotB / (rb * rb * rb) -
                   y2 * y2 / (W6 * W6) * W4 / rb * y1 - y2 * y2 / W6 * a / (rb * rb * rb) * y1) -
              N1 * cosB * cotB / W7 * W2 + N1 * z1b * cotB / (W7 * W7) * W2 * (y1 / rb - sinB) +
              N1 * z1b * cotB / W7 * a / (rb * rb * rb) * y1 - a * W8 * cotB / (rb * rb * rb) +
              3.0 * a * y1 * y1 * W8 * cotB / (rb * rb * rb * rb * rb) -
              W8 / (W6 * W6) *
                  (-2.0 * nu + 1.0 / rb * (N1 * y1 * cotB - a) + y2 * y2 / rb / W6 * W5 +
                   a * y2 * y2 / (rb * rb * rb)) /
                  rb * y1 +
              W8 / W6 *
                  (-1.0 / (rb * rb * rb) * (N1 * y1 * cotB - a) * y1 + 1.0 / rb * N1 * cotB -
                   y2 * y2 / (rb * rb * rb) / W6 * W5 * y1 - y2 * y2 / rb2 / (W6 * W6) * W5 * y1 -
                   y2 * y2 / (rb2 * rb2) / W6 * a * y1 -
                   3.0 * a * y2 * y2 / (rb * rb * rb * rb * rb) * y1) -
              W8 / (W7 * W7) *
                  (cosB * cosB - 1.0 / rb * (N1 * z1b * cotB + a * cosB) +
                   a * y3b * z1b * cotB / (rb * rb * rb) -
                   1.0 / rb / W7 * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * W1)) *
                  (y1 / rb - sinB) +
              W8 / W7 *
                  (1.0 / (rb * rb * rb) * (N1 * z1b * cotB + a * cosB) * y1 -
                   1.0 / rb * N1 * cosB * cotB + a * y3b * cosB * cotB / (rb * rb * rb) -
                   3.0 * a * y3b * z1b * cotB / (rb * rb * rb * rb * rb) * y1 +
                   1.0 / (rb * rb * rb) / W7 * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * W1) *
                       y1 +
                   1.0 / rb / (W7 * W7) * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * W1) *
                       (y1 / rb - sinB) -
                   1.0 / rb / W7 *
                       (-a * cosB * cotB / rb * W1 + a * z1b * cotB / (rb * rb * rb) * W1 * y1 -
                        a * z1b * cotB / rb2 * cosB * y1))) /
             (M_PI * (1.0 - nu))) +
        b2 / 2.0 *
            (0.25 *
             ((2.0 - 2.0 * nu) * N1 * rFib_ry1 * cotB * cotB -
              N1 * y2 / (W6 * W6) * ((W5 - 1.0) * cotB + y1 / W6 * W4) / rb * y1 +
              N1 * y2 / W6 *
                  (-a / (rb * rb * rb) * y1 * cotB + 1.0 / W6 * W4 - y1 * y1 / (W6 * W6) * W4 / rb -
                   y1 * y1 / W6 * a / (rb * rb * rb)) +
              N1 * y2 * cotB / (W7 * W7) * W9 * (y1 / rb - sinB) +
              N1 * y2 * cotB / W7 * a / (rb * rb * rb) / cosB * y1 +
              3.0 * a * y2 * W8 * cotB / (rb * rb * rb * rb * rb) * y1 -
              y2 * W8 / (rb * rb * rb) / W6 *
                  (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb * (1.0 / rb + 1.0 / W6)) * y1 -
              y2 * W8 / rb2 / (W6 * W6) *
                  (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb * (1.0 / rb + 1.0 / W6)) * y1 +
              y2 * W8 / rb / W6 *
                  (-2.0 * nu / W6 + 2.0 * nu * y1 * y1 / (W6 * W6) / rb -
                   a / rb * (1.0 / rb + 1.0 / W6) +
                   a * y1 * y1 / (rb * rb * rb) * (1.0 / rb + 1.0 / W6) -
                   a * y1 / rb * (-1.0 / (rb * rb * rb) * y1 - 1.0 / (W6 * W6) / rb * y1)) -
              y2 * W8 * cotB / (rb * rb * rb) / W7 *
                  ((-2.0 + 2.0 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB) * y1 -
              y2 * W8 * cotB / rb / (W7 * W7) *
                  ((-2.0 + 2.0 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB) *
                  (y1 / rb - sinB) +
              y2 * W8 * cotB / rb / W7 *
                  (1.0 / rb * cosB * y1 / W7 * W9 - W1 / (W7 * W7) * W9 * (y1 / rb - sinB) -
                   W1 / W7 * a / (rb * rb * rb) / cosB * y1 -
                   2.0 * a * y3b / (rb2 * rb2) / cosB * y1)) /
             (M_PI * (1.0 - nu))) +
        b3 / 2.0 *
            (0.25 *
             (N1 * (-sinB * (y1 / rb - sinB) / W7 - 1.0 / W6 * (1.0 + a / rb) +
                    y1 * y1 / (W6 * W6) * (1.0 + a / rb) / rb + y1 * y1 / W6 * a / (rb * rb * rb) +
                    cosB / W7 * W2 - z1b / (W7 * W7) * W2 * (y1 / rb - sinB) -
                    z1b / W7 * a / (rb * rb * rb) * y1) +
              W8 / rb * (a / rb2 + 1.0 / W6) -
              y1 * y1 * W8 / (rb * rb * rb) * (a / rb2 + 1.0 / W6) +
              y1 * W8 / rb * (-2.0 * a / (rb2 * rb2) * y1 - 1.0 / (W6 * W6) / rb * y1) +
              W8 / (W7 * W7) *
                  (sinB * (cosB - a / rb) + z1b / rb * (1.0 + a * y3b / rb2) -
                   1.0 / rb / W7 * (y2 * y2 * cosB * sinB - a * z1b / rb * W1)) *
                  (y1 / rb - sinB) -
              W8 / W7 *
                  (sinB * a / (rb * rb * rb) * y1 + cosB / rb * (1.0 + a * y3b / rb2) -
                   z1b / (rb * rb * rb) * (1.0 + a * y3b / rb2) * y1 -
                   2.0 * z1b / (rb * rb * rb * rb * rb) * a * y3b * y1 +
                   1.0 / (rb * rb * rb) / W7 * (y2 * y2 * cosB * sinB - a * z1b / rb * W1) * y1 +
                   1.0 / rb / (W7 * W7) * (y2 * y2 * cosB * sinB - a * z1b / rb * W1) *
                       (y1 / rb - sinB) -
                   1.0 / rb / W7 *
                       (-a * cosB / rb * W1 + a * z1b / (rb * rb * rb) * W1 * y1 -
                        a * z1b / rb2 * cosB * y1))) /
             (M_PI * (1.0 - nu)));

    // v13
    *v13 =
        b1 / 2.0 *
            (0.25 *
             ((-2.0 + 2.0 * nu) * N1 * rFib_ry3 * cotB * cotB -
              N1 * y2 / (W6 * W6) * ((1.0 - W5) * cotB - y1 / W6 * W4) * (y3b / rb + 1.0) +
              N1 * y2 / W6 *
                  (0.5 * a / (rb * rb * rb) * 2.0 * y3b * cotB +
                   y1 / (W6 * W6) * W4 * (y3b / rb + 1.0) +
                   0.5 * y1 / W6 * a / (rb * rb * rb) * 2.0 * y3b) -
              N1 * y2 * cosB * cotB / (W7 * W7) * W2 * W3 -
              0.5 * N1 * y2 * cosB * cotB / W7 * a / (rb * rb * rb) * 2.0 * y3b +
              a / (rb * rb * rb) * y2 * cotB -
              1.5 * a * y2 * W8 * cotB / (rb * rb * rb * rb * rb) * 2.0 * y3b +
              y2 / rb / W6 * (-N1 * cotB + y1 / W6 * W5 + a * y1 / rb2) -
              0.5 * y2 * W8 / (rb * rb * rb) / W6 * (-N1 * cotB + y1 / W6 * W5 + a * y1 / rb2) *
                  2.0 * y3b -
              y2 * W8 / rb / (W6 * W6) * (-N1 * cotB + y1 / W6 * W5 + a * y1 / rb2) *
                  (y3b / rb + 1.0) +
              y2 * W8 / rb / W6 *
                  (-y1 / (W6 * W6) * W5 * (y3b / rb + 1.0) -
                   0.5 * y1 / W6 * a / (rb * rb * rb) * 2.0 * y3b -
                   a * y1 / (rb2 * rb2) * 2.0 * y3b) +
              y2 / rb / W7 *
                  (cosB / W7 *
                       (W1 * (N1 * cosB - a / rb) * cotB +
                        (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) -
                   a * y3b * cosB * cotB / rb2) -
              0.5 * y2 * W8 / (rb * rb * rb) / W7 *
                  (cosB / W7 *
                       (W1 * (N1 * cosB - a / rb) * cotB +
                        (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) -
                   a * y3b * cosB * cotB / rb2) *
                  2.0 * y3b -
              y2 * W8 / rb / (W7 * W7) *
                  (cosB / W7 *
                       (W1 * (N1 * cosB - a / rb) * cotB +
                        (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) -
                   a * y3b * cosB * cotB / rb2) *
                  W3 +
              y2 * W8 / rb / W7 *
                  (-cosB / (W7 * W7) *
                       (W1 * (N1 * cosB - a / rb) * cotB +
                        (2.0 - 2.0 * nu) * (rb * sinB - y1) * cosB) *
                       W3 +
                   cosB / W7 *
                       ((cosB * y3b / rb + 1.0) * (N1 * cosB - a / rb) * cotB +
                        0.5 * W1 * a / (rb * rb * rb) * 2.0 * y3b * cotB +
                        0.5 * (2.0 - 2.0 * nu) / rb * sinB * 2.0 * y3b * cosB) -
                   a * cosB * cotB / rb2 + a * y3b * cosB * cotB / (rb2 * rb2) * 2.0 * y3b)) /
             (M_PI * (1.0 - nu))) +
        b2 / 2.0 *
            (0.25 *
             (N1 * (((2.0 - 2.0 * nu) * cotB * cotB + nu) * (y3b / rb + 1.0) / W6 -
                    ((2.0 - 2.0 * nu) * cotB * cotB + 1.0) * cosB * W3 / W7) -
              N1 / (W6 * W6) *
                  (-N1 * y1 * cotB + nu * y3b - a + a * y1 * cotB / rb + y1 * y1 / W6 * W4) *
                  (y3b / rb + 1.0) +
              N1 / W6 *
                  (nu - 0.5 * a * y1 * cotB / (rb * rb * rb) * 2.0 * y3b -
                   y1 * y1 / (W6 * W6) * W4 * (y3b / rb + 1.0) -
                   0.5 * y1 * y1 / W6 * a / (rb * rb * rb) * 2.0 * y3b) +
              N1 * cotB / (W7 * W7) * (z1b * cosB - a * (rb * sinB - y1) / rb / cosB) * W3 -
              N1 * cotB / W7 *
                  (cosB * sinB - 0.5 * a / rb2 * sinB * 2.0 * y3b / cosB +
                   0.5 * a * (rb * sinB - y1) / (rb * rb * rb) / cosB * 2.0 * y3b) -
              a / (rb * rb * rb) * y1 * cotB +
              1.5 * a * y1 * W8 * cotB / (rb * rb * rb * rb * rb) * 2.0 * y3b +
              1.0 / W6 *
                  (2.0 * nu + 1.0 / rb * (N1 * y1 * cotB + a) - y1 * y1 / rb / W6 * W5 -
                   a * y1 * y1 / (rb * rb * rb)) -
              W8 / (W6 * W6) *
                  (2.0 * nu + 1.0 / rb * (N1 * y1 * cotB + a) - y1 * y1 / rb / W6 * W5 -
                   a * y1 * y1 / (rb * rb * rb)) *
                  (y3b / rb + 1.0) +
              W8 / W6 *
                  (-0.5 / (rb * rb * rb) * (N1 * y1 * cotB + a) * 2.0 * y3b +
                   0.5 * y1 * y1 / (rb * rb * rb) / W6 * W5 * 2.0 * y3b +
                   y1 * y1 / rb / (W6 * W6) * W5 * (y3b / rb + 1.0) +
                   0.5 * y1 * y1 / (rb2 * rb2) / W6 * a * 2.0 * y3b +
                   1.5 * a * y1 * y1 / (rb * rb * rb * rb * rb) * 2.0 * y3b) +
              cotB / W7 *
                  (-cosB * sinB + a * y1 * y3b / (rb * rb * rb) / cosB +
                   (rb * sinB - y1) / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7 * W9)) -
              W8 * cotB / (W7 * W7) *
                  (-cosB * sinB + a * y1 * y3b / (rb * rb * rb) / cosB +
                   (rb * sinB - y1) / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7 * W9)) *
                  W3 +
              W8 * cotB / W7 *
                  (a / (rb * rb * rb) / cosB * y1 -
                   1.5 * a * y1 * y3b / (rb * rb * rb * rb * rb) / cosB * 2.0 * y3b +
                   0.5 / rb2 * sinB * 2.0 * y3b * ((2.0 - 2.0 * nu) * cosB - W1 / W7 * W9) -
                   0.5 * (rb * sinB - y1) / (rb * rb * rb) *
                       ((2.0 - 2.0 * nu) * cosB - W1 / W7 * W9) * 2.0 * y3b +
                   (rb * sinB - y1) / rb *
                       (-(cosB * y3b / rb + 1.0) / W7 * W9 + W1 / (W7 * W7) * W9 * W3 +
                        0.5 * W1 / W7 * a / (rb * rb * rb) / cosB * 2.0 * y3b))) /
             (M_PI * (1.0 - nu))) +
        b3 / 2.0 *
            (0.25 *
             (N1 * (-y2 / (W6 * W6) * (1.0 + a / rb) * (y3b / rb + 1.0) -
                    0.5 * y2 / W6 * a / (rb * rb * rb) * 2.0 * y3b +
                    y2 * cosB / (W7 * W7) * W2 * W3 +
                    0.5 * y2 * cosB / W7 * a / (rb * rb * rb) * 2.0 * y3b) -
              y2 / rb * (a / rb2 + 1.0 / W6) +
              0.5 * y2 * W8 / (rb * rb * rb) * (a / rb2 + 1.0 / W6) * 2.0 * y3b -
              y2 * W8 / rb * (-a / (rb2 * rb2) * 2.0 * y3b - 1.0 / (W6 * W6) * (y3b / rb + 1.0)) +
              y2 * cosB / rb / W7 * (W1 / W7 * W2 + a * y3b / rb2) -
              0.5 * y2 * W8 * cosB / (rb * rb * rb) / W7 * (W1 / W7 * W2 + a * y3b / rb2) * 2.0 *
                  y3b -
              y2 * W8 * cosB / rb / (W7 * W7) * (W1 / W7 * W2 + a * y3b / rb2) * W3 +
              y2 * W8 * cosB / rb / W7 *
                  ((cosB * y3b / rb + 1.0) / W7 * W2 - W1 / (W7 * W7) * W2 * W3 -
                   0.5 * W1 / W7 * a / (rb * rb * rb) * 2.0 * y3b + a / rb2 -
                   a * y3b / (rb2 * rb2) * 2.0 * y3b)) /
             (M_PI * (1.0 - nu))) +
        b1 / 2.0 *
            (0.25 *
             ((2.0 - 2.0 * nu) * (N1 * rFib_ry1 * cotB - y1 / (W6 * W6) * W5 / rb * y2 -
                                  y2 / W6 * a / (rb * rb * rb) * y1 +
                                  y2 * cosB / (W7 * W7) * W2 * (y1 / rb - sinB) +
                                  y2 * cosB / W7 * a / (rb * rb * rb) * y1) -
              y2 * W8 / (rb * rb * rb) * (2.0 * nu / W6 + a / rb2) * y1 +
              y2 * W8 / rb * (-2.0 * nu / (W6 * W6) / rb * y1 - 2.0 * a / (rb2 * rb2) * y1) -
              y2 * W8 * cosB / (rb * rb * rb) / W7 *
                  (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3b / rb2) * y1 -
              y2 * W8 * cosB / rb / (W7 * W7) * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3b / rb2) *
                  (y1 / rb - sinB) +
              y2 * W8 * cosB / rb / W7 *
                  (-1.0 / rb * cosB * y1 / W7 * W2 + W1 / (W7 * W7) * W2 * (y1 / rb - sinB) +
                   W1 / W7 * a / (rb * rb * rb) * y1 + 2.0 * a * y3b / (rb2 * rb2) * y1)) /
             (M_PI * (1.0 - nu))) +
        b2 / 2.0 *
            (0.25 *
             ((-2.0 + 2.0 * nu) * N1 * cotB * (1.0 / rb * y1 / W6 - cosB * (y1 / rb - sinB) / W7) -
              (2.0 - 2.0 * nu) / W6 * W5 + (2.0 - 2.0 * nu) * y1 * y1 / (W6 * W6) * W5 / rb +
              (2.0 - 2.0 * nu) * y1 * y1 / W6 * a / (rb * rb * rb) +
              (2.0 - 2.0 * nu) * cosB / W7 * W2 -
              (2.0 - 2.0 * nu) * z1b / (W7 * W7) * W2 * (y1 / rb - sinB) -
              (2.0 - 2.0 * nu) * z1b / W7 * a / (rb * rb * rb) * y1 -
              W8 / (rb * rb * rb) * (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb2) * y1 +
              W8 / rb *
                  (-2.0 * nu / W6 + 2.0 * nu * y1 * y1 / (W6 * W6) / rb - a / rb2 +
                   2.0 * a * y1 * y1 / (rb2 * rb2)) +
              W8 / (W7 * W7) *
                  (cosB * sinB + W1 * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) +
                   a / rb * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7)) *
                  (y1 / rb - sinB) -
              W8 / W7 *
                  (1.0 / rb2 * cosB * y1 * cotB * ((2.0 - 2.0 * nu) * cosB - W1 / W7) -
                   W1 * cotB / (rb * rb * rb) * ((2.0 - 2.0 * nu) * cosB - W1 / W7) * y1 +
                   W1 * cotB / rb *
                       (-1.0 / rb * cosB * y1 / W7 + W1 / (W7 * W7) * (y1 / rb - sinB)) -
                   a / (rb * rb * rb) * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7) * y1 +
                   a / rb *
                       (-y3b * cosB / rb2 + 2.0 * y3b * z1b / (rb2 * rb2) * y1 -
                        cosB * W1 / rb / W7 - z1b / rb2 * cosB * y1 / W7 +
                        z1b * W1 / (rb * rb * rb) / W7 * y1 +
                        z1b * W1 / rb / (W7 * W7) * (y1 / rb - sinB)))) /
             (M_PI * (1.0 - nu))) +
        b3 / 2.0 *
            (0.25 *
             ((2.0 - 2.0 * nu) * rFib_ry1 -
              (2.0 - 2.0 * nu) * y2 * sinB / (W7 * W7) * W2 * (y1 / rb - sinB) -
              (2.0 - 2.0 * nu) * y2 * sinB / W7 * a / (rb * rb * rb) * y1 -
              y2 * W8 * sinB / (rb * rb * rb) / W7 * (1.0 + W1 / W7 * W2 + a * y3b / rb2) * y1 -
              y2 * W8 * sinB / rb / (W7 * W7) * (1.0 + W1 / W7 * W2 + a * y3b / rb2) *
                  (y1 / rb - sinB) +
              y2 * W8 * sinB / rb / W7 *
                  (1.0 / rb * cosB * y1 / W7 * W2 - W1 / (W7 * W7) * W2 * (y1 / rb - sinB) -
                   W1 / W7 * a / (rb * rb * rb) * y1 - 2.0 * a * y3b / (rb2 * rb2) * y1)) /
             (M_PI * (1.0 - nu)));

    // v23
    *v23 =
        b1 / 2.0 *
            (0.25 *
             (N1 * (((2.0 - 2.0 * nu) * cotB * cotB - nu) * (y3b / rb + 1.0) / W6 -
                    ((2.0 - 2.0 * nu) * cotB * cotB + 1.0 - 2.0 * nu) * cosB * W3 / W7) +
              N1 / (W6 * W6) * (y1 * cotB * (1.0 - W5) + nu * y3b - a + y2 * y2 / W6 * W4) *
                  (y3b / rb + 1.0) -
              N1 / W6 *
                  (0.5 * a * y1 * cotB / (rb * rb * rb) * 2.0 * y3b + nu -
                   y2 * y2 / (W6 * W6) * W4 * (y3b / rb + 1.0) -
                   0.5 * y2 * y2 / W6 * a / (rb * rb * rb) * 2.0 * y3b) -
              N1 * sinB * cotB / W7 * W2 + N1 * z1b * cotB / (W7 * W7) * W2 * W3 +
              0.5 * N1 * z1b * cotB / W7 * a / (rb * rb * rb) * 2.0 * y3b -
              a / (rb * rb * rb) * y1 * cotB +
              1.5 * a * y1 * W8 * cotB / (rb * rb * rb * rb * rb) * 2.0 * y3b +
              1.0 / W6 *
                  (-2.0 * nu + 1.0 / rb * (N1 * y1 * cotB - a) + y2 * y2 / rb / W6 * W5 +
                   a * y2 * y2 / (rb * rb * rb)) -
              W8 / (W6 * W6) *
                  (-2.0 * nu + 1.0 / rb * (N1 * y1 * cotB - a) + y2 * y2 / rb / W6 * W5 +
                   a * y2 * y2 / (rb * rb * rb)) *
                  (y3b / rb + 1.0) +
              W8 / W6 *
                  (-0.5 / (rb * rb * rb) * (N1 * y1 * cotB - a) * 2.0 * y3b -
                   0.5 * y2 * y2 / (rb * rb * rb) / W6 * W5 * 2.0 * y3b -
                   y2 * y2 / rb / (W6 * W6) * W5 * (y3b / rb + 1.0) -
                   0.5 * y2 * y2 / (rb2 * rb2) / W6 * a * 2.0 * y3b -
                   1.5 * a * y2 * y2 / (rb * rb * rb * rb * rb) * 2.0 * y3b) +
              1.0 / W7 *
                  (cosB * cosB - 1.0 / rb * (N1 * z1b * cotB + a * cosB) +
                   a * y3b * z1b * cotB / (rb * rb * rb) -
                   1.0 / rb / W7 * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * W1)) -
              W8 / (W7 * W7) *
                  (cosB * cosB - 1.0 / rb * (N1 * z1b * cotB + a * cosB) +
                   a * y3b * z1b * cotB / (rb * rb * rb) -
                   1.0 / rb / W7 * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * W1)) *
                  W3 +
              W8 / W7 *
                  (0.5 / (rb * rb * rb) * (N1 * z1b * cotB + a * cosB) * 2.0 * y3b -
                   1.0 / rb * N1 * sinB * cotB + a * z1b * cotB / (rb * rb * rb) +
                   a * y3b * sinB * cotB / (rb * rb * rb) -
                   1.5 * a * y3b * z1b * cotB / (rb * rb * rb * rb * rb) * 2.0 * y3b +
                   0.5 / (rb * rb * rb) / W7 * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * W1) *
                       2.0 * y3b +
                   1.0 / rb / (W7 * W7) * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * W1) * W3 -
                   1.0 / rb / W7 *
                       (-a * sinB * cotB / rb * W1 +
                        0.5 * a * z1b * cotB / (rb * rb * rb) * W1 * 2.0 * y3b -
                        a * z1b * cotB / rb * (cosB * y3b / rb + 1.0)))) /
             (M_PI * (1.0 - nu))) +
        b2 / 2.0 *
            (0.25 *
             ((2.0 - 2.0 * nu) * N1 * rFib_ry3 * cotB * cotB -
              N1 * y2 / (W6 * W6) * ((W5 - 1.0) * cotB + y1 / W6 * W4) * (y3b / rb + 1.0) +
              N1 * y2 / W6 *
                  (-0.5 * a / (rb * rb * rb) * 2.0 * y3b * cotB -
                   y1 / (W6 * W6) * W4 * (y3b / rb + 1.0) -
                   0.5 * y1 / W6 * a / (rb * rb * rb) * 2.0 * y3b) +
              N1 * y2 * cotB / (W7 * W7) * W9 * W3 +
              0.5 * N1 * y2 * cotB / W7 * a / (rb * rb * rb) / cosB * 2.0 * y3b -
              a / (rb * rb * rb) * y2 * cotB +
              1.5 * a * y2 * W8 * cotB / (rb * rb * rb * rb * rb) * 2.0 * y3b +
              y2 / rb / W6 *
                  (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb * (1.0 / rb + 1.0 / W6)) -
              0.5 * y2 * W8 / (rb * rb * rb) / W6 *
                  (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb * (1.0 / rb + 1.0 / W6)) * 2.0 *
                  y3b -
              y2 * W8 / rb / (W6 * W6) *
                  (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb * (1.0 / rb + 1.0 / W6)) *
                  (y3b / rb + 1.0) +
              y2 * W8 / rb / W6 *
                  (2.0 * nu * y1 / (W6 * W6) * (y3b / rb + 1.0) +
                   0.5 * a * y1 / (rb * rb * rb) * (1.0 / rb + 1.0 / W6) * 2.0 * y3b -
                   a * y1 / rb *
                       (-0.5 / (rb * rb * rb) * 2.0 * y3b - 1.0 / (W6 * W6) * (y3b / rb + 1.0))) +
              y2 * cotB / rb / W7 *
                  ((-2.0 + 2.0 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB) -
              0.5 * y2 * W8 * cotB / (rb * rb * rb) / W7 *
                  ((-2.0 + 2.0 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB) * 2.0 * y3b -
              y2 * W8 * cotB / rb / (W7 * W7) *
                  ((-2.0 + 2.0 * nu) * cosB + W1 / W7 * W9 + a * y3b / rb2 / cosB) * W3 +
              y2 * W8 * cotB / rb / W7 *
                  ((cosB * y3b / rb + 1.0) / W7 * W9 - W1 / (W7 * W7) * W9 * W3 -
                   0.5 * W1 / W7 * a / (rb * rb * rb) / cosB * 2.0 * y3b + a / rb2 / cosB -
                   a * y3b / (rb2 * rb2) / cosB * 2.0 * y3b)) /
             (M_PI * (1.0 - nu))) +
        b3 / 2.0 *
            (0.25 *
             (N1 * (-sinB * W3 / W7 + y1 / (W6 * W6) * (1.0 + a / rb) * (y3b / rb + 1.0) +
                    0.5 * y1 / W6 * a / (rb * rb * rb) * 2.0 * y3b + sinB / W7 * W2 -
                    z1b / (W7 * W7) * W2 * W3 - 0.5 * z1b / W7 * a / (rb * rb * rb) * 2.0 * y3b) +
              y1 / rb * (a / rb2 + 1.0 / W6) -
              0.5 * y1 * W8 / (rb * rb * rb) * (a / rb2 + 1.0 / W6) * 2.0 * y3b +
              y1 * W8 / rb * (-a / (rb2 * rb2) * 2.0 * y3b - 1.0 / (W6 * W6) * (y3b / rb + 1.0)) -
              1.0 / W7 *
                  (sinB * (cosB - a / rb) + z1b / rb * (1.0 + a * y3b / rb2) -
                   1.0 / rb / W7 * (y2 * y2 * cosB * sinB - a * z1b / rb * W1)) +
              W8 / (W7 * W7) *
                  (sinB * (cosB - a / rb) + z1b / rb * (1.0 + a * y3b / rb2) -
                   1.0 / rb / W7 * (y2 * y2 * cosB * sinB - a * z1b / rb * W1)) *
                  W3 -
              W8 / W7 *
                  (0.5 * sinB * a / (rb * rb * rb) * 2.0 * y3b + sinB / rb * (1.0 + a * y3b / rb2) -
                   0.5 * z1b / (rb * rb * rb) * (1.0 + a * y3b / rb2) * 2.0 * y3b +
                   z1b / rb * (a / rb2 - a * y3b / (rb2 * rb2) * 2.0 * y3b) +
                   0.5 / (rb * rb * rb) / W7 * (y2 * y2 * cosB * sinB - a * z1b / rb * W1) * 2.0 *
                       y3b +
                   1.0 / rb / (W7 * W7) * (y2 * y2 * cosB * sinB - a * z1b / rb * W1) * W3 -
                   1.0 / rb / W7 *
                       (-a * sinB / rb * W1 + 0.5 * a * z1b / (rb * rb * rb) * W1 * 2.0 * y3b -
                        a * z1b / rb * (cosB * y3b / rb + 1.0)))) /
             (M_PI * (1.0 - nu))) +
        b1 / 2.0 *
            (0.25 *
             ((2.0 - 2.0 * nu) *
                  (N1 * rFib_ry2 * cotB + 1.0 / W6 * W5 - y2 * y2 / (W6 * W6) * W5 / rb -
                   y2 * y2 / W6 * a / (rb * rb * rb) - cosB / W7 * W2 +
                   y2 * y2 * cosB / (W7 * W7) * W2 / rb +
                   y2 * y2 * cosB / W7 * a / (rb * rb * rb)) +
              W8 / rb * (2.0 * nu / W6 + a / rb2) -
              y2 * y2 * W8 / (rb * rb * rb) * (2.0 * nu / W6 + a / rb2) +
              y2 * W8 / rb * (-2.0 * nu / (W6 * W6) / rb * y2 - 2.0 * a / (rb2 * rb2) * y2) +
              W8 * cosB / rb / W7 * (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3b / rb2) -
              y2 * y2 * W8 * cosB / (rb * rb * rb) / W7 *
                  (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3b / rb2) -
              y2 * y2 * W8 * cosB / rb2 / (W7 * W7) *
                  (1.0 - 2.0 * nu - W1 / W7 * W2 - a * y3b / rb2) +
              y2 * W8 * cosB / rb / W7 *
                  (-1.0 / rb * cosB * y2 / W7 * W2 + W1 / (W7 * W7) * W2 / rb * y2 +
                   W1 / W7 * a / (rb * rb * rb) * y2 + 2.0 * a * y3b / (rb2 * rb2) * y2)) /
             (M_PI * (1.0 - nu))) +
        b2 / 2.0 *
            (0.25 *
             ((-2.0 + 2.0 * nu) * N1 * cotB * (1.0 / rb * y2 / W6 - cosB / rb * y2 / W7) +
              (2.0 - 2.0 * nu) * y1 / (W6 * W6) * W5 / rb * y2 +
              (2.0 - 2.0 * nu) * y1 / W6 * a / (rb * rb * rb) * y2 -
              (2.0 - 2.0 * nu) * z1b / (W7 * W7) * W2 / rb * y2 -
              (2.0 - 2.0 * nu) * z1b / W7 * a / (rb * rb * rb) * y2 -
              W8 / (rb * rb * rb) * (N1 * cotB - 2.0 * nu * y1 / W6 - a * y1 / rb2) * y2 +
              W8 / rb * (2.0 * nu * y1 / (W6 * W6) / rb * y2 + 2.0 * a * y1 / (rb2 * rb2) * y2) +
              W8 / (W7 * W7) *
                  (cosB * sinB + W1 * cotB / rb * ((2.0 - 2.0 * nu) * cosB - W1 / W7) +
                   a / rb * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7)) /
                  rb * y2 -
              W8 / W7 *
                  (1.0 / rb2 * cosB * y2 * cotB * ((2.0 - 2.0 * nu) * cosB - W1 / W7) -
                   W1 * cotB / (rb * rb * rb) * ((2.0 - 2.0 * nu) * cosB - W1 / W7) * y2 +
                   W1 * cotB / rb * (-cosB / rb * y2 / W7 + W1 / (W7 * W7) / rb * y2) -
                   a / (rb * rb * rb) * (sinB - y3b * z1b / rb2 - z1b * W1 / rb / W7) * y2 +
                   a / rb *
                       (2.0 * y3b * z1b / (rb2 * rb2) * y2 - z1b / rb2 * cosB * y2 / W7 +
                        z1b * W1 / (rb * rb * rb) / W7 * y2 + z1b * W1 / rb2 / (W7 * W7) * y2))) /
             (M_PI * (1.0 - nu))) +
        b3 / 2.0 *
            (0.25 *
             ((2.0 - 2.0 * nu) * rFib_ry2 + (2.0 - 2.0 * nu) * sinB / W7 * W2 -
              (2.0 - 2.0 * nu) * y2 * y2 * sinB / (W7 * W7) * W2 / rb -
              (2.0 - 2.0 * nu) * y2 * y2 * sinB / W7 * a / (rb * rb * rb) +
              W8 * sinB / rb / W7 * (1.0 + W1 / W7 * W2 + a * y3b / rb2) -
              y2 * y2 * W8 * sinB / (rb * rb * rb) / W7 * (1.0 + W1 / W7 * W2 + a * y3b / rb2) -
              y2 * y2 * W8 * sinB / rb2 / (W7 * W7) * (1.0 + W1 / W7 * W2 + a * y3b / rb2) +
              y2 * W8 * sinB / rb / W7 *
                  (1.0 / rb * cosB * y2 / W7 * W2 - W1 / (W7 * W7) * W2 / rb * y2 -
                   W1 / W7 * a / (rb * rb * rb) * y2 - 2.0 * a * y3b / (rb2 * rb2) * y2)) /
             (M_PI * (1.0 - nu)));
}

/*-----------------------------------------------------------------------------------------------*/

// Function to compute Free Surface Correction to stresses and strains
void AngSetupFSC_S(double X, double Y, double Z, double bX, double bY, double bZ,
                   double PA[3], double PB[3], double mu, double lambda,
                   double Stress[6], double Strain[6]) {
    // Calculate Poisson's ratio
    double nu = 1.0 / (1.0 + lambda / mu) / 2.0;

    // Calculate TD side vector and angle of the angular dislocation pair
    double SideVec[3] = {PB[0] - PA[0], PB[1] - PA[1], PB[2] - PA[2]};
    double eZ[3] = {0.0, 0.0, 1.0};
    double normSideVec = sqrt(SideVec[0] * SideVec[0] + SideVec[1] * SideVec[1] + SideVec[2] * SideVec[2]);
    double beta = acos(-SideVec[2] / normSideVec); // Dot product: SideVec' * eZ = SideVec[2]

    // Check if beta is close to 0 or π
    // double eps = 2.220446049250313e-16; // MATLAB's eps
    if (fabs(beta) < EPSILON || fabs(M_PI - beta) < EPSILON) {
        // Return zero arrays for Stress and Strain
        for (int i = 0; i < 6; i++) {
            Stress[i] = 0.0;
            Strain[i] = 0.0;
        }
        return;
    }

    // Calculate transformation matrix A
    double ey1[3] = {SideVec[0], SideVec[1], 0.0};
    double normEy1 = sqrt(ey1[0] * ey1[0] + ey1[1] * ey1[1]);
    ey1[0] /= normEy1;
    ey1[1] /= normEy1;
    ey1[2] = 0.0;

    double ey3[3] = {0.0, 0.0, -1.0};
    double ey2[3] = {ey3[1] * ey1[2] - ey3[2] * ey1[1],
                     ey3[2] * ey1[0] - ey3[0] * ey1[2],
                     ey3[0] * ey1[1] - ey3[1] * ey1[0]}; // Cross product: ey3 x ey1

    double A[9] = {ey1[0], ey2[0], ey3[0],
                   ey1[1], ey2[1], ey3[1],
                   ey1[2], ey2[2], ey3[2]};

    // Transform coordinates from EFCS to the first ADCS (point A)
    double y1A, y2A, y3A;
    CoordTrans(X - PA[0], Y - PA[1], Z - PA[2], A, &y1A, &y2A, &y3A);

    // Transform SideVec to ADCS for the second ADCS (point B)
    double y1AB, y2AB, y3AB;
    CoordTrans(SideVec[0], SideVec[1], SideVec[2], A, &y1AB, &y2AB, &y3AB);

    double y1B = y1A - y1AB;
    double y2B = y2A - y2AB;
    double y3B = y3A - y3AB;

    // Transform slip vector components from EFCS to ADCS
    double b1, b2, b3;
    CoordTrans(bX, bY, bZ, A, &b1, &b2, &b3);

    // Determine the configuration based on artifact-free condition
    int I = (beta * y1A) >= 0;

    // Initialize strain components for points A and B
    double v11A = 0.0, v22A = 0.0, v33A = 0.0, v12A = 0.0, v13A = 0.0, v23A = 0.0;
    double v11B = 0.0, v22B = 0.0, v33B = 0.0, v12B = 0.0, v13B = 0.0, v23B = 0.0;

    // Configuration I
    if (I) {
        AngDisStrainFSC(-y1A, -y2A, y3A, M_PI - beta, -b1, -b2, b3, nu, -PA[2], &v11A, &v22A, &v33A, &v12A, &v13A, &v23A);
        v13A = -v13A; // Adjust sign for v13
        v23A = -v23A; // Adjust sign for v23

        AngDisStrainFSC(-y1B, -y2B, y3B, M_PI - beta, -b1, -b2, b3, nu, -PB[2], &v11B, &v22B, &v33B, &v12B, &v13B, &v23B);
        v13B = -v13B; // Adjust sign for v13
        v23B = -v23B; // Adjust sign for v23
    }
    // Configuration II
    else {
        AngDisStrainFSC(y1A, y2A, y3A, beta, b1, b2, b3, nu, -PA[2], &v11A, &v22A, &v33A, &v12A, &v13A, &v23A);
        AngDisStrainFSC(y1B, y2B, y3B, beta, b1, b2, b3, nu, -PB[2], &v11B, &v22B, &v33B, &v12B, &v13B, &v23B);
    }

    // Calculate total Free Surface Correction to strains in ADCS
    double v11 = v11B - v11A;
    double v22 = v22B - v22A;
    double v33 = v33B - v33A;
    double v12 = v12B - v12A;
    double v13 = v13B - v13A;
    double v23 = v23B - v23A;

    // Transform strains from ADCS to EFCS
    double Exx, Eyy, Ezz, Exy, Exz, Eyz;
    double AT[9] = {A[0], A[3], A[6],
                    A[1], A[4], A[7],
                    A[2], A[5], A[8]}; // Transpose of A
    TensTrans(v11, v22, v33, v12, v13, v23, AT, &Exx, &Eyy, &Ezz, &Exy, &Exz, &Eyz);

    // Calculate stresses in EFCS
    Stress[0] = 2.0 * mu * Exx + lambda * (Exx + Eyy + Ezz); // Sxx
    Stress[1] = 2.0 * mu * Eyy + lambda * (Exx + Eyy + Ezz); // Syy
    Stress[2] = 2.0 * mu * Ezz + lambda * (Exx + Eyy + Ezz); // Szz
    Stress[3] = 2.0 * mu * Exy; // Sxy
    Stress[4] = 2.0 * mu * Exz; // Sxz
    Stress[5] = 2.0 * mu * Eyz; // Syz

    // Assign strain components
    Strain[0] = Exx;
    Strain[1] = Eyy;
    Strain[2] = Ezz;
    Strain[3] = Exy;
    Strain[4] = Exz;
    Strain[5] = Eyz;
}


/*-----------------------------------------------------------------------------------------------*/

// Function to compute harmonic function contribution to stresses and strains
void TDstress_HarFunc(double X, double Y, double Z, double P1[3], double P2[3], double P3[3],
                      double Ss, double Ds, double Ts, double mu, double lambda,
                      double Stress[6], double Strain[6]) {
    // Assign slip components
    double bx = Ts; // Tensile-slip
    double by = Ss; // Strike-slip
    double bz = Ds; // Dip-slip

    // Calculate unit normal vector to the triangular dislocation (TD)
    double Vnorm[3];
    Vnorm[0] = (P2[1] - P1[1]) * (P3[2] - P1[2]) - (P2[2] - P1[2]) * (P3[1] - P1[1]); // Cross product: (P2-P1) x (P3-P1)
    Vnorm[1] = (P2[2] - P1[2]) * (P3[0] - P1[0]) - (P2[0] - P1[0]) * (P3[2] - P1[2]);
    Vnorm[2] = (P2[0] - P1[0]) * (P3[1] - P1[1]) - (P2[1] - P1[1]) * (P3[0] - P1[0]);
    double normVnorm = sqrt(Vnorm[0] * Vnorm[0] + Vnorm[1] * Vnorm[1] + Vnorm[2] * Vnorm[2]);
    Vnorm[0] /= normVnorm;
    Vnorm[1] /= normVnorm;
    Vnorm[2] /= normVnorm;

    // Define unit vectors
    double eX[3] = {1.0, 0.0, 0.0};
    double eY[3] = {0.0, 1.0, 0.0}; // Northward direction
    double eZ[3] = {0.0, 0.0, 1.0}; // Upward direction

    // Calculate strike vector (Vstrike = eZ x Vnorm)
    double Vstrike[3];
    Vstrike[0] = eZ[1] * Vnorm[2] - eZ[2] * Vnorm[1];
    Vstrike[1] = eZ[2] * Vnorm[0] - eZ[0] * Vnorm[2];
    Vstrike[2] = eZ[0] * Vnorm[1] - eZ[1] * Vnorm[0];

    // Handle special case where Vstrike is zero (horizontal TD)
    double normVstrike = sqrt(Vstrike[0] * Vstrike[0] + Vstrike[1] * Vstrike[1] + Vstrike[2] * Vstrike[2]);
    if (normVstrike == 0.0) {
        Vstrike[0] = eX[0] * Vnorm[2];
        Vstrike[1] = eX[1] * Vnorm[2];
        Vstrike[2] = eX[2] * Vnorm[2];
        normVstrike = sqrt(Vstrike[0] * Vstrike[0] + Vstrike[1] * Vstrike[1] + Vstrike[2] * Vstrike[2]);
    }
    Vstrike[0] /= normVstrike;
    Vstrike[1] /= normVstrike;
    Vstrike[2] /= normVstrike;

    // Calculate dip vector (Vdip = Vnorm x Vstrike)
    double Vdip[3];
    Vdip[0] = Vnorm[1] * Vstrike[2] - Vnorm[2] * Vstrike[1];
    Vdip[1] = Vnorm[2] * Vstrike[0] - Vnorm[0] * Vstrike[2];
    Vdip[2] = Vnorm[0] * Vstrike[1] - Vnorm[1] * Vstrike[0];

    // Define transformation matrix A (columns: Vnorm, Vstrike, Vdip)
    double A[9] = {
        Vnorm[0], Vstrike[0], Vdip[0],
        Vnorm[1], Vstrike[1], Vdip[1],
        Vnorm[2], Vstrike[2], Vdip[2]
    };

    // Transform slip vector components from TDCS to EFCS
    double bX, bY, bZ;
    CoordTrans(bx, by, bz, A, &bX, &bY, &bZ);

    // Calculate contribution of angular dislocation pairs for each TD side
    double Stress1[6], Strain1[6];
    double Stress2[6], Strain2[6];
    double Stress3[6], Strain3[6];

    AngSetupFSC_S(X, Y, Z, bX, bY, bZ, P1, P2, mu, lambda, Stress1, Strain1); // Side P1P2
    AngSetupFSC_S(X, Y, Z, bX, bY, bZ, P2, P3, mu, lambda, Stress2, Strain2); // Side P2P3
    AngSetupFSC_S(X, Y, Z, bX, bY, bZ, P3, P1, mu, lambda, Stress3, Strain3); // Side P3P1

    // Calculate total harmonic function contribution to stresses and strains
    for (int i = 0; i < 6; i++) {
        Stress[i] = Stress1[i] + Stress2[i] + Stress3[i];
        Strain[i] = Strain1[i] + Strain2[i] + Strain3[i];
    }
}


/*-----------------------------------------------------------------------------------------------*/
