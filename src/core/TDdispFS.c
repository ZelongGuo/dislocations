#include <math.h>
#include <stdlib.h>
#include "mehdi.h"

/*---------------------------------------------------------
 *  Main function used for calculating disp in full space.
 *
 *  Referring to the Matlab codes of Meade
 *
 *  Author: Zelong Guo
 *  03.2025, @ Potsdam, Germany
 *  zelong.guo@outlook.com
 *
 * ---------------------------------------------------------*/

void TDdispFS(double X, double Y, double Z, double P1[3], double P2[3], double P3[3],
              double Ss, double Ds, double Ts, double nu,
              double *ue, double *un, double *uv) {
    // Slip components
    double bx = Ts;  // Tensile-slip
    double by = Ss;  // Strike-slip
    double bz = Ds;  // Dip-slip

    // Calculate unit strike, dip and normal vectors
    double Vnorm[3], Vstrike[3], Vdip[3];

    // Vnorm = cross(P2-P1, P3-P1)
    Vnorm[0] = (P2[1] - P1[1]) * (P3[2] - P1[2]) - (P2[2] - P1[2]) * (P3[1] - P1[1]);
    Vnorm[1] = (P2[2] - P1[2]) * (P3[0] - P1[0]) - (P2[0] - P1[0]) * (P3[2] - P1[2]);
    Vnorm[2] = (P2[0] - P1[0]) * (P3[1] - P1[1]) - (P2[1] - P1[1]) * (P3[0] - P1[0]);

    // Normalize Vnorm
    double norm_Vnorm = sqrt(Vnorm[0] * Vnorm[0] + Vnorm[1] * Vnorm[1] + Vnorm[2] * Vnorm[2]);
    Vnorm[0] /= norm_Vnorm;
    Vnorm[1] /= norm_Vnorm;
    Vnorm[2] /= norm_Vnorm;

    // Base vectors
    double eY[3] = {0.0, 1.0, 0.0};
    double eZ[3] = {0.0, 0.0, 1.0};

    // Vstrike = cross(eZ, Vnorm)
    Vstrike[0] = eZ[1] * Vnorm[2] - eZ[2] * Vnorm[1];
    Vstrike[1] = eZ[2] * Vnorm[0] - eZ[0] * Vnorm[2];
    Vstrike[2] = eZ[0] * Vnorm[1] - eZ[1] * Vnorm[0];

    // Check if Vstrike is zero and adjust
    double norm_Vstrike = sqrt(Vstrike[0] * Vstrike[0] + Vstrike[1] * Vstrike[1] + Vstrike[2] * Vstrike[2]);
    if (norm_Vstrike == 0.0) {
        Vstrike[0] = eY[0] * Vnorm[2];
        Vstrike[1] = eY[1] * Vnorm[2];
        Vstrike[2] = eY[2] * Vnorm[2];
        norm_Vstrike = sqrt(Vstrike[0] * Vstrike[0] + Vstrike[1] * Vstrike[1] + Vstrike[2] * Vstrike[2]);
    }

    // Normalize Vstrike
    Vstrike[0] /= norm_Vstrike;
    Vstrike[1] /= norm_Vstrike;
    Vstrike[2] /= norm_Vstrike;

    // Vdip = cross(Vnorm, Vstrike)
    Vdip[0] = Vnorm[1] * Vstrike[2] - Vnorm[2] * Vstrike[1];
    Vdip[1] = Vnorm[2] * Vstrike[0] - Vnorm[0] * Vstrike[2];
    Vdip[2] = Vnorm[0] * Vstrike[1] - Vnorm[1] * Vstrike[0];

    // Transformation matrix At (3x3 stored as 1D array of 9 elements)
    double At[9] = {
        Vnorm[0],   Vnorm[1],   Vnorm[2],
        Vstrike[0], Vstrike[1], Vstrike[2],
        Vdip[0],    Vdip[1],    Vdip[2]
    };

    // For converting back
    double AT[9] = {
        Vnorm[0], Vstrike[0], Vdip[0],
        Vnorm[1], Vstrike[1], Vdip[1],
        Vnorm[2], Vstrike[2], Vdip[2]
    };

    // Transform coordinates from EFCS to TDCS
    double x, y, z;
    CoordTrans(X - P2[0], Y - P2[1], Z - P2[2], At, &x, &y, &z);

    double p1[3], p2[3] = {0.0, 0.0, 0.0}, p3[3];
    CoordTrans(P1[0] - P2[0], P1[1] - P2[1], P1[2] - P2[2], At, &p1[0], &p1[1], &p1[2]);
    CoordTrans(P3[0] - P2[0], P3[1] - P2[1], P3[2] - P2[2], At, &p3[0], &p3[1], &p3[2]);

    // Calculate unit vectors along TD sides
    double e12[3], e13[3], e23[3];
    double norm_e12, norm_e13, norm_e23;

    for (int i = 0; i < 3; i++) {
        e12[i] = p2[i] - p1[i];
        e13[i] = p3[i] - p1[i];
        e23[i] = p3[i] - p2[i];
    }

    norm_e12 = sqrt(e12[0] * e12[0] + e12[1] * e12[1] + e12[2] * e12[2]);
    norm_e13 = sqrt(e13[0] * e13[0] + e13[1] * e13[1] + e13[2] * e13[2]);
    norm_e23 = sqrt(e23[0] * e23[0] + e23[1] * e23[1] + e23[2] * e23[2]);

    for (int i = 0; i < 3; i++) {
        e12[i] /= norm_e12;
        e13[i] /= norm_e13;
        e23[i] /= norm_e23;
    }

    // Calculate TD angles
    double A = acos(e12[0] * e13[0] + e12[1] * e13[1] + e12[2] * e13[2]);
    double B = acos(-(e12[0] * e23[0] + e12[1] * e23[1] + e12[2] * e23[2]));
    double C = acos(e23[0] * e13[0] + e23[1] * e13[1] + e23[2] * e13[2]);

    // Determine configuration
    double p1_23[2] = {p1[1], p1[2]};
    double p2_23[2] = {p2[1], p2[2]};
    double p3_23[2] = {p3[1], p3[2]};
    int Trimode;
    trimodefinder(y, z, x, p1_23, p2_23, p3_23, &Trimode);

    // Displacement components
    double u = 0.0, v = 0.0, w = 0.0;
    double u1, v1, w1, u2, v2, w2, u3, v3, w3;

    if (Trimode == 1) {  // Configuration I
        double neg_e13[3] = {-e13[0], -e13[1], -e13[2]};
        TDSetupD(x, y, z, A, bx, by, bz, nu, p1, neg_e13, &u1, &v1, &w1);
        TDSetupD(x, y, z, B, bx, by, bz, nu, p2, e12, &u2, &v2, &w2);
        TDSetupD(x, y, z, C, bx, by, bz, nu, p3, e23, &u3, &v3, &w3);
        u = u1 + u2 + u3;
        v = v1 + v2 + v3;
        w = w1 + w2 + w3;
    }
    else if (Trimode == -1) {  // Configuration II
        double neg_e12[3] = {-e12[0], -e12[1], -e12[2]};
        double neg_e23[3] = {-e23[0], -e23[1], -e23[2]};
        TDSetupD(x, y, z, A, bx, by, bz, nu, p1, e13, &u1, &v1, &w1);
        TDSetupD(x, y, z, B, bx, by, bz, nu, p2, neg_e12, &u2, &v2, &w2);
        TDSetupD(x, y, z, C, bx, by, bz, nu, p3, neg_e23, &u3, &v3, &w3);
        u = u1 + u2 + u3;
        v = v1 + v2 + v3;
        w = w1 + w2 + w3;
    }
    else {  // Trimode == 0
        u = NAN;
        v = NAN;
        w = NAN;
    }

    // Burgers' function contribution
    double a[3] = {-x, p1[1] - y, p1[2] - z};
    double b[3] = {-x, -y, -z};
    double c[3] = {-x, p3[1] - y, p3[2] - z};

    double na = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    double nb = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
    double nc = sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);

    double numerator = a[0] * (b[1] * c[2] - b[2] * c[1]) -
                      a[1] * (b[0] * c[2] - b[2] * c[0]) +
                      a[2] * (b[0] * c[1] - b[1] * c[0]);
    double denominator = na * nb * nc + 
                        (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) * nc +
                        (a[0] * c[0] + a[1] * c[1] + a[2] * c[2]) * nb +
                        (b[0] * c[0] + b[1] * c[1] + b[2] * c[2]) * na;

    double Fi = -2.0 * atan2(numerator, denominator) / (4.0 * M_PI);

    // Complete displacement in TDCS
    u = bx * Fi + u;
    v = by * Fi + v;
    w = bz * Fi + w;

    // Transform back to EFCS
    CoordTrans(u, v, w, AT, ue, un, uv);
}
