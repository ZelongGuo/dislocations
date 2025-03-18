#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mehdi.h"

/*-----------------------------------------------------------------------------------------------
 * For TDdispHS.c
 *
 * Zelong Guo, @ Potsdam, Germany, 2025
 * zelong.guo@outlook.com
 *
 ------------------------------------------------------------------------------------------------*/

void TDdispFS4HS(double X, double Y, double Z, double P1[3], double P2[3], double P3[3],
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
    // For horizontal elements in case of half-space calculation!!!
    // Correct the strike vector of image dislocation only
        if (P1[3] > 0) {
            Vstrike[0] = -Vstrike[0];
            Vstrike[1] = -Vstrike[1];
            Vstrike[2] = -Vstrike[2];
        }
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

 /* ------------------------------------------------------------------------------------------------*/

void AngDisDispFSC(double y1, double y2, double y3, double beta, double b1, double b2, double b3, double nu, double a,
                   double *v1, double *v2, double *v3) {
    // Trigonometric calculations
    double sinB = sin(beta);
    double cosB = cos(beta);
    double cotB = 1.0 / tan(beta); // Cotangent as 1/tan(beta)
    double y3b = y3 + 2 * a;
    double z1b = y1 * cosB + y3b * sinB;
    double z3b = -y1 * sinB + y3b * cosB;
    double r2b = y1 * y1 + y2 * y2 + y3b * y3b;
    double rb = sqrt(r2b);

    // Burgers' function
    double Fib = 2.0 * atan(-y2 / (-(rb + y3b) * (1.0 / tan(beta / 2.0)) + y1));

    // Calculate v1 components
    double v1cb1 = b1 / (4.0 * M_PI * (1.0 - nu)) * (
        -2.0 * (1.0 - nu) * (1.0 - 2.0 * nu) * Fib * cotB * cotB +
        (1.0 - 2.0 * nu) * y2 / (rb + y3b) * ((1.0 - 2.0 * nu - a / rb) * cotB - y1 / (rb + y3b) * (nu + a / rb)) +
        (1.0 - 2.0 * nu) * y2 * cosB * cotB / (rb + z3b) * (cosB + a / rb) +
        a * y2 * (y3b - a) * cotB / (rb * rb * rb) +
        y2 * (y3b - a) / (rb * (rb + y3b)) * (-(1.0 - 2.0 * nu) * cotB + y1 / (rb + y3b) * (2.0 * nu + a / rb) + a * y1 / (rb * rb)) +
        y2 * (y3b - a) / (rb * (rb + z3b)) * (
            cosB / (rb + z3b) * ((rb * cosB + y3b) * ((1.0 - 2.0 * nu) * cosB - a / rb) * cotB + 2.0 * (1.0 - nu) * (rb * sinB - y1) * cosB) -
            a * y3b * cosB * cotB / (rb * rb)
        )
    );

    double v1cb2 = b2 / (4.0 * M_PI * (1.0 - nu)) * (
        (1.0 - 2.0 * nu) * ((2.0 * (1.0 - nu) * cotB * cotB + nu) * log(rb + y3b) - (2.0 * (1.0 - nu) * cotB * cotB + 1.0) * cosB * log(rb + z3b)) +
        (1.0 - 2.0 * nu) / (rb + y3b) * (-(1.0 - 2.0 * nu) * y1 * cotB + nu * y3b - a + a * y1 * cotB / rb + y1 * y1 / (rb + y3b) * (nu + a / rb)) -
        (1.0 - 2.0 * nu) * cotB / (rb + z3b) * (z1b * cosB - a * (rb * sinB - y1) / (rb * cosB)) -
        a * y1 * (y3b - a) * cotB / (rb * rb * rb) +
        (y3b - a) / (rb + y3b) * (2.0 * nu + 1.0 / rb * ((1.0 - 2.0 * nu) * y1 * cotB + a) - y1 * y1 / (rb * (rb + y3b)) * (2.0 * nu + a / rb) - a * y1 * y1 / (rb * rb * rb)) +
        (y3b - a) * cotB / (rb + z3b) * (
            -cosB * sinB + a * y1 * y3b / (rb * rb * rb * cosB) +
            (rb * sinB - y1) / rb * (2.0 * (1.0 - nu) * cosB - (rb * cosB + y3b) / (rb + z3b) * (1.0 + a / (rb * cosB)))
        )
    );

    double v1cb3 = b3 / (4.0 * M_PI * (1.0 - nu)) * (
        (1.0 - 2.0 * nu) * (y2 / (rb + y3b) * (1.0 + a / rb) - y2 * cosB / (rb + z3b) * (cosB + a / rb)) -
        y2 * (y3b - a) / rb * (a / (rb * rb) + 1.0 / (rb + y3b)) +
        y2 * (y3b - a) * cosB / (rb * (rb + z3b)) * ((rb * cosB + y3b) / (rb + z3b) * (cosB + a / rb) + a * y3b / (rb * rb))
    );

    // Calculate v2 components
    double v2cb1 = b1 / (4.0 * M_PI * (1.0 - nu)) * (
        (1.0 - 2.0 * nu) * ((2.0 * (1.0 - nu) * cotB * cotB - nu) * log(rb + y3b) - (2.0 * (1.0 - nu) * cotB * cotB + 1.0 - 2.0 * nu) * cosB * log(rb + z3b)) -
        (1.0 - 2.0 * nu) / (rb + y3b) * (y1 * cotB * (1.0 - 2.0 * nu - a / rb) + nu * y3b - a + y2 * y2 / (rb + y3b) * (nu + a / rb)) -
        (1.0 - 2.0 * nu) * z1b * cotB / (rb + z3b) * (cosB + a / rb) -
        a * y1 * (y3b - a) * cotB / (rb * rb * rb) +
        (y3b - a) / (rb + y3b) * (-2.0 * nu + 1.0 / rb * ((1.0 - 2.0 * nu) * y1 * cotB - a) + y2 * y2 / (rb * (rb + y3b)) * (2.0 * nu + a / rb) + a * y2 * y2 / (rb * rb * rb)) +
        (y3b - a) / (rb + z3b) * (
            cosB * cosB - 1.0 / rb * ((1.0 - 2.0 * nu) * z1b * cotB + a * cosB) +
            a * y3b * z1b * cotB / (rb * rb * rb) -
            1.0 / (rb * (rb + z3b)) * (y2 * y2 * cosB * cosB - a * z1b * cotB / rb * (rb * cosB + y3b))
        )
    );

    double v2cb2 = b2 / (4.0 * M_PI * (1.0 - nu)) * (
        2.0 * (1.0 - nu) * (1.0 - 2.0 * nu) * Fib * cotB * cotB +
        (1.0 - 2.0 * nu) * y2 / (rb + y3b) * (-(1.0 - 2.0 * nu - a / rb) * cotB + y1 / (rb + y3b) * (nu + a / rb)) -
        (1.0 - 2.0 * nu) * y2 * cotB / (rb + z3b) * (1.0 + a / (rb * cosB)) -
        a * y2 * (y3b - a) * cotB / (rb * rb * rb) +
        y2 * (y3b - a) / (rb * (rb + y3b)) * ((1.0 - 2.0 * nu) * cotB - 2.0 * nu * y1 / (rb + y3b) - a * y1 / rb * (1.0 / rb + 1.0 / (rb + y3b))) +
        y2 * (y3b - a) * cotB / (rb * (rb + z3b)) * (
            -2.0 * (1.0 - nu) * cosB + (rb * cosB + y3b) / (rb + z3b) * (1.0 + a / (rb * cosB)) + a * y3b / (rb * rb * cosB)
        )
    );

    double v2cb3 = b3 / (4.0 * M_PI * (1.0 - nu)) * (
        (1.0 - 2.0 * nu) * (-sinB * log(rb + z3b) - y1 / (rb + y3b) * (1.0 + a / rb) + z1b / (rb + z3b) * (cosB + a / rb)) +
        y1 * (y3b - a) / rb * (a / (rb * rb) + 1.0 / (rb + y3b)) -
        (y3b - a) / (rb + z3b) * (
            sinB * (cosB - a / rb) + z1b / rb * (1.0 + a * y3b / (rb * rb)) -
            1.0 / (rb * (rb + z3b)) * (y2 * y2 * cosB * sinB - a * z1b / rb * (rb * cosB + y3b))
        )
    );

    // Calculate v3 components
    double v3cb1 = b1 / (4.0 * M_PI * (1.0 - nu)) * (
        2.0 * (1.0 - nu) * (((1.0 - 2.0 * nu) * Fib * cotB) + (y2 / (rb + y3b) * (2.0 * nu + a / rb)) - (y2 * cosB / (rb + z3b) * (cosB + a / rb))) +
        y2 * (y3b - a) / rb * (2.0 * nu / (rb + y3b) + a / (rb * rb)) +
        y2 * (y3b - a) * cosB / (rb * (rb + z3b)) * (
            1.0 - 2.0 * nu - (rb * cosB + y3b) / (rb + z3b) * (cosB + a / rb) - a * y3b / (rb * rb)
        )
    );

    double v3cb2 = b2 / (4.0 * M_PI * (1.0 - nu)) * (
        -2.0 * (1.0 - nu) * (1.0 - 2.0 * nu) * cotB * (log(rb + y3b) - cosB * log(rb + z3b)) -
        2.0 * (1.0 - nu) * y1 / (rb + y3b) * (2.0 * nu + a / rb) +
        2.0 * (1.0 - nu) * z1b / (rb + z3b) * (cosB + a / rb) +
        (y3b - a) / rb * ((1.0 - 2.0 * nu) * cotB - 2.0 * nu * y1 / (rb + y3b) - a * y1 / (rb * rb)) -
        (y3b - a) / (rb + z3b) * (
            cosB * sinB + (rb * cosB + y3b) * cotB / rb * (2.0 * (1.0 - nu) * cosB - (rb * cosB + y3b) / (rb + z3b)) +
            a / rb * (sinB - y3b * z1b / (rb * rb) - z1b * (rb * cosB + y3b) / (rb * (rb + z3b)))
        )
    );

    double v3cb3 = b3 / (4.0 * M_PI * (1.0 - nu)) * (
        2.0 * (1.0 - nu) * Fib +
        2.0 * (1.0 - nu) * (y2 * sinB / (rb + z3b) * (cosB + a / rb)) +
        y2 * (y3b - a) * sinB / (rb * (rb + z3b)) * (
            1.0 + (rb * cosB + y3b) / (rb + z3b) * (cosB + a / rb) + a * y3b / (rb * rb)
        )
    );

    // Final displacement components
    *v1 = v1cb1 + v1cb2 + v1cb3;
    *v2 = v2cb1 + v2cb2 + v2cb3;
    *v3 = v3cb1 + v3cb2 + v3cb3;
}

 /* ------------------------------------------------------------------------------------------------*/

// AngSetupFSC calculates the Free Surface Correction to displacements 
// associated with angular dislocation pair on each TD side.
void AngSetupFSC(double X, double Y, double Z, double bX, double bY, double bZ,
                 double PA[3], double PB[3], double nu, double *ue, double *un, double *uv) {
    // Calculate TD side vector
    double SideVec[3] = {PB[0] - PA[0], PB[1] - PA[1], PB[2] - PA[2]};

    // Define the unit vector along Z-axis
    double eZ[3] = {0, 0, 1};

    // Calculate the angle of the angular dislocation pair
    double dotProduct = SideVec[0] * eZ[0] + SideVec[1] * eZ[1] + SideVec[2] * eZ[2];
    double normSideVec = sqrt(SideVec[0] * SideVec[0] + SideVec[1] * SideVec[1] + SideVec[2] * SideVec[2]);
    double beta = acos(-dotProduct / normSideVec);

    if (fabs(beta) < EPSILON || fabs(M_PI - beta) < EPSILON) {
        // If the angle is close to 0 or pi, the correction is zero
        *ue = 0;
        *un = 0;
        *uv = 0;
    }
    else {
        // Define transformation matrix A
        double ey1[3] = {SideVec[0], SideVec[1], 0};
        double normEy1 = sqrt(ey1[0] * ey1[0] + ey1[1] * ey1[1]);
        ey1[0] /= normEy1;
        ey1[1] /= normEy1;
        ey1[2] = 0;

        double ey3[3] = {-eZ[0], -eZ[1], -eZ[2]};
        double ey2[3] = {ey3[1] * ey1[2] - ey3[2] * ey1[1],
                         ey3[2] * ey1[0] - ey3[0] * ey1[2],
                         ey3[0] * ey1[1] - ey3[1] * ey1[0]};
        double A[9] = {
            ey1[0], ey2[0], ey3[0],
            ey1[1], ey2[1], ey3[1],
            ey1[2], ey2[2], ey3[2]};

        double AT[9] = {
            ey1[0], ey1[1], ey1[2],
            ey2[0], ey2[1], ey2[2],
            ey3[0], ey3[1], ey3[2]};

        // Transform coordinates from EFCS to the first ADCS
        double y1A, y2A, y3A;
        CoordTrans(X - PA[0], Y - PA[1], Z - PA[2], A, &y1A, &y2A, &y3A);

        // Transform coordinates from EFCS to the second ADCS
        double y1AB, y2AB, y3AB;
        CoordTrans(SideVec[0], SideVec[1], SideVec[2], A, &y1AB, &y2AB, &y3AB);

        double y1B = y1A - y1AB;
        double y2B = y2A - y2AB;
        double y3B = y3A - y3AB;

        // Transform slip vector components from EFCS to ADCS
        double b1, b2, b3;
        CoordTrans(bX, bY, bZ, A, &b1, &b2, &b3);

        // Determine the best artefact-free configuration for the calculation
        // points near the free surface
        int I = (beta * y1A) >= 0;

        // Define variables for displacement corrections
        double v1A, v2A, v3A, v1B, v2B, v3B;

        // Configuration I
        if (I) {
            AngDisDispFSC(y1A, y2A, y3A, -M_PI + beta, b1, b2, b3, nu, -PA[2], &v1A, &v2A, &v3A);
            AngDisDispFSC(y1B, y2B, y3B, -M_PI + beta, b1, b2, b3, nu, -PB[2], &v1B, &v2B, &v3B);
        }
        // Configuration II
        else {
            AngDisDispFSC(y1A, y2A, y3A, beta, b1, b2, b3, nu, -PA[2], &v1A, &v2A, &v3A);
            AngDisDispFSC(y1B, y2B, y3B, beta, b1, b2, b3, nu, -PB[2], &v1B, &v2B, &v3B);
        }

        // Calculate total Free Surface Correction to displacements in ADCS
        double v1 = v1B - v1A;
        double v2 = v2B - v2A;
        double v3 = v3B - v3A;

        // Transform total Free Surface Correction to displacements from ADCS to EFCS
        CoordTrans(v1, v2, v3, AT, ue, un, uv);
    }
}


 /* ------------------------------------------------------------------------------------------------*/

// TDdisp_HarFunc calculates the harmonic function contribution to the
// displacements associated with a triangular dislocation in a half-space.
void TDdisp_HarFunc(double X, double Y, double Z, double P1[3], double P2[3], double P3[3], 
                    double Ss, double Ds, double Ts, double nu, double *ue, double *un, double *uv) {
    // Define slip vector components
    double bx = Ts; // Tensile-slip
    double by = Ss; // Strike-slip
    double bz = Ds; // Dip-slip

    // Calculate unit normal vector to the triangular dislocation (TD)
    double Vnorm[3] = {
        (P2[1] - P1[1]) * (P3[2] - P1[2]) - (P2[2] - P1[2]) * (P3[1] - P1[1]),
        (P2[2] - P1[2]) * (P3[0] - P1[0]) - (P2[0] - P1[0]) * (P3[2] - P1[2]),
        (P2[0] - P1[0]) * (P3[1] - P1[1]) - (P2[1] - P1[1]) * (P3[0] - P1[0])
    };

    double normV = sqrt(Vnorm[0] * Vnorm[0] + Vnorm[1] * Vnorm[1] + Vnorm[2] * Vnorm[2]);
    Vnorm[0] /= normV;
    Vnorm[1] /= normV;
    Vnorm[2] /= normV;

    // Define standard unit vectors
    double eY[3] = {0, 1, 0};
    double eZ[3] = {0, 0, 1};

    // Calculate strike vector
    double Vstrike[3] = {
        eZ[1] * Vnorm[2] - eZ[2] * Vnorm[1],
        eZ[2] * Vnorm[0] - eZ[0] * Vnorm[2],
        eZ[0] * Vnorm[1] - eZ[1] * Vnorm[0]
    };

    // Handle case where the normal vector is purely vertical
    double normVstrike = sqrt(Vstrike[0] * Vstrike[0] + Vstrike[1] * Vstrike[1] + Vstrike[2] * Vstrike[2]);
    if (normVstrike == 0) {
        Vstrike[0] = eY[0] * Vnorm[2];
        Vstrike[1] = eY[1] * Vnorm[2];
        Vstrike[2] = eY[2] * Vnorm[2];
    } else {
        Vstrike[0] /= normVstrike;
        Vstrike[1] /= normVstrike;
        Vstrike[2] /= normVstrike;
    }

    // Calculate dip vector
    double Vdip[3] = {
        Vnorm[1] * Vstrike[2] - Vnorm[2] * Vstrike[1],
        Vnorm[2] * Vstrike[0] - Vnorm[0] * Vstrike[2],
        Vnorm[0] * Vstrike[1] - Vnorm[1] * Vstrike[0]
    };

    // Define transformation matrix A
    double A[9] = {
        Vnorm[0], Vstrike[0], Vdip[0],
        Vnorm[1], Vstrike[1], Vdip[1],
        Vnorm[2], Vstrike[2], Vdip[2]
    };

    // Transform slip vector components from TDCS into EFCS
    double bX, bY, bZ;
    CoordTrans(bx, by, bz, A, &bX, &bY, &bZ);

    // Calculate contribution of angular dislocation pair on each TD side
    double u1, v1, w1, u2, v2, w2, u3, v3, w3;

    AngSetupFSC(X, Y, Z, bX, bY, bZ, P1, P2, nu, &u1, &v1, &w1); // Side P1P2
    AngSetupFSC(X, Y, Z, bX, bY, bZ, P2, P3, nu, &u2, &v2, &w2); // Side P2P3
    AngSetupFSC(X, Y, Z, bX, bY, bZ, P3, P1, nu, &u3, &v3, &w3); // Side P3P1

    // Calculate total harmonic function contribution to displacements
    *ue = u1 + u2 + u3;
    *un = v1 + v2 + v3;
    *uv = w1 + w2 + w3;
}

 /* ------------------------------------------------------------------------------------------------*/

