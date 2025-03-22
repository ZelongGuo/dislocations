#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mehdi.h"

/*---------------------------------------------------------
 *  Functions used for calculating stress in both full and
 *  half space.
 *
 *  Referring to the Matlab codes of Mehdi
 *
 *  Author: Zelong Guo
 *  03.2025, @ Potsdam, Germany
 *  zelong.guo@outlook.com
 *
 * ---------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/
void AngDisStrain(double x, double y, double z, double alpha, double bx, double by, double bz, double nu,
                  double *Exx, double *Eyy, double *Ezz, double *Exy, double *Exz, double *Eyz) {
    // Precompute trigonometric functions
    double sinA = sin(alpha);
    double cosA = cos(alpha);

    // Intermediate variables
    double eta = y * cosA - z * sinA;
    double zeta = y * sinA + z * cosA;

    double x2 = x * x;
    double y2 = y * y;
    double z2 = z * z;
    double r2 = x2 + y2 + z2;
    double r = sqrt(r2);
    double r3 = r * r2;
    double rz = r * (r - z);
    double r2z2 = r2 * (r - z) * (r - z);
    double r3z = r3 * (r - z);

    double W = zeta - r;
    double W2 = W * W;
    double Wr = W * r;
    double W2r = W2 * r;
    double Wr3 = W * r3;
    double W2r2 = W2 * r2;

    double C = (r * cosA - z) / Wr;
    double S = (r * sinA - y) / Wr;

    // Partial derivatives of the Burgers' function
    double rFi_rx = (eta / r / (r - zeta) - y / r / (r - z)) / (4.0 * M_PI);
    double rFi_ry = (x / r / (r - z) - cosA * x / r / (r - zeta)) / (4.0 * M_PI);
    double rFi_rz = (sinA * x / r / (r - zeta)) / (4.0 * M_PI);

    // Strain components
    *Exx = bx * rFi_rx +
           bx / (8.0 * M_PI * (1.0 - nu)) * (eta / Wr + eta * x2 / W2r2 - eta * x2 / Wr3 + y / rz -
                                             x2 * y / r2z2 - x2 * y / r3z) -
           by * x / (8.0 * M_PI * (1.0 - nu)) * (((2.0 * nu + 1.0) / Wr + x2 / W2r2 - x2 / Wr3) * cosA +
                                                 (2.0 * nu + 1.0) / rz - x2 / r2z2 - x2 / r3z) +
           bz * x * sinA / (8.0 * M_PI * (1.0 - nu)) * ((2.0 * nu + 1.0) / Wr + x2 / W2r2 - x2 / Wr3);

    *Eyy = by * rFi_ry +
           bx / (8.0 * M_PI * (1.0 - nu)) * ((1.0 / Wr + S * S - y2 / Wr3) * eta + (2.0 * nu + 1.0) * y / rz -
                                             y * y * y / r2z2 - y * y * y / r3z - 2.0 * nu * cosA * S) -
           by * x / (8.0 * M_PI * (1.0 - nu)) * (1.0 / rz - y2 / r2z2 - y2 / r3z +
                                                 (1.0 / Wr + S * S - y2 / Wr3) * cosA) +
           bz * x * sinA / (8.0 * M_PI * (1.0 - nu)) * (1.0 / Wr + S * S - y2 / Wr3);

    *Ezz = bz * rFi_rz +
           bx / (8.0 * M_PI * (1.0 - nu)) * (eta / W / r + eta * C * C - eta * z2 / Wr3 + y * z / r3 +
                                             2.0 * nu * sinA * C) -
           by * x / (8.0 * M_PI * (1.0 - nu)) * ((1.0 / Wr + C * C - z2 / Wr3) * cosA + z / r3) +
           bz * x * sinA / (8.0 * M_PI * (1.0 - nu)) * (1.0 / Wr + C * C - z2 / Wr3);

    *Exy = bx * rFi_ry / 2.0 + by * rFi_rx / 2.0 -
           bx / (8.0 * M_PI * (1.0 - nu)) * (x * y2 / r2z2 - nu * x / rz + x * y2 / r3z - nu * x * cosA / Wr +
                                             eta * x * S / Wr + eta * x * y / Wr3) +
           by / (8.0 * M_PI * (1.0 - nu)) * (x2 * y / r2z2 - nu * y / rz + x2 * y / r3z + nu * cosA * S +
                                             x2 * y * cosA / Wr3 + x2 * cosA * S / Wr) -
           bz * sinA / (8.0 * M_PI * (1.0 - nu)) * (nu * S + x2 * S / Wr + x2 * y / Wr3);

    *Exz = bx * rFi_rz / 2.0 + bz * rFi_rx / 2.0 -
           bx / (8.0 * M_PI * (1.0 - nu)) * (-x * y / r3 + nu * x * sinA / Wr + eta * x * C / Wr +
                                             eta * x * z / Wr3) +
           by / (8.0 * M_PI * (1.0 - nu)) * (-x2 / r3 + nu / r + nu * cosA * C + x2 * z * cosA / Wr3 +
                                             x2 * cosA * C / Wr) -
           bz * sinA / (8.0 * M_PI * (1.0 - nu)) * (nu * C + x2 * C / Wr + x2 * z / Wr3);

    *Eyz = by * rFi_rz / 2.0 + bz * rFi_ry / 2.0 +
           bx / (8.0 * M_PI * (1.0 - nu)) * (y2 / r3 - nu / r - nu * cosA * C + nu * sinA * S +
                                             eta * sinA * cosA / W2 - eta * (y * cosA + z * sinA) / W2r +
                                             eta * y * z / W2r2 - eta * y * z / Wr3) -
           by * x / (8.0 * M_PI * (1.0 - nu)) * (y / r3 + sinA * cosA * cosA / W2 -
                                                 cosA * (y * cosA + z * sinA) / W2r +
                                                 y * z * cosA / W2r2 - y * z * cosA / Wr3) -
           bz * x * sinA / (8.0 * M_PI * (1.0 - nu)) * (y * z / Wr3 - sinA * cosA / W2 +
                                                        (y * cosA + z * sinA) / W2r - y * z / W2r2);
}

/*-----------------------------------------------------------------------------------------------*/

void TensTrans(double Txx1, double Tyy1, double Tzz1, double Txy1, double Txz1, double Tyz1,
               double A[9],  // 3x3 transformation matrix stored as 1D array (row-major order)
               double *Txx2, double *Tyy2, double *Tzz2, double *Txy2, double *Txz2, double *Tyz2) {
    // Transform Txx1, Tyy1, Tzz1, Txy1, Txz1, Tyz1 from x1y1z1 to x2y2z2 using matrix A
    // A[0] to A[8] correspond to MATLAB's A(1) to A(9) in row-major order:
    // A[0] = A(1,1), A[1] = A(1,2), A[2] = A(1,3)
    // A[3] = A(2,1), A[4] = A(2,2), A[5] = A(2,3)
    // A[6] = A(3,1), A[7] = A(3,2), A[8] = A(3,3)

    /*
     * Note here in MATLAB the array is stored with COLUMN firstly, while in C it's line firstly
     * Thus here we need the transpose of the input matrix A
     *
     * */

    double At[9] = {
        A[0], A[3], A[6],
        A[1], A[4], A[7],
        A[2], A[5], A[8]
    };

    // Calculate Txx2
    *Txx2 = At[0] * At[0] * Txx1 +  // A(1)^2 * Txx1  -- Here A is the matrix corresponding in Matlab
            2.0 * At[0] * At[3] * Txy1 +  // 2 * A(1) * A(4) * Txy1
            2.0 * At[0] * At[6] * Txz1 +  // 2 * A(1) * A(7) * Txz1
            2.0 * At[3] * At[6] * Tyz1 +  // 2 * A(4) * A(7) * Tyz1
            At[3] * At[3] * Tyy1 +  // A(4)^2 * Tyy1
            At[6] * At[6] * Tzz1;   // A(7)^2 * Tzz1

    // Calculate Tyy2
    *Tyy2 = At[1] * At[1] * Txx1 +  // A(2)^2 * Txx1
            2.0 * At[1] * At[4] * Txy1 +  // 2 * A(2) * A(5) * Txy1
            2.0 * At[1] * At[7] * Txz1 +  // 2 * A(2) * A(8) * Txz1
            2.0 * At[4] * At[7] * Tyz1 +  // 2 * A(5) * A(8) * Tyz1
            At[4] * At[4] * Tyy1 +  // A(5)^2 * Tyy1
            At[7] * At[7] * Tzz1;   // A(8)^2 * Tzz1

    // Calculate Tzz2
    *Tzz2 = At[2] * At[2] * Txx1 +  // A(3)^2 * Txx1
            2.0 * At[2] * At[5] * Txy1 +  // 2 * A(3) * A(6) * Txy1
            2.0 * At[2] * At[8] * Txz1 +  // 2 * A(3) * A(9) * Txz1
            2.0 * At[5] * At[8] * Tyz1 +  // 2 * A(6) * A(9) * Tyz1
            At[5] * At[5] * Tyy1 +  // A(6)^2 * Tyy1
            At[8] * At[8] * Tzz1;   // A(9)^2 * Tzz1

    // Calculate Txy2
    *Txy2 = At[0] * At[1] * Txx1 +  // A(1) * A(2) * Txx1
            (At[0] * At[4] + At[1] * At[3]) * Txy1 +  // (A(1) * A(5) + A(2) * A(4)) * Txy1
            (At[0] * At[7] + At[1] * At[6]) * Txz1 +  // (A(1) * A(8) + A(2) * A(7)) * Txz1
            (At[7] * At[3] + At[6] * At[4]) * Tyz1 +  // (A(8) * A(4) + A(7) * A(5)) * Tyz1
            At[4] * At[3] * Tyy1 +  // A(5) * A(4) * Tyy1
            At[6] * At[7] * Tzz1;   // A(7) * A(8) * Tzz1

    // Calculate Txz2
    *Txz2 = At[0] * At[2] * Txx1 +  // A(1) * A(3) * Txx1
            (At[0] * At[5] + At[2] * At[3]) * Txy1 +  // (A(1) * A(6) + A(3) * A(4)) * Txy1
            (At[0] * At[8] + At[2] * At[6]) * Txz1 +  // (A(1) * A(9) + A(3) * A(7)) * Txz1
            (At[8] * At[3] + At[6] * At[5]) * Tyz1 +  // (A(9) * A(4) + A(7) * A(6)) * Tyz1
            At[5] * At[3] * Tyy1 +  // A(6) * A(4) * Tyy1
            At[6] * At[8] * Tzz1;   // A(7) * A(9) * Tzz1

    // Calculate Tyz2
    *Tyz2 = At[1] * At[2] * Txx1 +  // A(2) * A(3) * Txx1
            (At[2] * At[4] + At[1] * At[5]) * Txy1 +  // (A(3) * A(5) + A(2) * A(6)) * Txy1
            (At[2] * At[7] + At[1] * At[8]) * Txz1 +  // (A(3) * A(8) + A(2) * A(9)) * Txz1
            (At[7] * At[5] + At[8] * At[4]) * Tyz1 +  // (A(8) * A(6) + A(9) * A(5)) * Tyz1
            At[4] * At[5] * Tyy1 +  // A(5) * A(6) * Tyy1
            At[7] * At[8] * Tzz1;   // A(8) * A(9) * Tzz1
}

/* ---------------------------------------------------------------------------------------------- */

void TDSetupS(double x, double y, double z, double alpha, double bx, double by, double bz, double nu,
              double TriVertex[3], double SideVec[3],
              double *exx, double *eyy, double *ezz, double *exy, double *exz, double *eyz) {
    // Transformation matrix A (2x2 matrix stored as 1D array of 4 elements in row-major order)
    double A[4] = {
        SideVec[2], -SideVec[1],  // Row 1: [SideVec(3), -SideVec(2)]
        SideVec[1], SideVec[2]    // Row 2: [SideVec(2), SideVec(3)]
    };

    // Transform coordinates from TDCS to ADCS
    double r1[2];
    r1[0] = A[0] * (y - TriVertex[1]) + A[1] * (z - TriVertex[2]);  // y1 = A(1,1)*(y-TriVertex(2)) + A(1,2)*(z-TriVertex(3))
    r1[1] = A[2] * (y - TriVertex[1]) + A[3] * (z - TriVertex[2]);  // z1 = A(2,1)*(y-TriVertex(2)) + A(2,2)*(z-TriVertex(3))
    double y1 = r1[0];
    double z1 = r1[1];

    // Transform slip vector components from TDCS to ADCS
    double r2[2];
    r2[0] = A[0] * by + A[1] * bz;  // by1 = A(1,1)*by + A(1,2)*bz
    r2[1] = A[2] * by + A[3] * bz;  // bz1 = A(2,1)*by + A(2,2)*bz
    double by1 = r2[0];
    double bz1 = r2[1];

    // Calculate strains in ADCS using AngDisStrain
    double exx_adcs, eyy_adcs, ezz_adcs, exy_adcs, exz_adcs, eyz_adcs;
    AngDisStrain(x, y1, z1, -M_PI + alpha, bx, by1, bz1, nu,
                 &exx_adcs, &eyy_adcs, &ezz_adcs, &exy_adcs, &exz_adcs, &eyz_adcs);

    // Transformation matrix B (3x3 matrix stored as 1D array of 9 elements in row-major order)
    double B[9] = {
        1.0, 0.0, 0.0,        // Row 1: [1, 0, 0]
        0.0, A[0], A[2],      // Row 2: [0, A(1,1), A(2,1)]
        0.0, A[1], A[3]       // Row 3: [0, A(1,2), A(2,2)]
    };

    // Transform strains from ADCS to TDCS using TensTrans
    TensTrans(exx_adcs, eyy_adcs, ezz_adcs, exy_adcs, exz_adcs, eyz_adcs, B,
              exx, eyy, ezz, exy, exz, eyz);
}

/* ---------------------------------------------------------------------------------------------- */


