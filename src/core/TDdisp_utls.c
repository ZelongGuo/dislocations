#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mehdi.h"

/*---------------------------------------------------------
 *  Functions used for calculating disp in both full and 
 *  half space.
 *
 *  Referring to the Matlab codes of Mehdi
 *
 *  Author: Zelong Guo
 *  03.2025, @ Potsdam, Germany
 *  zelong.guo@outlook.com
 *
 *---------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*/

void AngDisDisp(double x, double y, double z, double alpha,
                double bx, double by, double bz, double nu,
                double *u, double *v, double *w) {
    /*
     * AngDisDisp calculates the "incomplete" displacements (without the Burgers' function 
     * contribution) associated with an angular dislocation in an elastic full-space.
     */


    // Pre-compute trigonometric functions
    double cosA = cos(alpha);
    double sinA = sin(alpha);

    // Intermediate variables
    double eta = y * cosA - z * sinA;
    double zeta = y * sinA + z * cosA;
    double r = sqrt(x * x + y * y + z * z);

    // Adjust zeta and z to avoid complex logarithmic results
    double zeta_adj = (zeta > r) ? r : zeta;
    double z_adj = (z > r) ? r : z;

    // Common factor
    double factor = 1.0 / (8.0 * M_PI * (1.0 - nu));

    // bx components
    double ux = bx * factor * (x * y / r / (r - z_adj) - x * eta / r / (r - zeta_adj));
    double vx = bx * factor * (eta * sinA / (r - zeta_adj) - y * eta / r / (r - zeta_adj) +
                              y * y / r / (r - z_adj) + (1.0 - 2.0 * nu) * 
                              (cosA * log(r - zeta_adj) - log(r - z_adj)));
    double wx = bx * factor * (eta * cosA / (r - zeta_adj) - y / r - 
                              eta * z / r / (r - zeta_adj) - 
                              (1.0 - 2.0 * nu) * sinA * log(r - zeta_adj));

    // by components
    double uy = by * factor * (x * x * cosA / r / (r - zeta_adj) - 
                              x * x / r / (r - z_adj) - 
                              (1.0 - 2.0 * nu) * (cosA * log(r - zeta_adj) - 
                              log(r - z_adj)));
    double vy = by * x * factor * (y * cosA / r / (r - zeta_adj) - 
                                  sinA * cosA / (r - zeta_adj) - 
                                  y / r / (r - z_adj));
    double wy = by * x * factor * (z * cosA / r / (r - zeta_adj) - 
                                  cosA * cosA / (r - zeta_adj) + 
                                  1.0 / r);

    // bz components
    double uz = bz * sinA * factor * ((1.0 - 2.0 * nu) * log(r - zeta_adj) - 
                                     x * x / r / (r - zeta_adj));
    double vz = bz * x * sinA * factor * (sinA / (r - zeta_adj) - 
                                         y / r / (r - zeta_adj));
    double wz = bz * x * sinA * factor * (cosA / (r - zeta_adj) - 
                                         z / r / (r - zeta_adj));

    // Final displacements
    *u = ux + uy + uz;
    *v = vx + vy + vz;
    *w = wx + wy + wz;
}

/*-----------------------------------------------------------------------------------------------*/
void TDSetupD(double x, double y, double z, double alpha,
              double bx, double by, double bz, double nu,
              double TriVertex[3], double SideVec[3],
              double *u, double *v, double *w) {
    /*
     * TDSetupD transforms coordinates of the calculation points as well as slip vector components
     * from ADCS into TDCS. It then calculates the displacements in ADCS and transforms them into TDCS.
     */

    // Transformation matrix A (2x2)
    double A[2][2] = {
        {SideVec[2], -SideVec[1]},
        {SideVec[1], SideVec[2]}
    };

    // Temporary variables
    double r1[2], r2[2], r3[2];
    double y1, z1, by1, bz1, v0, w0;

    // Transform coordinates from TDCS into ADCS
    // r1 = A * [y-TriVertex(2); z-TriVertex(3)]
    r1[0] = A[0][0] * (y - TriVertex[1]) + A[0][1] * (z - TriVertex[2]);
    r1[1] = A[1][0] * (y - TriVertex[1]) + A[1][1] * (z - TriVertex[2]);
    y1 = r1[0];
    z1 = r1[1];

    // Transform slip vector components from TDCS into ADCS
    // r2 = A * [by; bz]
    r2[0] = A[0][0] * by + A[0][1] * bz;
    r2[1] = A[1][0] * by + A[1][1] * bz;
    by1 = r2[0];
    bz1 = r2[1];

    // Calculate displacements in ADCS
    AngDisDisp(x, y1, z1, -M_PI + alpha, bx, by1, bz1, nu, u, &v0, &w0);

    // Transform displacements from ADCS into TDCS
    // r3 = A' * [v0; w0]
    // Note: A' (transpose) swaps rows and columns
    r3[0] = A[0][0] * v0 + A[1][0] * w0;
    r3[1] = A[0][1] * v0 + A[1][1] * w0;
    *v = r3[0];
    *w = r3[1];
}

