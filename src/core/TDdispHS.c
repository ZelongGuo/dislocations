#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mehdi.h"

/*---------------------------------------------------------
 *  Main function used for calculating disp in half space.
 *
 *  Referring to the Matlab codes of Mehdi
 *
 *  Author: Zelong Guo
 *  03.2025, @ Potsdam, Germany
 *  zelong.guo@outlook.com
 *
 * ---------------------------------------------------------*/

// TDdispHS calculates displacements associated with a triangular dislocation in an elastic half-space.
void TDdispHS(double X, double Y, double Z, double P1[3], double P2[3], double P3[3],
              double Ss, double Ds, double Ts, double nu, double *ue, double *un, double *uv) {

    // Ensure that all Z coordinates are negative (Half-space assumption)
    if (Z > 0 || P1[2] > 0 || P2[2] > 0 || P3[2] > 0) {
        fprintf(stderr, "Error: Half-space solution requires Z coordinates to be negative!\n");
        exit(EXIT_FAILURE);
    }

    // Calculate unit normal vector to the fault plane
    double Vnorm[3];
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
            double temp = P2[i];
            P2[i] = P3[i];
            P3[i] = temp;
        }
    }

    // Calculate main dislocation contribution to displacements
    double ueMS, unMS, uvMS;
    TDdispFS4HS(X, Y, Z, P1, P2, P3, Ss, Ds, Ts, nu, &ueMS, &unMS, &uvMS);

    // Calculate harmonic function contribution to displacements
    double ueFSC, unFSC, uvFSC;
    TDdisp_HarFunc(X, Y, Z, P1, P2, P3, Ss, Ds, Ts, nu, &ueFSC, &unFSC, &uvFSC);

    // Calculate image dislocation contribution to displacements
    double P1_img[3] = { P1[0], P1[1], -P1[2] };
    double P2_img[3] = { P2[0], P2[1], -P2[2] };
    double P3_img[3] = { P3[0], P3[1], -P3[2] };

    double ueIS, unIS, uvIS;
    TDdispFS4HS(X, Y, Z, P1_img, P2_img, P3_img, Ss, Ds, Ts, nu, &ueIS, &unIS, &uvIS);

    if (P1[2] == 0 && P2[2] == 0 && P3[2] == 0) {
        uvIS = -uvIS;
    }

    // Calculate the complete displacement vector components in EFCS
    *ue = ueMS + ueIS + ueFSC;
    *un = unMS + unIS + unFSC;
    *uv = uvMS + uvIS + uvFSC;

    if (P1[2] == 0 && P2[2] == 0 && P3[2] == 0) {
        *ue = -(*ue);
        *un = -(*un);
        *uv = -(*uv);
    }
}

