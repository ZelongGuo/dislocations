#include <stdio.h>
#include <stdlib.h>
#include "mehdi.h"

void mehdi_disloc3d(double *models, int nmodel, double *obss, int nobs, double mu, double nu,
                    double *U, double *S, double *E) {
    /*
     * Input Parameters:
     *
     * models:
     *        [nmodel * 12], a pointer of 1-D array, including the 3 coordinates
     * components of the 3 vertexes of the triangle, as well as the 3
     * dislocation components x1, y1, z1, x2, y2, z2, x3, y3, z3, str-slip,
     * dip-slip, opening
     *
     *        Note the vertexes are in the UTM coordinate system (right-hand
     * rule), the str-slip,dip-slip, and openings are based on the fault
     * coordiante system
     *
     * obss  : [nobs * 3], a pointer of 1-D array, in which the Z <= 0
     *
     * mu    : shear modulus
     *
     * nu    : Poisson's ratio
     *
     * Output:
     * U     :
     *        [nobs x 3], DISPLACEMENT, the unit is same to those defined by
     * dislocation slip in models
     *
     * S     : [nobs x 6], 6 independent
     * STRESS tensor components, the unit depends on that of shear modulus
     *
     * E :
     *        [nobs x 6], 6 independent STRAIN tensor components, dimensionless
     */

    // --------------------------------------------------------------------------

    double lamda;
    lamda = 2.0 * mu * nu / (1.0 - 2.0 * nu);

    double *model = NULL;
    double *obs = NULL;
    double *Uout = NULL, *Sout = NULL, *Eout = NULL;

    double P1[3] = {0};
    double P2[3] = {0};
    double P3[3] = {0};
    double ss, ds, ts;

    double sx, sy, sz;

    double ux, uy, uz;
    double uxt, uyt, uzt;

    double Et[6] = {0};
    double St[6] = {0};
    double Exxt, Exyt, Exzt, Eyyt, Eyzt, Ezzt;
    double Sxxt, Sxyt, Sxzt, Syyt, Syzt, Szzt;

    int i, j;

    for (i = 0; i < nobs; i++) {
        obs = obss + 3 * i;

        if (*(obs + 2) > 0.0) {
            // positive z value of the station in UTM is given, let flag1 = 1
            printf("\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
                   "-+-+-+-+-+-+-+-+-+-+-+-+-+-\n");
            fprintf(stderr,
                    "Error, Observation station (ID: %d) in UTM system has "
                    "positive depth %f, "
                    "output set to 0, also see flags!\n",
                    i, *(obs + 2));
            printf("\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
                   "-+-+-+-+-+-+-+-+-+-+-+-+-+-\n");
        }

        // Initialized
        uxt = uyt = uzt = 0;
        Exxt = Exyt = Exzt = Eyyt = Eyzt = Ezzt = 0;
        Sxxt = Sxyt = Sxzt = Syyt = Syzt = Szzt = 0;

        for (j = 0; j < nmodel; j++) {

            model = models + 12 * j;

            // The 3 vertexes of the triangle under UTM
            P1[0] = model[0], P1[1] = model[1], P1[2] = model[2];
            P2[0] = model[3], P2[1] = model[4], P2[2] = model[5];
            P3[0] = model[6], P3[1] = model[7], P3[2] = model[8];

            // The dislocation components under fault coordinate system
            ss = model[9];
            ds = model[10];
            ts = model[11];

            // Observation point in UTM
            sx = *obs;
            sy = *(obs + 1);
            sz = *(obs + 2);

            // Call for the function to calculate the dislocation components
            TDdispHS(sx, sy, sz, P1, P2, P3, ss, ds, ts, nu, &ux, &uy, &uz);
            TDstressHS(sx, sy, sz, P1, P2, P3, ss, ds, ts, mu, lamda, St, Et);

            // Add
            uxt += ux;
            uyt += uy;
            uzt += uz;

            // Add 6 independent strain components
            Exxt += Et[0]; // e11
            Exyt += Et[3]; // e12
            Exzt += Et[4]; // e13
            Eyyt += Et[1]; // e22
            Eyzt += Et[5]; // e23
            Ezzt += Et[2]; // e33

            // Add 6 independent stress components
            Sxxt += St[0]; // e11
            Sxyt += St[3]; // e12
            Sxzt += St[4]; // e13
            Syyt += St[1]; // e22
            Syzt += St[5]; // e23
            Szzt += St[2]; // e33
        }

        // Calculate U, S, D
        Uout = U + 3 * i;
        Uout[0] = uxt;
        Uout[1] = uyt;
        Uout[2] = uzt;

        // if you want to calculate Strains ...
        // symmetry with 6 independent elements
        // E11 E12 E13
        //     E22 E23
        //         E33
        Eout = E + 6 * i;
        Eout[0] = Exxt; // e11
        Eout[1] = Exyt; // e12
        Eout[2] = Exzt; // e13
        Eout[3] = Eyyt; // e22
        Eout[4] = Eyzt; // e23
        Eout[5] = Ezzt; // e33

        // Calculate Stress
        // S11 S12 S13
        //     S22 S23
        //         S33
        Sout = S + 6 * i;
        Sout[0] = Sxxt; // e11
        Sout[1] = Sxyt; // e12
        Sout[2] = Sxzt; // e13
        Sout[3] = Syyt; // e22
        Sout[4] = Syzt; // e23
        Sout[5] = Szzt; // e33
    }
}
