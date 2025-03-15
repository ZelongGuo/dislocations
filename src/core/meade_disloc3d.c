#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "meade.h"

#define DEG2RAD (M_PI / 180)
#define cosd(a) (cos((a) * DEG2RAD))
#define sind(a) (sin((a) * DEG2RAD))

void meade_disloc3d(double *models, int nmodel, double *obss, int nobs,
                    double mu, double nu, double *U, double *S, double *E) {
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
     * flags :
     *        [nobs * nmodle], a pointer of an 1-D array 0 normal; 1 the Z value
     * of the obs > 0 10 the depth of the fault upper center point reached to
     * surface (depth < 0) 100 singular point (observation point lies on the
     * fault edges); and more combination scenarios.
     */

    // --------------------------------------------------------------------------

    double lamda;
    lamda = 2.0 * mu * nu / (1.0 - 2.0 * nu);

    double *model = NULL;
    double *obs = NULL;
    double *Uout = NULL, *Sout = NULL, *Eout = NULL;

    double x1, y1, z1;
    double x2, y2, z2;
    double x3, y3, z3;
    double ss, ds, ts;

    double sx, sy, sz;

    double uxt, uyt, uzt;
    double Exxt, Exyt, Exzt, Eyyt, Eyzt, Ezzt;

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

        for (j = 0; j < nmodel; j++) {

            model = models + 12 * j;

            // From UTM to Angular x1x2x3 coordinate system
            // First vertex
            x1 = model[0];
            y1 = -model[1];
            z1 = -model[2];
            // Second vertex
            x2 = model[3];
            y2 = -model[4];
            z2 = -model[5];
            // Third vertex
            x3 = model[6];
            y3 = -model[7];
            z3 = -model[8];

            // Same to the dislocation components
            ss = model[9];
            ds = -model[10];
            ts = -model[11];

            // Observation point to x1x2x3
            sx = *obs;
            sy = -*(obs + 1);
            sz = -*(obs + 2);

            double x[3] = {x1, x2, x3};
            double y[3] = {y1, y2, y3};
            double z[3] = {z1, z2, z3};
            double Ut[3] = {0};
            double Et[6] = {0};

            // Call for the function to calculate the dislocation components
            CalTriDisps(sx, sy, sz, x, y, z, nu, ss, ts, ds, Ut);
            CalTriStrains(sx, sy, sz, x, y, z, nu, ss, ts, ds, Et);

            // Add under the x1x2x3 coordinate system
            uxt += Ut[0];
            uyt += Ut[1];
            uzt += Ut[2];

            // Add 6 independent strain components under the x1x2x3 coordinate
            // system
            Exxt += Et[0]; // e11
            Exyt += Et[1]; // e12
            Exzt += Et[2]; // e13
            Eyyt += Et[3]; // e22
            Eyzt += Et[4]; // e23
            Ezzt += Et[5]; // e33
        }

        // Transform the dislocation components to UTM coordinate system
        uxt = uxt;
        uyt = -uyt;
        uzt = -uzt;

        // Transform the strain components to UTM coordinate system
        Exxt = Exxt;
        Eyyt = Eyyt;
        Ezzt = Ezzt;
        Exyt = -Exyt;
        Exzt = -Exzt;
        Eyzt = Eyzt;

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
        Sout = S + 6 * i;
        CalTriStress(Eout, lamda, mu, Sout);
    }
}
