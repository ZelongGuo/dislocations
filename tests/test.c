#include <stdio.h>
#include <stdlib.h>
#include "meade.h"
#include "mehdi.h"
#include "okada.h"

// -----------------------------------------------------------------------------
// This is a test file for the Okada wrapper function
// -----------------------------------------------------------------------------

int main() {
    // models: nmodel * 10
    // length, width, depth, dip, strike, easting, northing, str-slip,
    // dip-selip. opening
    //  * easting, northing, depth (>=0, defined as the depth of fault upper
    //  center point, easting and northing likewise) length, width, strike, dip,
    //  str-slip, dip-selip, opening

    double model[20] = {440.58095043254673, 3940.114839963042, 15, 80, 50, 50, 45, 0.01, 0.01, 0,
                        440.58095043254673, 3940.114839963042, 15, 80, 50, 50, 45, 0.01, 0.01, 0};

    int nmodel = 2;

    //--------------------------------------------------
    // double observations[6] = {454, 3943, 0,
    //     		      454, 3943, 0,
    //     		       };
    // int nobs = 2;
    // int flags[4] = {0};
    // double mu = 3e10;
    // double nu = 0.25;

    // double U[6]={0};
    // double D[18]={0};
    // double S[18]={0};
    //--------------------------------------------------
    double observations[9] = {454, 3943, 0, 454, 3943, 0, 454, 3943, 0};
    int nobs = 3;
    int flags[6] = {0};
    double mu = 3e10;
    double nu = 0.25;

    double U[9] = {0};
    double S[18] = {0};
    double E[18] = {0};
    //--------------------------------------------------

    okada_disloc3d(model, nmodel, observations, nobs, mu, nu, U, S, E, flags);

    printf("\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
           "-+-+-+-+-+-+-+-+-\n");
    for (int i = 0; i < nobs; i++) {
        printf("The ux is: %f, the uy is %f, the uz is %f.\n", *(U + i * 3), *(U + i * 3 + 1),
               *(U + i * 3 + 2));

        printf("The dxx is: %f, the dxy is %f, the dxz is %f.\n", *(S + i * 6), *(S + i * 6 + 1),
               *(S + i * 6 + 2));
        printf("The dyy is %f, the dyz is %f.\n", *(S + i * 6 + 3), *(S + i * 6 + 4));
        printf("The dzz is %f.\n", *(S + i * 6 + 5));

        printf("The exx is: %f, the exy is %f, the exz is %f.\n", *(E + i * 6), *(E + i * 6 + 1),
               *(E + i * 6 + 2));
        printf("The eyy is %f, the eyz is %f.\n", *(E + i * 6 + 3), *(E + i * 6 + 4));
        printf("The ezz is %f.\n", *(E + i * 6 + 5));

        printf("\n");
    }
    printf("\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
           "-+-+-+-+-+-+-+-+-\n");

    printf("flags = ");
    for (int i = 0; i < nobs * nmodel; i++) {
        printf(" %d ", *(flags + i));
    }
    printf("\n");

    // ---------------------------------------------------------------------------------
    // ---------------------------------------------------------------------------------
    // Meade triangle test
    double mu2 = 3.3e10;
    double nu2 = 0.25;

    // double model2[12] = {-1, -1, -4, 1, -1, -3, -1, -1, -2, 1, -1, 2};
    double model2[12] = {-1, -1, -4, 1, -1, -3, -1, -1, -2, 0, -1, 0};
    // double model3[12] = {-1.0, -1.0, -5.0, 1.0, -1.0, -5.0, -1.0, 1.0, -5.0, 1.0, -4.0, 2.0};
    int nmodel2 = 1;
    // double model2[24] = {-1, -1, -5, 1, -1, -5, -1, 1, -4, 1, -1, 2,
    //                      -1, -1, -5, 1, -1, -5, -1, 1, 0, 1, -1, 2
    // };
    // int nmodel2 = 2;

    double observations2[12] = {
        // // sx,      sy,     sz,      sx,      sy,         sz,      sx,       sy,      sz,
        // -1 / 3.0, -1 / 3.0, -3,   -1 / 3.0, -1 / 3.0, -14.0 / 3.0, -1 / 3.0, -1 / 3.0, -6,
        // 7.0,      -1.0,     -5.0, -7.0,     -1.0,     -5.0,        -1.0,     -3.0,     -6.0,
        // -1.0,     3.0,      -3.0, 3.0,      -3.0,     -6.0,        -3.0,     3.0,      -3.0,
        // -1.0,     -1.0,     -1.0, -1.0,     1.0,      -1.0,        1.0,      -1.0,     -1.0,
        // -1.0,     -1.0,     -8.0, -1.0,     1.0,      -8.0,        1.0,      -1.0,     -8.0,
        -1/3.0, -1/3.0, 0.0, -1/3.0, -1/3.0, -3.0, -1/3.0, -1/3.0, -5.0, -1/3.0, -1/3.0, -8.0
    };
    int nobs2 = 4;
    // double U2[6] = {0};
    // double S2[12] = {0};
    // double E2[12] = {0};

    double *U2 = calloc(nobs2 * 3, sizeof(double));
    double *S2 = calloc(nobs2 * 6, sizeof(double));
    double *E2 = calloc(nobs2 * 6, sizeof(double));
    double *U3 = calloc(nobs2 * 3, sizeof(double));
    double *S3 = calloc(nobs2 * 6, sizeof(double));
    double *E3 = calloc(nobs2 * 6, sizeof(double));

    if (!U2 || !S2 || !E2 || !U3 || !S3 || !E3) {
        printf("Memory allocation failed\n");
        return 1;
    }

    meade_disloc3d(model2, nmodel2, observations2, nobs2, mu2, nu2, U2, S2, E2);
    mehdi_disloc3d(model2, nmodel2, observations2, nobs2, mu2, nu2, U3, S3, E3);

    // for (int i = 0; i < nobs2; i++) {
    //     printf("\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+ Triangle "
    //            "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"
    //            "-+-+-+-+-+-+-+-+-\n");
    //     printf("The ux is: %f, the uy is %f, the uz is %f.\n", *(U2 + i * 3),
    //            *(U2 + i * 3 + 1), *(U2 + i * 3 + 2));

    //     printf("The exx is: %f, The eyy is: %f, The ezz is: %f, the exy is
    //     %f, "
    //            "the exz is %f, the eyz is %f.\n",
    //            *(E2 + i * 6), *(E2 + i * 6 + 3), *(E2 + i * 6 + 5),
    //            *(E2 + i * 6 + 1), *(E2 + i * 6 + 2), *(E2 + i * 6 + 4));

    //     printf("The sxx is: %f, The syy is: %f, The szz is: %f, the sxy is
    //     %f, "
    //            "the sxz is %f, the syz is %f.\n",
    //            *(S2 + i * 6), *(S2 + i * 6 + 3), *(S2 + i * 6 + 5),
    //            *(S2 + i * 6 + 1), *(S2 + i * 6 + 2), *(S2 + i * 6 + 4));

    // }

    printf("\n+-+-+-+-+-+-+-+-+-+-++-+-+ Meade Triangle -+-+-+--+-+-+-+-+-+-+-+-+-+-+-+\n");

    printf("Ue = \n");
    for (int i = 0; i < nobs2; i++) {
        printf("%.8f\n", *(U2 + i * 3));
    }

    printf("------------------------------------\n");

    printf("Un = \n");
    for (int i = 0; i < nobs2; i++) {
        printf("%.8f\n", *(U2 + i * 3 + 1));
    }

    printf("------------------------------------\n");
    printf("Uu = \n");
    for (int i = 0; i < nobs2; i++) {
        printf("%.8f\n", *(U2 + i * 3 + 2));
    }

    printf("------------------------------------\n");
    printf("E / strian = \n");
    for (int i = 0; i < nobs2; i++) {
        printf("%.8f %.8f  %.8f  %.8f  %.8f  %.8f\n", *(E2 + i * 6), *(E2 + i * 6 + 3),
               *(E2 + i * 6 + 5), *(E2 + i * 6 + 1), *(E2 + i * 6 + 2), *(E2 + i * 6 + 4));
    }

    printf("------------------------------------\n");
    printf("S / Stress = \n");
    for (int i = 0; i < nobs2; i++) {
        printf("%.8e %.8e %.8e %.8e %.8e %.8e \n", *(S2 + i * 6), *(S2 + i * 6 + 3),
               *(S2 + i * 6 + 5), *(S2 + i * 6 + 1), *(S2 + i * 6 + 2), *(S2 + i * 6 + 4));
    }

    // -------------------------------------------------------------------------------- 

    printf("\n+-+-+-+-+-+-+-+-+-+-++-+-+ Mehdi Triangle -+-+-+--+-+-+-+-+-+-+-+-+-+-+-+\n");
    printf("Ue = \n");
    for (int i = 0; i < nobs2; i++) {
        printf("%.8f\n", *(U3 + i * 3));
    }

    printf("------------------------------------\n");

    printf("Un = \n");
    for (int i = 0; i < nobs2; i++) {
        printf("%.8f\n", *(U3 + i * 3 + 1));
    }

    printf("------------------------------------\n");
    printf("Uu = \n");
    for (int i = 0; i < nobs2; i++) {
        printf("%.8f\n", *(U3 + i * 3 + 2));
    }

    printf("------------------------------------\n");
    printf("E / strian = \n");
    for (int i = 0; i < nobs2; i++) {
        printf("%15.8f %15.8f  %15.8f  %15.8f  %15.8f  %15.8f\n", *(E3 + i * 6), *(E3 + i * 6 + 3),
               *(E3 + i * 6 + 5), *(E3 + i * 6 + 1), *(E3 + i * 6 + 2), *(E3 + i * 6 + 4));
    }

    printf("------------------------------------\n");
    printf("S / Stress = \n");
    for (int i = 0; i < nobs2; i++) {
        printf("%15.8e %15.8e %15.8e %15.8e %15.8e %15.8e \n", *(S3 + i * 6), *(S3 + i * 6 + 3),
               *(S3 + i * 6 + 5), *(S3 + i * 6 + 1), *(S3 + i * 6 + 2), *(S3 + i * 6 + 4));
    }

    free(U2);
    free(S2);
    free(E2);
    free(U3);
    free(S3);
    free(E3);

    return 0;
}
