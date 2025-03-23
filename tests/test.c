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

    double model2[12] = {-1, -1, -4, 1, -1, -3, -1, -1, -2, 1, -1, 2};
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

    /*
    // ---------------------------------------------------------------------------------
    // Mehdi triangle test
    printf("\n+-+-+-+-+-+-+-+-+-+-+-+-+-+ Mehdi Triangle -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n");

    double mu3 = 3.3e10;
    double nu3 = 0.25;

    double model3[12] = {-1.0, -1.0, -5.0, 1.0, -1.0, -5.0, -1.0, 1.0, -5.0, 1.0, -4.0, 2.0};
    int nmodel3 = 1;

    double observations3[3] = {
        -1.0,
        -1.0,
         // 0.0,
        -8.0,
        // -4
    };

    int nobs3 = 1;
    double ue, un, uv;
    double ss, ds, ts;

    // -----------------------------------------------------------------

    double P1[3] = {model3[0], model3[1], model3[2]};
    double P2[3] = {model3[3], model3[4], model3[5]};
    double P3[3] = {model3[6], model3[7], model3[8]};
    ss = model3[9];
    ds = model3[10];
    ts = model3[11];

    // TDdispFS(observations3[0], observations3[1], observations3[2], P1, P2, P3, ss, ds, ts, nu3, &ue, &un, &uv);
    // TDdispHS(observations3[0], observations3[1], observations3[2], P1, P2, P3, ss, ds, ts, nu3, &ue, &un, &uv);

    // printf("x = %f, y = %f, z = %f, ss = %f, ds = %f, ts = %f, nu2 = %f\n", observations2[0],
    // observations2[1], observations2[2],ss, ds, ts, nu2); printf("P1: %f,  %f, %f\n", P1[0],
    // P1[1], P1[2]); printf("P2: %f,  %f, %f\n", P2[0], P2[1], P2[2]); printf("P3: %f,  %f, %f\n",
    // P3[0], P3[1], P3[2]);
    printf("ue = %.9f, un = %.9f, uv = %.9f\n", ue, un, uv);
    printf("ue = %f, un = %f, uv = %f\n", ue, un, uv);

    double lambda = 3.3e10;
    double Stress[6], Strain[6];

    // TDstressFS(observations3[0], observations3[1], observations3[2], P1, P2, P3, ss, ds, ts, mu3, lambda, Stress, Strain);
    // TDstressHS(observations3[0], observations3[1], observations3[2], P1, P2, P3, ss, ds, ts, mu3, lambda, Stress, Strain);
    printf("sxx = %f, syy = %f, szz = %f\n", Stress[0], Stress[1], Stress[2]);
    printf("sxy = %f, sxz = %f, syz = %f\n", Stress[3], Stress[4], Stress[5]);

    printf("\n");
    printf("exx = %f, eyy = %f, ezz = %f\n", Strain[0], Strain[1], Strain[2]);
    printf("exy = %f, exz = %f, eyz = %f\n", Strain[3], Strain[4], Strain[5]);

    // double y1 = 4.0, y2 = 999999999.0, y3 = 1.0, beta = 1.0, b1 = 1.0, b2 = 1.0, b3 = 1.0, a = 1.0;
    // double v11, v22, v33, v12, v13, v23;
    // AngDisStrainFSC(y1, y2, y3, beta, b1, b2, b3, nu, a, &v11, &v22, &v33, &v12, &v13, &v23);
    // printf("AngDisStrainFSC: v11: %12.5e, v22: %12.5e, v33: %12.5e, v12 %12.5e, v13: %12.5e, v23: %12.5e\n", v11, v22,
    //        v33, v12, v13, v23);
    */
    return 0;
}
