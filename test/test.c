#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// #include "dc3d.h"
// #include "dc3d.c"
#include "okada.h"
// #include "disloc3d.c"
// #include "disloc3d0.c"

int main() {
  // models: nmodel * 10
  // length, width, depth, dip, strike, easting, northing, str-slip, dip-selip.
  // opening
  //  * easting, northing, depth (>=0, defined as the depth of fault upper
  //  center point, easting and northing likewise) length, width, strike, dip,
  //  str-slip, dip-selip, opening

  double model[20] = {
      440.58095043254673, 3940.114839963042, 15, 80, 50, 50, 45, 0.01, 0.01, 0,
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
  double observations[15] = {454, 3943, 0,    454, 3943, 0,    454, 3943,
                             0,   454,  3943, 0,   454,  3943, 0};
  int nobs = 5;
  int flags[10] = {0};
  double mu = 3e10;
  double nu = 0.25;

  double U[15] = {0};
  double D[45] = {0};
  double S[45] = {0};
  double E[45] = {0};
  //--------------------------------------------------

  disloc3d(model, nmodel, observations, nobs, mu, nu, U, D, S, E, flags);

  printf("\n%s\n", STARS);
  for (int i = 0; i < nobs; i++) {
    printf("The ux is: %f, the uy is %f, the uz is %f.\n", *(U + i * 3),
           *(U + i * 3 + 1), *(U + i * 3 + 2));

    printf("The uxx is: %f, the uxy is %f, the uxz is %f.\n", *(D + i * 9),
           *(D + i * 9 + 1), *(D + i * 9 + 2));
    printf("The uyx is: %f, the uyy is %f, the uyz is %f.\n", *(D + i * 9 + 3),
           *(D + i * 9 + 4), *(D + i * 9 + 5));
    printf("The uzx is: %f, the uzy is %f, the uzz is %f.\n", *(D + i * 9 + 6),
           *(D + i * 9 + 7), *(D + i * 9 + 8));

    printf("The dxx is: %f, the dxy is %f, the dxz is %f.\n", *(S + i * 9),
           *(S + i * 9 + 1), *(S + i * 9 + 2));
    printf("The dyx is: %f, the dyy is %f, the dyz is %f.\n", *(S + i * 9 + 3),
           *(S + i * 9 + 4), *(S + i * 9 + 5));
    printf("The dzx is: %f, the dzy is %f, the dzz is %f.\n", *(S + i * 9 + 6),
           *(S + i * 9 + 7), *(S + i * 9 + 8));

    printf("The exx is: %f, the exy is %f, the exz is %f.\n", *(E + i * 9),
           *(E + i * 9 + 1), *(E + i * 9 + 2));
    printf("The eyx is: %f, the eyy is %f, the eyz is %f.\n", *(E + i * 9 + 3),
           *(E + i * 9 + 4), *(E + i * 9 + 5));
    printf("The ezx is: %f, the ezy is %f, the ezz is %f.\n", *(E + i * 9 + 6),
           *(E + i * 9 + 7), *(E + i * 9 + 8));

    printf("\n");
  }
  printf("\n%s\n", STARS);

  printf("flags = ");
  for (int i = 0; i < nobs * nmodel; i++) {
    printf(" %d ", *(flags + i));
  }
  printf("\n");

  // printf("\n%s\n", STARS);
  // printf("The ux is: %f, the uy is %f, the uz is %f.\n", U[0], U[1], U[2]);
  // printf("\n%s\n", STARS);
  // printf("The uxx is: %f, the uxy is %f, the uxz is %f.\n", D[0], D[1],
  // D[2]); printf("The uyx is: %f, the uyy is %f, the uyz is %f.\n", D[3],
  // D[4], D[5]); printf("The uzx is: %f, the uzy is %f, the uzz is %f.\n",
  // D[6], D[7], D[8]); printf("\n%s\n", STARS); printf("The d11 is: %f, the d12
  // is %f, the d13 is %f.\n", S[0], S[1], S[2]); printf("The d21 is: %f, the
  // d22 is %f, the d23 is %f.\n", S[3], S[4], S[5]); printf("The d31 is: %f,
  // the d32 is %f, the d33 is %f.\n", S[6], S[7], S[8]); printf("\n%s\n",
  // STARS); printf("The flag1 is: %d, the flag2 is %d.\n", flags[0], flags[1]);
  // printf("\n%s\n", STARS);

  return 0;

  // ---------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------
  //
  //    // test for point source model
  //    // models: nmodel * 6
  //    // depth, dip, pot1, pot2, pot3, pot4
  //                                //
  //    double model[6] = {1, 90, 1, 2, 3, 4};
  //    double observations[3] = {1,1,-1};
  //    int nobs = 1;
  //    int flags[1] = {0};
  //    double mu = 3e10;
  //    double nu = 0.25;
  //
  //    double U[3] = {0};
  //    double D[9] = {0};
  //    double S[6] = {0};
  //
  //    disloc3d0(model, observations, nobs, mu, nu,
  //	     U, D, S, flags);
  //
  //    printf("\n%s\n", STARS);
  //    printf("The ux is: %f, the uy is %f, the uz is %f.\n", U[0], U[1],
  //    U[2]); printf("\n%s\n", STARS); printf("The uxx is: %f, the uxy is %f,
  //    the uxz is %f.\n", D[0], D[1], D[2]); printf("The uyx is: %f, the uyy is
  //    %f, the uyz is %f.\n", D[3], D[4], D[5]); printf("The uzx is: %f, the
  //    uzy is %f, the uzz is %f.\n", D[6], D[7], D[8]); printf("\n%s\n",
  //    STARS); printf("The d11 is: %f, the d12 is %f, the d13 is %f.\n", S[0],
  //    S[1], S[2]); printf("The d22 is: %f, the d23 is %f, the d33 is %f.\n",
  //    S[3], S[4], S[5]); printf("\n%s\n", STARS);
  //
  //
  //    return 0;

  // ---------------------------------------------------------------------------------
  // ---------------------------------------------------------------------------------

  //    double alpha = 0.6, x = 1.0, y = 1.0, z = -1.0;
  //    double depth = 1.0, dip = 45.0, al1 = -1.7, al2 = 1.7;
  //    double aw1 = -1.7, aw2 = 1.7, dis1 = 5.0, dis2 = 5.0, dis3 = 0.0;
  //    double ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz;
  //    int iret;
  //    dc3d_(&alpha, &x, &y, &z, &depth, &dip,
  //        		&al1, &al2, &aw1, &aw2,
  //        		&dis1, &dis2, &dis3,
  //        		&ux, &uy, &uz,
  //        		&uxx, &uyx, &uzx,
  //        		&uxy, &uyy, &uzy,
  //        		&uxz, &uyz, &uzz,
  //        		&iret);
  //
  //    printf("The ux is: %f, the uy is %f, the uz is %f.\n", ux, uy, uz);
  //    printf("The uxx is: %f, the uyx is %f, the uzx is %f.\n", uxx, uyx,
  //    uzx); printf("The uxy is: %f, the uyy is %f, the uzy is %f.\n", uxy,
  //    uyy, uzy); printf("The uxz is: %f, the uyz is %f, the uzz is %f.\n",
  //    uxz, uyz, uzz);
  //
  //    return 0;
}
