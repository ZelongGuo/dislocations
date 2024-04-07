#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "okada_dc3d.h"
#include "okada_disloc3d.h"

#define DEG2RAD (M_PI / 180)
#define cosd(a) (cos((a)*DEG2RAD))
#define sind(a) (sin((a)*DEG2RAD))

void disloc3d(double *models, int nmodel, double *obss, int nobs, double mu, double nu, double *U, double *D, double *S, int *flags)
{
    /*	
     * Input Parameters: 
     *
     * models: [nmodel * 10], a pointer of 1-D array 
     *         easting, northing, depth (>=0, defined as the depth of fault upper center point, easting and northing likewise)
     *         length, width, strike, dip, str-slip, dip-selip, opening
     * obs   : [nobs * 3], a pointer of 1-D array, in which the Z <= 0
     * mu    : shear modulus
     * nu    : Poisson's ratio
     *
     * Output: 
     * U     : [nobs x 3], DISPLACEMENT, the unit is same to those defined by dislocation slip in models
     * D     : [nobs x 9], 9 spatial derivatives of the displacements having 3 elements
     * S     : [nobs x 9], STRESS, the unit depends on that of shear modulus, 6 of them are independent
     * flags : [nobs * nmodle], a pointer of an 1-D array
     *         0 normal; 
     *         1 the Z value of the obs > 0
     *         10 the depth of the fault upper center point reached to surface (depth < 0)
     *         100 singular point (observation point lies on the fault edges);
     *         and more combination scenarios.
    */ 
   
    double lambda; 
    double alpha;
    double theta;
    
    lambda = 2.0*mu*nu / (1.0 - 2.0*nu);
    alpha = (lambda + mu)/(lambda + 2.0*mu);

    int flag1;
    int flag2;
    int iret;

    double *model;
    double *obs;
    double strike, dip;
    double cs, ss;
    double cd, sd; 
    double cs2, ss2; 
    double csss;
    double disl1, disl2, disl3;
    double al1, al2, aw1, aw2; 
    double depth;

    double x, y, z;
    double ux, uy, uz;
    double uxx, uxy, uxz;
    double uyx, uyy, uyz;
    double uzx, uzy, uzz;

    double uxt,  uyt,  uzt;
    double uxxt, uxyt, uxzt;
    double uyxt, uyyt, uyzt;
    double uzxt, uzyt, uzzt;

    int i, j;

    for (i = 0; i < nobs; i++)
    {
	obs = obss + 3*i;
	flag1 = 0;
	flag2 = 0;

	if (*(obs + 2) > 0)
	{
	    // positive z value of the station is given, let flag = 1
	    flag1 = 1;
	    printf("\n%s\n", STARS);
  	    fprintf(stderr, "Error, Observation station (ID: %d) has positive depth!" ,i);
	    printf("\n%s\n", STARS);
	    // exit(EXIT_FAILURE);
	}
	
	// Initialized 
	uxt  = uyt  = uzt  = 0;
        uxxt = uxyt = uxzt = 0;
        uyxt = uyyt = uyzt = 0;
        uzxt = uzyt = uzzt = 0;
	
	for (j = 0; j < nmodel; j++)
	{
	    model = models + 10*j;
	    strike = model[5] - 90.0;
            cs     = cosd(strike);
            ss     = sind(strike);
            cs2    = cs * cs;
            ss2    = ss * ss;
            csss   = cs * ss;

	    dip    = model[6];
            cd     = cosd(dip);
            sd     = sind(dip);

            disl1  = model[7];
            disl2  = model[8];
            disl3  = model[9];
	    
	    // the fault reference point is upper center point
	    // the depth is the depth of upper center point
	    depth  =        model[2];
            al1    = -0.5 * model[3];
            al2    =  0.5 * model[3];
            aw1    = -1.0 * model[4];
            aw2    =  0.0;

            // Can also use R = [cs ss 0; -ss cs 0; 0 0 1].
            // Apply some translations to transfer Observation Cartesian to Fault Coordinate
            x = cs * (obs[0] - model[0]) - ss * (obs[1] - model[1]);
            y = ss * (obs[0] - model[0]) + cs * (obs[1] - model[1]);
            z = obs[2];

            if ((model[3] <= 0.0) || (model[4] <= 0.0) || (depth < 0.0)) {
		flag2 = 10;
	    	printf("\n%s\n", STARS);
	    	fprintf(stderr, "Error, unphysical model! Check fault length, width and the center point depth on upper fault edge should all be positive values.\n");
	    	fprintf(stderr, "Observation Station ID: %d, Fault Patch ID: %d\n", i, j);
	    	fprintf(stderr, "Patch length: %f, width: %f, upper center depth: %f.\n", model[3], model[4], depth);
	    	printf("\n%s\n", STARS);
	    	// exit(EXIT_FAILURE);
	    }
	    
	    dc3d_(&alpha, &x, &y, &z,
	          &depth, &dip,
	          &al1, &al2, &aw1, &aw2,
	          &disl1, &disl2, &disl3,
                  &ux,  &uy,  &uz, 
                  &uxx, &uyx, &uzx,
                  &uxy, &uyy, &uzy,
                  &uxz, &uyz, &uzz,
                  &iret);
	    // printf("alpha: %f, x:%f, y:%f, z:%f, depth:%f, dip:%f, al1:%f, al2:%f, aw1:%f, aw2:%f, dis1:%f, dis2:%f, dis3:%f\n", alpha, x, y, z, depth, dip, al1, al2, aw1, aw2, disl1, disl2, disl3);


             /* flags = 0: normal
              * flags = 1: the Z value of the obs > 0
	      * flags = 10: the depth of the fault upper center point < 0
	      * flags = 100: singular point, observation is on fault edges

              * flag1 = 1: the Z value of the obs > 0
              * flag2 = 10: the depth of the fault upper center point < 0
	      
	      * iret = 0: normal
	      * iret = 1: singular point
	      * iret = 2, the Z value of the obs > 0
	     */
	    if (iret == 1) {iret = 100;}
	    if (iret == 2) {iret = iret - 2;}
	    *(flags + i*nmodel + j) = flag1 + flag2 + iret;

            // rotate then add
            uxt +=  cs*ux + ss*uy;
            uyt += -ss*ux + cs*uy;
            uzt += uz;

	    // 9 spatial derivatives of the displacements
            uxxt += cs2*uxx + csss*(uxy + uyx) + ss2*uyy;
            uxyt += cs2*uxy - ss2*uyx + csss*(-uxx + uyy);
            uxzt += cs*uxz + ss*uyz;
            
            uyxt += -ss*(cs*uxx + ss*uxy) + cs*(cs*uyx + ss*uyy);
            uyyt += ss2*uxx - csss*(uxy + uyx) + cs2*uyy;
            uyzt += -ss*uxz + cs*uyz;
            
            uzxt += cs*uzx + ss*uzy;
            uzyt += -ss*uzx + cs*uzy;
            uzzt += uzz;
	    
	}

	// Calculate U, S, D
	U += 3*i;
	D += 9*i;
	S += 9*i;
	
        U[0] = uxt;
        U[1] = uyt;
        U[2] = uzt;
        
        D[0] = uxxt;  // d11
        D[1] = uxyt;  // d12
        D[2] = uxzt;  // d13
        D[3] = uyxt;  // d21
        D[4] = uyyt;  // d22
        D[5] = uyzt;  // d23
        D[6] = uzxt;  // d31
        D[7] = uzyt;  // d32
        D[8] = uzzt;  // d33
			 //
	// if you want to calculate Strains ...
	// symmetry with 6 independent elements
	// S_ += 9*i;
	// S_[0] = D[0];                 // s_11
	// S_[1] = 0.5 * (D[1] + D[3]);  // s_12
	// S_[2] = 0.5 * (D[2] + D[6]);  // s_13
	// S_[3] = 0.5 * (D[1] + D[3]);  // s_21
	// S_[4] = D[i][4];  	         // s_22
	// S_[5] = 0.5 * (D[5] + D[7]);  // s_23
	// S_[6] = 0.5 * (D[2] + D[6]);  // s_31
	// S_[7] = 0.5 * (D[5] + D[7]);  // s_32
	// S_[8] = D[8] 		 // S-33
	
        // calculate stresses, symmetry with 6 independent elements
        theta   = D[0] + D[4] + D[8];
        S[0] = lambda*theta + 2*mu*D[0];  // s11
        S[1] = mu*(D[1] + D[3]);          // s12
        S[2] = mu*(D[2] + D[6]);          // s13
	S[3] = mu*(D[1] + D[3]);          // s21
        S[4] = lambda*theta + 2*mu*D[4];  // s22
        S[5] = mu*(D[5] + D[7]);          // s23
        S[6] = mu*(D[2] + D[6]);          // s31
	S[7] = mu*(D[5] + D[7]); 	  // s32 
	S[8] = lambda*theta + 2*mu*D[8];  // s33 

    }
}


