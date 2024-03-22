#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dc3d.h"
#include "dc3d.c"
#include "disloc3d.h"

#define DEG2RAD (M_PI / 180)
#define cosd(a) (cos((a)*DEG2RAD))
#define sind(a) (sin((a)*DEG2RAD))

void disloc3d(double *models,         int nmodel,
              double *observations,   int nobs,
	      double mu,              double nu,
	      double *U,              double *D,           double *S,
	      int *flags )
{
    // input parameters
    // models: nmodel * 10
    // length, width, depth (positive value), dip, strike, easting, northing, str-slip, dip-selip. opening
    // obs: nons * 3
    // easting, northing, depth (negative values)
    // mu: shear modulus
    // nu: Poisson's ratio
       	
    // output
    // U: DISPLACEMENT
    // D: STRAIN
    // S: STRESS
    // flags
   
    double lambda; 
    double alpha;
    double theta;
    
    lambda = 2.0*mu*nu / (1.0 - 2.0*nu);
    alpha = (lambda + mu)/(lambda + 2.0*mu);

    double *model, *obs;
    double *u, *d, *s;
    int *flag;
    int iret;

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
        obs = observations + 3*i;
	flag = flags + i;
	*flag = 0;

	if (*(obs + 2) > 0)
	{
	    *flag = 1;
	    printf("\n%s\n", STARS);
  	    fprintf(stderr, "Error, Observation (ID: %d) has positive depth!" ,i);
	    printf("\n%s\n", STARS);
	    exit(EXIT_FAILURE);
	}
	else // Initialized 
	{
	    uxt = uyt = uzt = 0;
            uxxt = uxyt = uxzt = 0;
            uyxt = uyyt = uyzt = 0;
            uzxt = uzyt = uzzt = 0;
	}

	for (j = 0; j < nmodel; j++)
	{
	    model = models + 10*j;
            strike = model[4] - 90.0;
            cs   = cosd(strike);
            ss   = sind(strike);
            cs2  = cs * cs;
            ss2  = ss * ss;
            csss = cs * ss;

            dip = model[3];
            cd  = cosd(dip);
            sd  = sind(dip);

            disl1 = model[7];
            disl2 = model[8];
            disl3 = model[9];
	    // the depth is upper center point
	    depth = model[2];
            //depth = model[2] + 0.5*model[1]*sd;
            al1 = -0.5 * model[0];
            al2 =  0.5 * model[0];
            aw1 = -model[1];
            aw2 =  0.0;


            // Can also use R = [cs ss 0; -ss cs 0; 0 0 1].
            // Apply some translations to transfer Cartesian to Fault 
            x = cs * (obs[0] - model[5]) - ss * (obs[1] - model[6]);
            y = ss * (obs[0] - model[5]) + cs * (obs[1] - model[6]);
            z = obs[2];

            if ((model[0] <= 0.0) ||
	        (model[1] <= 0.0) ||
	        (depth < 0.0))
	    {    
		*flag = 1;
	    	printf("\n%s\n", STARS);
	    	fprintf(stderr, "Error, unphysical model!!!\n");
	    	fprintf(stderr, "Observation ID: %d, patches ID: %d\n", i, j);
	    	fprintf(stderr, "Patch width: %f, length: %f, upper center depth: %f.\n", model[0], model[1], depth);
	    	printf("\n%s\n", STARS);
	    	exit(EXIT_FAILURE);
	    }
	    else
	    {
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


		*flag = iret;

                // rotate then add
                uxt +=  cs*ux + ss*uy;
                uyt += -ss*ux + cs*uy;
                uzt += uz;

		// strain tensors transformation 
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
	}

	// Calculate U, S, D
        u = U + 3*i;
	d = D + 3*i;
	s = S + 6*i;

        u[0] = uxt;
        u[1] = uyt;
        u[2] = uzt;
        
        d[0] = uxxt;  // d11
        d[1] = uxyt;  // d12
        d[2] = uxzt;  // d13

        d[3] = uyxt;  // d21
        d[4] = uyyt;  // d22
        d[5] = uyzt;  // d23

        d[6] = uzxt;  // d31
        d[7] = uzyt;  // d32
        d[8] = uzzt;  // d33

        // calculate stresses
        theta = d[0] + d[4] + d[8];
        s[0] = lambda*theta + 2*mu*d[0];  // s11
        s[1] = mu*(d[1] + d[3]);          // s12
        s[2] = mu*(d[2] + d[6]);          // s13
        s[3] = lambda*theta + 2*mu*d[4];  // s22
        s[4] = mu*(d[5] + d[7]);          // s23
        s[5] = lambda*theta + 2*mu*d[8];  // s33

    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// int main()
// {
//     // models: nmodel * 10
//     // length, width, depth, dip, strike, easting, northing, str-slip, dip-selip. opening
//     double alpha = 0.6, x = 1.0, y = 1.0, z = -1.0;
//     double depth = 1.0, dip = 45.0, al1 = -1.7, al2 = 1.7;
//     double aw1 = -1.7, aw2 = 1.7, dis1 = 5.0, dis2 = 5.0, dis3 = 0.0;
//     double ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz;
//     int iret;
//     dc3d_(&alpha, &x, &y, &z, &depth, &dip,
//    			&al1, &al2, &aw1, &aw2,
//    			&dis1, &dis2, &dis3,
//    			&ux, &uy, &uz, 
//    			&uxx, &uyx, &uzx,
//    			&uxy, &uyy, &uzy,
//    			&uxz, &uyz, &uzz,
//    			&iret);
//    
//     printf("The ux is: %f, the uy is %f, the uz is %f.\n", ux, uy, uz);
//     printf("The uxx is: %f, the uyx is %f, the uzx is %f.\n", uxx, uyx, uzx);
//     printf("The uxy is: %f, the uyy is %f, the uzy is %f.\n", uxy, uyy, uzy);
//     printf("The uxz is: %f, the uyz is %f, the uzz is %f.\n", uxz, uyz, uzz);
//     printf("The iret is: %d\n", iret);
//    
//     return 0;
// }
// 
