#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dc3d0.h"
#include "dc3d0.c"


#define DEG2RAD (M_PI / 180)
#define cosd(a) (cos((a)*DEG2RAD))
#define sind(a) (sin((a)*DEG2RAD))

void disloc3d0(double *model,
	       double *observations, int nobs, 
	       double mu, double nu,
	       double *U, double *D, double *S,
               int *flags)
{
     /*
     *
     * Note: for only one point source and many observations.
     * 
     * */
    
    // input parameters:
    // model: 1 * 6
    // depth (positive value), dip (degree), pot1, pot2, pot3, pot4
    // obs: nobs * 3
    // easting, northing, depth (negative values)
    // mu : Shear modulus
    // nu : Poisson's ratio

    
    // output parameters:
    // U: DISPLACEMENT
    // D: STRAIN
    // S: STRESS
    // flag

    double lambda; 
    double alpha;
    double theta;
    
    lambda = 2.0*mu*nu / (1.0 - 2.0*nu);
    alpha = (lambda + mu)/(lambda + 2.0*mu);

    double *obs;
    double *u, *d, *s;
    int *flag;
    int iret;

    double depth, dip;
    double pot1, pot2, pot3, pot4;

    double x, y, z;
    double ux, uy, uz;
    double uxx, uxy, uxz;
    double uyx, uyy, uyz;
    double uzx, uzy, uzz;


    int i;

    for (i = 0; i < nobs; i++)
    {
        obs = observations + 3*i;
	flag = flags + i;
	*flag = 0;

	x = obs[0];
	y = obs[1];
	z = obs[2];

	if (z > 0)
	{
	    *flag = 1;
	    printf("\n%s\n", STARS);
  	    fprintf(stderr, "Error, Observation (ID: %d) has positive depth!" ,i);
	    printf("\n%s\n", STARS);
	    exit(EXIT_FAILURE);
	}

	// else 
	// {
	
	// point source model related
	depth = model[0];
	dip   = model[1];
	pot1  = model[2];
	pot2  = model[3];
	pot3  = model[4];
	pot4  = model[5];

	if ( depth < 0.0)
	{
            *flag = 1;
		printf("\n%s\n", STARS);
		fprintf(stderr, "Error, unphysical model!!!\n");
		fprintf(stderr, "The point source model should have positive depth (here depth = %f)!\n", depth);
		printf("\n%s\n", STARS);
		exit(EXIT_FAILURE);
	}
	// }

	dc3d0_(&alpha, &x, &y, &z,
	       &depth, &dip,
	       &pot1, &pot2, &pot3, &pot4,
               &ux,  &uy,  &uz, 
               &uxx, &uyx, &uzx,
               &uxy, &uyy, &uzy,
               &uxz, &uyz, &uzz,
               &iret);

	u = U + 3*i;
	d = D + 3*i;
	s = S + 6*i;

        u[0] = ux;
        u[1] = uy;
        u[2] = uz;
        
        d[0] = uxx;  // d11
        d[1] = uxy;  // d12
        d[2] = uxz;  // d13

        d[3] = uyx;  // d21
        d[4] = uyy;  // d22
        d[5] = uyz;  // d23

        d[6] = uzx;  // d31
        d[7] = uzy;  // d32
        d[8] = uzz;  // d33

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


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

 // int main()
 // {
 // 	double alpha = 0.6, x = 1.0, y = 1.0, z = -1.0;
 // 	double depth = 1.0, dip = 90.0;
 // 	double pot1 = 1.0, pot2 = 2.0, pot3 = 3.0, pot4 = 4.0;
 // 	double ux0, uy0, uz0, uxx0, uyx0, uzx0, uxy0, uyy0, uzy0, uxz0, uyz0, uzz0;
 // 	int iret0;
 // 
 // 	printf("+-+-+-+-+-+-+-+-+-++-+-+-+-++-+-+-+-++-+-+-+-++-+-+-+-++-+-+-+-+\n");
 // 	printf("+-+-+-+-+-+-+-+-+-++-+-+-+-++-+-+-+-++-+-+-+-++-+-+-+-++-+-+-+-+\n");
 // 
 // 	dc3d0_(&alpha, &x, &y, &z, &depth, &dip,
 // 		&pot1, &pot2, &pot3, &pot4,
 // 	        &ux0, &uy0, &uz0, 
 // 	        &uxx0, &uyx0, &uzx0,
 // 	        &uxy0, &uyy0, &uzy0,
 // 	        &uxz0, &uyz0, &uzz0,
 // 	        &iret0);
 // 
 // 	printf("The ux0 is: %f, the uy0 is %f, the uz0 is %f.\n", ux0, uy0, uz0);
 //         printf("The uxx0 is: %f, the uyx0 is %f, the uzx0 is %f.\n", uxx0, uyx0, uzx0);
 //         printf("The uxy0 is: %f, the uyy0 is %f, the uzy0 is %f.\n", uxy0, uyy0, uzy0);
 //         printf("The uxz0 is: %f, the uyz0 is %f, the uzz0 is %f.\n", uxz0, uyz0, uzz0);
 //         printf("The iret is: %d\n", iret0);
 // 
 // 	return 0;
 // 
 // }

