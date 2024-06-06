#include <stdio.h>
#include <math.h>
#include "meade_dc3d.c"

int main() {

    /* ----------------------------------------------------------------------------
     * Example usage of  adv
     * ----------------------------------------------------------------------------
     */

    /*
    // Define input variables 
    double y1 = 3.0, y2 = 6.0, y3 = 9.0;
    double a = 4.0, beta = M_PI / 4, nu = 0.5;
    double B1 = 5.0, B2 = 10.0, B3 = 10.0;
    double v1, v2, v3;

    // Call the main function 
    adv(y1, y2, y3, a, beta, nu, B1, B2, B3, &v1, &v2, &v3);
    printf("B1 = %f, B2 = %f, B3 = %f \n", B1, B2, B3);

    // Print the results 
    printf("v1 = %f\n", v1);
    printf("v2 = %f\n", v2);
    printf("v3 = %f\n", v3);
    */

    /* ----------------------------------------------------------------------------
     * Example usage of linePlaneIntersect 
     * ----------------------------------------------------------------------------
     */

    /*
    Vector3 p1 = {-50,  -80, -3};
    Vector3 p2 = {52, 80, 59};
    Vector3 p3 = {45,  -80, 34};

    //Vector3 p1 = {1, 2, 3};
    //Vector3 p2 = {1, 5, 5};
    //Vector3 p3 = {1, 0, 34};

    //Vector3 p1 = {1, 2, 8};
    //Vector3 p2 = {9, 5, 8};
    //Vector3 p3 = {3, 0, 8};
    

    //Vector3 p1 = {1, 2, 8};
    //Vector3 p2 = {1, 5, 8};
    //Vector3 p3 = {1, 0, 8};

    // start point and direction 
    Vector3 linePoint = {10, 11, 56};
    Vector3 lineDir = {0, 0, -56};

    Vector3 intersection;
    linePlaneIntersect(p1, p2, p3, linePoint, lineDir, &intersection);

    printf("Intersection point: (%f, %f, %f)\n", intersection.x, intersection.y, intersection.z);

    // if (result) {
    //     printf("Intersection point: (%f, %f, %f)\n", intersection.x, intersection.y, intersection.z);
    // } else {
    //     printf("No intersection or line is parallel to the plane.\n");
    // }

    */

    /* ----------------------------------------------------------------------------
     * Example usage of advs
     * ----------------------------------------------------------------------------
     */

    /*

    double y1 = 1.0; 
    double y2 = 2.0; 
    double y3 = 3.0; 
    double a = 0.5; 
    double b = 0.2; 
    double nu = 0.3; 
		     
    double B1 = 1.0;
    double B2 = 2.0; 
    double B3 = -3.0; 
		     
    double e11, e22, e33;
    double e12, e13, e23;


    advs(y1, y2, y3, a, b, nu, B1, B2, B3, &e11, &e22, &e33, &e12, &e13, &e23);

    printf("e11=%f, e22=%f, e33=%f\ne12=%f, e13=%f, e23=%f\n", e11, e22, e33, e12, e13, e23);

    */


    /* ----------------------------------------------------------------------------
     * Example usage of CalTriDisps and CalTriStrains
     * ----------------------------------------------------------------------------
     */

    double sx = 2.0, sy = 3.0, sz = 0.0;
    double x[3] = {5.0, 0.0, 2.0};
    double y[3] = {0.0, 5.0, 3.0};
    double z[3] = {0.0, 0.0, 12.0};
    double pr = 0.25;
    double ss = 1.0, ts =2.0, ds = 3.0;

    double U[3] = {0};
    double E[9] = {0};
    CalTriDisps(sx, sy, sz, x, y, z, pr, ss, ts, ds, U);
    CalTriStrains(sx, sy, sz, x, y, z, pr, ss, ts, ds, E);

    printf("Displacement U:\n x = %f, y = %f, z = %f\n", U[0], U[1], U[2]);
    printf("Strains E:\ne11 = %f, e12 = %f, e13 = %f\ne21 = %f, e22 = %f, e23 = %f\ne31 = %f, e32 = %f, e33 = %f\n", E[0], E[1], E[2], E[3], E[4], E[5], E[6], E[7], E[8]);

    return 0;

}




