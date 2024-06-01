#include <stdio.h>
#include <math.h>
#include "meade_disp.c"

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


    /* ----------------------------------------------------------------------------
     * Example usage of CalTriDisps
     * ----------------------------------------------------------------------------
     */

    double sx = 2.0, sy = 3.0, sz = -1.0;
    double x[3] = {5.0, 0.0, 2.0};
    double y[3] = {0.0, 5.0, 3.0};
    double z[3] = {0.0, 0.0, 12.0};
    double pr = 0.25;
    double ss = 1.0, ts =2.0, ds = 3.0;

    double U[3] = {0};
    CalTriDisps(sx, sy, sz, x, y, z, pr, ss, ts, ds, U);

    printf("Displacement U: x = %f, y = %f, z = %f\n", U[0], U[1], U[2]);

    return 0;

}




