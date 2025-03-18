#include <stdio.h>
#include <stdlib.h>
#include "mehdi.h"


/*-----------------------------------------------------------------------------------------------*/

// Calculating X1, X2, X3
void CoordTrans(double x1, double x2, double x3, double A[9],
                double *X1, double *X2, double *X3) {
    /*
     * CoordTrans transforms the coordinates of the vectors, from x1x2x3 coordinate system to X1X2X3 
     * coordinate system. "A" is the transformation matrix, whose columns e1,e2 and e3 are the unit base 
     * vectors of the x1x2x3. The coordinates of e1,e2 and e3 in A must be given in X1X2X3. 
     * The transpose of A (i.e., A') will transform the coordinates from X1X2X3 into x1x2x3.
     */

    // r = A * [x1; x2; x3]
    *X1 = A[0] * x1 + A[1] * x2 + A[2] * x3;  // Row 0: A[0][0], A[0][1], A[0][2]
    *X2 = A[3] * x1 + A[4] * x2 + A[5] * x3;  // Row 1: A[1][0], A[1][1], A[1][2]
    *X3 = A[6] * x1 + A[7] * x2 + A[8] * x3;  // Row 2: A[2][0], A[2][1], A[2][2]
}


/*-----------------------------------------------------------------------------------------------*/
// Function to calculate the trimode values
void trimodefinder(double x, double y, double z, double p1[2], double p2[2], double p3[2], int *trimode) {
    /*
     * Trimodefinder calculates the normalized barycentric coordinates of the points with respect to
     * the TD vertices and specifies the appropriate artefact-free configuration of the angular
     * dislocations for the calculations. The input matrices x, y and z share the same size and
     * correspond to the y, z and x coordinates in the TDCS, respectively. p1, p2 and p3 are
     * two-component matrices representing the y and z coordinates of the TD vertices in the TDCS,
     * respectively. he components of the output (trimode) corresponding to each calculation
     * points, are 1 for the first configuration, -1 for the second configuration and 0 for the 
     * calculation point that lie on the TD sides.
     */

    // Calculate the denominator for barycentric coordinates
    double denominator = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1]);
    
    // Calculate the barycentric coordinates a, b, and c
    double a = ((p2[1] - p3[1]) * (x - p3[0]) + (p3[0] - p2[0]) * (y - p3[1])) / denominator;
    double b = ((p3[1] - p1[1]) * (x - p3[0]) + (p1[0] - p3[0]) * (y - p3[1])) / denominator;
    double c = 1.0 - a - b;

    // Initialize trimode to 1 by default
    *trimode = 1;

    // Check conditions to assign trimode value of -1
    if ((a <= 0 && b > c && c > a) ||
        (b <= 0 && c > a && a > b) ||
        (c <= 0 && a > b && b > c)) {
        *trimode = -1;
    } 
    // Check conditions to assign trimode value of 0
    else if ((a == 0 && b >= 0 && c >= 0) ||
             (a >= 0 && b == 0 && c >= 0) ||
             (a >= 0 && b >= 0 && c == 0)) {
        *trimode = 0;
    }

    // If trimode is 0 and z is not 0, set trimode to 1
    if (*trimode == 0 && z != 0) {
        *trimode = 1;
    }
}

