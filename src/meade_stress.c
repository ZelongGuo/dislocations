#include <stdio.h>
#include <math.h>
#include <float.h>

/*
 *  Calculating the stress and strains with triangular elements.
 *  Referring to the Matlab codes of Meade
 *
 *  NOTE: The downward Z is positive!!! 
 *
 *  Author: Zelong Guo
 *  05.25.2024, @ Potsdam, Germany
 *  zelong.guo@outlook.com
 *
 */


/* A struct for vector and also for point in 3D */
typedef struct {
    double x, y, z;
} Vector3;

/* Calculating the difference of vectors */
Vector3 subtract(const Vector3 a, const Vector3 b) {
    Vector3 result = {a.x - b.x, a.y - b.y, a.z - b.z};
    return result;
}

/* cross product of two vectors */
Vector3 crossProduct(const Vector3 a, const Vector3 b) {
    Vector3 result = {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
    return result;
}

/* dot product of two vectors */
double dotProduct(const Vector3 a, const Vector3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* norm */
double norm(const Vector3 a) {
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

void normalize(Vector3 *a) {
    double length = norm(*a);
    a->x /= length;
    a->y /= length;
    a->z /= length;
}

/* the point-normal form of the plane equation
 * normal vector (A, B, C), point (x0, y0, z0)
 * A(x-x0) + B(y-y0) + C(z-z0) = 0 --> Ax + By + Cz - (Ax0 + By0 + Cz0) = 0
 */
void planeFromPoints(Vector3 p1, Vector3 p2, Vector3 p3, double* A, double* B, double* C, double* D) {
    Vector3 v1 = subtract(p2, p1);
    Vector3 v2 = subtract(p3, p1);
    /* normal vector */
    Vector3 normal = crossProduct(v1, v2);
    *A = normal.x;
    *B = normal.y;
    *C = normal.z;
    *D = -dotProduct(normal, p1);
    
    // printf("v1: (%f, %f, %f)\n", v1.x, v1.y, v1.z);
    // printf("v2: (%f, %f, %f)\n", v2.x, v2.y, v2.z);
    // printf("normal: (%f, %f, %f)\n", normal.x, normal.y, normal.z);
    // printf("A: %f, B: %f, C: %f, D: %f\n", *A, *B, *C, *D);
}

/* Calculating the intersection point of Line and Plane using parameter t.
 * The Line goes through the station point (sx, sy, sz), and its direction vector is (0, 0 , -sz), see CalTriDisps function. The Plane is defined by 3 points of the triangle element.
 */
void linePlaneIntersect(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 linePoint, Vector3 lineDir, Vector3* intersection) {
    double A, B, C, D;
    planeFromPoints(p1, p2, p3, &A, &B, &C, &D);

    /* parametric equation of line: (linePoint) + t * (lineDir)
     * direction vector: (v1, v2, v3),  point (x0, y0, z0)
     * x = x0 + v1*t, y = y0 + v2 * t, z = z0 + v3 * t 
     */
    double numerator = -D - (A * linePoint.x + B * linePoint.y + C * linePoint.z);
    double denominator = A * lineDir.x + B * lineDir.y + C * lineDir.z;

    // printf("numerator: %e\ndenumerator: %e\n", numerator, denominator);
    if (denominator == 0) {
        if (numerator == 0) {
            /* if the point is on the plane, the intersection is itself */
            *intersection = linePoint;
	    // printf("The point in Plane, the intersection is itself.\n"); 
        } 
	/*
	else {
	    // if the line is parallel to the plane but no intersection
	    printf("Line is parallel to the Plane but no intersection.\n");
        }
	*/
	denominator = DBL_EPSILON;
    }

    double t = numerator / denominator;
    intersection->x = linePoint.x + t * lineDir.x;
    intersection->y = linePoint.y + t * lineDir.y;
    intersection->z = linePoint.z + t * lineDir.z;
}

/* Check if the point is inside the projection of triangle in xy plane, using barycentric coordinates */
int isPointInTriangle2D(double px, double py, const Vector3* a, const Vector3* b, const Vector3* c) {
    double denominator = ((b->y - c->y) * (a->x - c->x) + (c->x - b->x) * (a->y - c->y));
    double alpha = ((b->y - c->y) * (px - c->x) + (c->x - b->x) * (py - c->y)) / denominator;
    double beta = ((c->y - a->y) * (px - c->x) + (a->x - c->x) * (py - c->y)) / denominator;
    double gamma = 1.0 - alpha - beta;
    return (alpha >= 0) && (beta >= 0) && (gamma >= 0);
}

/* -------------------------------------------------------------------------- */
void swap(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

/* Rotate a vector by an angle alpha */
void RotateXyVec(const double x, const double y, double alpha, double *xp, double *yp) {
    alpha = M_PI / 180.0 * alpha;  
    *xp = cos(alpha) * x - sin(alpha) * y;  /* Rotate x coordinate */
    *yp = sin(alpha) * x + cos(alpha) * y;  /* Rotate y coordinate */
}

/* -------------------------------------------------------------------------- */
/* Define a function to calculate the cotangent */
double cot(double x) {
    return 1.0 / tan(x);
}

/* Define a function to calculate atan2 */ 
double atan2_custom(double y, double x) {
    return atan2(y, x);
}

void adv(double y1, double y2, double y3, double a, double beta, double nu, 
         double B1, double B2, double B3, double *v1, double *v2, double *v3) {
    
    double sinbeta = sin(beta);
    double cosbeta = cos(beta);
    double cotbeta = cot(beta);

    double z1 = y1 * cosbeta - y3 * sinbeta;
    double z3 = y1 * sinbeta + y3 * cosbeta;
    double R2 = y1 * y1 + y2 * y2 + y3 * y3;
    double R = sqrt(R2);
    double y3bar = y3 + 2.0 * a;
    double z1bar = y1 * cosbeta + y3bar * sinbeta;
    double z3bar = -y1 * sinbeta + y3bar * cosbeta;
    double R2bar = y1 * y1 + y2 * y2 + y3bar * y3bar;
    double Rbar = sqrt(R2bar);
    double F = -atan2_custom(y2, y1) + atan2_custom(y2, z1) + atan2_custom(y2 * R * sinbeta, y1 * z1 + (y2 * y2) * cosbeta);
    double Fbar = -atan2_custom(y2, y1) + atan2_custom(y2, z1bar) + atan2_custom(y2 * Rbar * sinbeta, y1 * z1bar + (y2 * y2) * cosbeta);

    /* Case I: Burgers vector (B1,0,0) */
    double v1InfB1 = 2.0 * (1.0 - nu) * (F + Fbar) - y1 * y2 * (1.0 / (R * (R - y3)) + 1.0 / (Rbar * (Rbar + y3bar))) - 
                     y2 * cosbeta * ((R * sinbeta - y1) / (R * (R - z3)) + (Rbar * sinbeta - y1) / (Rbar * (Rbar + z3bar)));
    double v2InfB1 = (1.0 - 2.0 * nu) * (log(R - y3) + log(Rbar + y3bar) - cosbeta * (log(R - z3) + log(Rbar + z3bar))) - 
                     y2 * y2 * (1.0 / (R * (R - y3)) + 1.0 / (Rbar * (Rbar + y3bar)) - cosbeta * (1.0 / (R * (R - z3)) + 1.0 / (Rbar * (Rbar + z3bar))));
    double v3InfB1 = y2 * (1.0 / R - 1.0 / Rbar - cosbeta * ((R * cosbeta - y3) / (R * (R - z3)) - (Rbar * cosbeta + y3bar) / (Rbar * (Rbar + z3bar))));
    v1InfB1 = v1InfB1 / (8.0 * M_PI * (1.0 - nu));
    v2InfB1 = v2InfB1 / (8.0 * M_PI * (1.0 - nu));
    v3InfB1 = v3InfB1 / (8.0 * M_PI * (1.0 - nu));

    double v1CB1 = -2.0 * (1.0 - nu) * (1.0 - 2.0 * nu) * Fbar * (cotbeta * cotbeta) + 
                   (1.0 - 2.0 * nu) * y2 / (Rbar + y3bar) * ((1.0 - 2.0 * nu - a / Rbar) * cotbeta - y1 / (Rbar + y3bar) * (nu + a / Rbar)) + 
                   (1.0 - 2.0 * nu) * y2 * cosbeta * cotbeta / (Rbar + z3bar) * (cosbeta + a / Rbar) + 
                   a * y2 * (y3bar - a) * cotbeta / (Rbar * Rbar * Rbar) + 
                   y2 * (y3bar - a) / (Rbar * (Rbar + y3bar)) * (-(1.0 - 2.0 * nu) * cotbeta + y1 / (Rbar + y3bar) * (2.0 * nu + a / Rbar) + a * y1 / (Rbar * Rbar)) + 
                   y2 * (y3bar - a) / (Rbar * (Rbar + z3bar)) * (cosbeta / (Rbar + z3bar) * ((Rbar * cosbeta + y3bar) * ((1.0 - 2.0 * nu) * cosbeta - a / Rbar) * cotbeta + 2.0 * (1.0 - nu) * (Rbar * sinbeta - y1) * cosbeta) - a * y3bar * cosbeta * cotbeta / (Rbar * Rbar));
    double v2CB1 = (1.0 - 2.0 * nu) * ((2.0 * (1.0 - nu) * (cotbeta * cotbeta) - nu) * log(Rbar + y3bar) - (2.0 * (1.0 - nu) * (cotbeta * cotbeta) + 1.0 - 2.0 * nu) * cosbeta * log(Rbar + z3bar)) - 
                   (1.0 - 2.0 * nu) / (Rbar + y3bar) * (y1 * cotbeta * (1.0 - 2.0 * nu - a / Rbar) + nu * y3bar - a + (y2 * y2) / (Rbar + y3bar) * (nu + a / Rbar)) - 
                   (1.0 - 2.0 * nu) * z1bar * cotbeta / (Rbar + z3bar) * (cosbeta + a / Rbar) - a * y1 * (y3bar - a) * cotbeta / (Rbar * Rbar * Rbar) + 
                   (y3bar - a) / (Rbar + y3bar) * (-2.0 * nu + 1.0 / Rbar * ((1.0 - 2.0 * nu) * y1 * cotbeta - a) + (y2 * y2) / (Rbar * (Rbar + y3bar)) * (2.0 * nu + a / Rbar) + a * (y2 * y2) / (Rbar * Rbar * Rbar)) + 
                   (y3bar - a) / (Rbar + z3bar) * ((cosbeta * cosbeta) - 1.0 / Rbar * ((1.0 - 2.0 * nu) * z1bar * cotbeta + a * cosbeta) + a * y3bar * z1bar * cotbeta / (Rbar * Rbar * Rbar) - 1.0 / (Rbar * (Rbar + z3bar)) * ((y2 * y2) * (cosbeta * cosbeta) - a * z1bar * cotbeta / Rbar * (Rbar * cosbeta + y3bar)));
    double v3CB1 = 2.0 * (1.0 - nu) * (((1.0 - 2.0 * nu) * Fbar * cotbeta) + (y2 / (Rbar + y3bar) * (2.0 * nu + a / Rbar)) - (y2 * cosbeta / (Rbar + z3bar) * (cosbeta + a / Rbar))) + 
                  y2 * (y3bar - a) / Rbar * (2.0 * nu / (Rbar + y3bar) + a / (Rbar * Rbar)) + 
                  y2 * (y3bar - a) * cosbeta / (Rbar * (Rbar + z3bar)) * (1.0 - 2.0 * nu - (Rbar * cosbeta + y3bar) / (Rbar + z3bar) * (cosbeta + a / Rbar) - a * y3bar / (Rbar * Rbar));

    v1CB1 = v1CB1 / (4.0 * M_PI * (1.0 - nu));
    v2CB1 = v2CB1 / (4.0 * M_PI * (1.0 - nu));
    v3CB1 = v3CB1 / (4.0 * M_PI * (1.0 - nu));


    double v1B1 = v1InfB1 + v1CB1;
    double v2B1 = v2InfB1 + v2CB1;
    double v3B1 = v3InfB1 + v3CB1;

    /* Case II: Burgers vector (0,B2,0) */
    double v1InfB2 = -(1 - 2 * nu) * (log(R - y3) + log(Rbar + y3bar) - cosbeta * (log(R - z3) + log(Rbar + z3bar))) + 
                     y1 * y1 * (1 / (R * (R - y3)) + 1 / (Rbar * (Rbar + y3bar))) + z1 * (R * sinbeta - y1) / (R * (R - z3)) + z1bar * (Rbar * sinbeta - y1) / (Rbar * (Rbar + z3bar));
    double v2InfB2 = 2 * (1 - nu) * (F + Fbar) + y1 * y2 * (1 / (R * (R - y3)) + 1 / (Rbar * (Rbar + y3bar))) - y2 * (z1 / (R * (R - z3)) + z1bar / (Rbar * (Rbar + z3bar)));
    double v3InfB2 = -(1 - 2 * nu) * sinbeta * (log(R - z3) - log(Rbar + z3bar)) - y1 * (1 / R - 1 / Rbar) + z1 * (R * cosbeta - y3) / (R * (R - z3)) - z1bar * (Rbar * cosbeta + y3bar) / (Rbar * (Rbar + z3bar));
    v1InfB2 = v1InfB2 / (8 * M_PI * (1 - nu));
    v2InfB2 = v2InfB2 / (8 * M_PI * (1 - nu));
    v3InfB2 = v3InfB2 / (8 * M_PI * (1 - nu));
    
    double v1CB2 = (1 - 2 * nu) * ((2 * (1 - nu) * (cotbeta * cotbeta) + nu) * log(Rbar + y3bar) - (2 * (1 - nu) * (cotbeta * cotbeta) + 1) * cosbeta * log(Rbar + z3bar)) + 
                   (1 - 2 * nu) / (Rbar + y3bar) * (-(1 - 2 * nu) * y1 * cotbeta + nu * y3bar - a + a * y1 * cotbeta / Rbar + (y1 * y1) / (Rbar + y3bar) * (nu + a / Rbar)) - 
                   (1 - 2 * nu) * cotbeta / (Rbar + z3bar) * (z1bar * cosbeta - a * (Rbar * sinbeta - y1) / (Rbar * cosbeta)) - a * y1 * (y3bar - a) * cotbeta / (Rbar * Rbar * Rbar) + 
                   (y3bar - a) / (Rbar + y3bar) * (2 * nu + 1 / Rbar * ((1 - 2 * nu) * y1 * cotbeta + a) - (y1 * y1) / (Rbar * (Rbar + y3bar)) * (2 * nu + a / Rbar) - a * (y1 * y1) / (Rbar * Rbar * Rbar)) + 
                   (y3bar - a) * cotbeta / (Rbar + z3bar) * (-cosbeta * sinbeta + a * y1 * y3bar / (Rbar * Rbar * Rbar * cosbeta) + (Rbar * sinbeta - y1) / Rbar * (2 * (1 - nu) * cosbeta - (Rbar * cosbeta + y3bar) / (Rbar + z3bar) * (1 + a / (Rbar * cosbeta))));
    double v2CB2 = 2 * (1 - nu) * (1 - 2 * nu) * Fbar * cotbeta * cotbeta + (1 - 2 * nu) * y2 / (Rbar + y3bar) * (-(1 - 2 * nu - a / Rbar) * cotbeta + y1 / (Rbar + y3bar) * (nu + a / Rbar)) - 
                   (1 - 2 * nu) * y2 * cotbeta / (Rbar + z3bar) * (1 + a / (Rbar * cosbeta)) - a * y2 * (y3bar - a) * cotbeta / (Rbar * Rbar * Rbar) + 
                   y2 * (y3bar - a) / (Rbar * (Rbar + y3bar)) * ((1 - 2 * nu) * cotbeta - 2 * nu * y1 / (Rbar + y3bar) - a * y1 / Rbar * (1 / Rbar + 1 / (Rbar + y3bar))) + 
                   y2 * (y3bar - a) * cotbeta / (Rbar * (Rbar + z3bar)) * (-2 * (1 - nu) * cosbeta + (Rbar * cosbeta + y3bar) / (Rbar + z3bar) * (1 + a / (Rbar * cosbeta)) + a * y3bar / ((Rbar * Rbar) * cosbeta));
    double v3CB2 = -2 * (1 - nu) * (1 - 2 * nu) * cotbeta * (log(Rbar + y3bar) - cosbeta * log(Rbar + z3bar)) - 
                   2 * (1 - nu) * y1 / (Rbar + y3bar) * (2 * nu + a / Rbar) + 2 * (1 - nu) * z1bar / (Rbar + z3bar) * (cosbeta + a / Rbar) + 
                   (y3bar - a) / Rbar * ((1 - 2 * nu) * cotbeta - 2 * nu * y1 / (Rbar + y3bar) - a * y1 / (Rbar * Rbar)) - 
                   (y3bar - a) / (Rbar + z3bar) * (cosbeta * sinbeta + (Rbar * cosbeta + y3bar) * cotbeta / Rbar * (2 * (1 - nu) * cosbeta - (Rbar * cosbeta + y3bar) / (Rbar + z3bar)) + a / Rbar * (sinbeta - y3bar * z1bar / (Rbar * Rbar) - z1bar * (Rbar * cosbeta + y3bar) / (Rbar * (Rbar + z3bar))));
    v1CB2 = v1CB2 / (4 * M_PI * (1 - nu));
    v2CB2 = v2CB2 / (4 * M_PI * (1 - nu));
    v3CB2 = v3CB2 / (4 * M_PI * (1 - nu));
    
    double v1B2 = v1InfB2 + v1CB2;
    double v2B2 = v2InfB2 + v2CB2;
    double v3B2 = v3InfB2 + v3CB2;
    
    /* Case III: Burgers vector (0,0,B3) */
    double v1InfB3 = y2 * sinbeta * ((R * sinbeta - y1) / (R * (R - z3)) + (Rbar * sinbeta - y1) / (Rbar * (Rbar + z3bar)));
    double v2InfB3 = (1 - 2 * nu) * sinbeta * (log(R - z3) + log(Rbar + z3bar)) - (y2 * y2) * sinbeta * (1 / (R * (R - z3)) + 1 / (Rbar * (Rbar + z3bar)));
    double v3InfB3 = 2 * (1 - nu) * (F - Fbar) + y2 * sinbeta * ((R * cosbeta - y3) / (R * (R - z3)) - (Rbar * cosbeta + y3bar) / (Rbar * (Rbar + z3bar)));
    v1InfB3 = v1InfB3 / (8 * M_PI * (1 - nu));
    v2InfB3 = v2InfB3 / (8 * M_PI * (1 - nu));
    v3InfB3 = v3InfB3 / (8 * M_PI * (1 - nu));
    
    double v1CB3 = (1 - 2 * nu) * (y2 / (Rbar + y3bar) * (1 + a / Rbar) - y2 * cosbeta / (Rbar + z3bar) * (cosbeta + a / Rbar)) -
        y2 * (y3bar - a) / Rbar * (a / (Rbar * Rbar) + 1 / (Rbar + y3bar)) +
        y2 * (y3bar - a) * cosbeta / (Rbar * (Rbar + z3bar)) * ((Rbar * cosbeta + y3bar) / (Rbar + z3bar) * (cosbeta + a / Rbar) + a * y3bar / (Rbar * Rbar));
    double v2CB3 = (1 - 2 * nu) * (-sinbeta * log(Rbar + z3bar) - y1 / (Rbar + y3bar) * (1 + a / Rbar) + z1bar / (Rbar + z3bar) * (cosbeta + a / Rbar)) +
        y1 * (y3bar - a) / Rbar * (a / (Rbar * Rbar) + 1 / (Rbar + y3bar)) -
        (y3bar - a) / (Rbar + z3bar) * (sinbeta * (cosbeta - a / Rbar) + z1bar / Rbar * (1 + a * y3bar / (Rbar * Rbar)) -
        1 / (Rbar * (Rbar + z3bar)) * ((y2 * y2) * cosbeta * sinbeta - a * z1bar / Rbar * (Rbar * cosbeta + y3bar)));
    double v3CB3 = 2 * (1 - nu) * Fbar + 2 * (1 - nu) * (y2 * sinbeta / (Rbar + z3bar) * (cosbeta + a / Rbar)) +
        y2 * (y3bar - a) * sinbeta / (Rbar * (Rbar + z3bar)) * (1 + (Rbar * cosbeta + y3bar) / (Rbar + z3bar) * (cosbeta + a / Rbar) + a * y3bar / (Rbar * Rbar));
    v1CB3 = v1CB3 / (4 * M_PI * (1 - nu));
    v2CB3 = v2CB3 / (4 * M_PI * (1 - nu));
    v3CB3 = v3CB3 / (4 * M_PI * (1 - nu));
    
    double v1B3 = v1InfB3 + v1CB3;
    double v2B3 = v2InfB3 + v2CB3;
    double v3B3 = v3InfB3 + v3CB3;
    
    /* Sum the for each slip component */
    *v1 = B1 * v1B1 + B2 * v1B2 + B3 * v1B3;
    *v2 = B1 * v2B1 + B2 * v2B2 + B3 * v2B3;
    *v3 = B1 * v3B1 + B2 * v3B2 + B3 * v3B3;
}


/* -------------------------------------------------------------------------- */
void CalTriDisps(const double sx, const double sy, const double sz, double *x, double *y, double *z, double pr, double ss, double ts, double ds, double *U) {

    Vector3 v1 = {x[1] - x[0], y[1] - y[0], z[1] - z[0]};
    Vector3 v2 = {x[2] - x[0], y[2] - y[0], z[2] - z[0]};
    Vector3 normVec = crossProduct(v1, v2);
    normalize(&normVec);
    printf("normVec: (%f, %f, %f). \n", normVec.x, normVec.y, normVec.z); 

    /* Enforce clockwise circulation */
    if (normVec.z < 0) { 
        normVec.x = -normVec.x;
        normVec.y = -normVec.y;
        normVec.z = -normVec.z;
        swap(&x[1], &x[2]);
        swap(&y[1], &y[2]);
        swap(&z[1], &z[2]);
    }

    /* ------------------------------------
     * NOTE: Based on IEC-60559:
     * atan2(±0, −0) returns ±π
     * atan2(±0, +0) returns ±0.
     * ------------------------------------
     */

    if (normVec.x == -0.0) {normVec.x = 0.0;}
    if (normVec.y == -0.0) {normVec.y = 0.0;}

    /*
    printf("normVec.xyz:(%f, %f, %f) \n", normVec.x, normVec.y, normVec.z);
    printf("x0, x1, x2: [%f %f %f]\n", x[0], x[1], x[2]); 
    printf("y0, y1, y2: [%f %f %f]\n", y[0], y[1], y[2]); 
    printf("z0, z1, z2: [%f %f %f]\n", z[0], z[1], z[2]); 
    */

    Vector3 strikeVec = {-sin(atan2(normVec.y, normVec.x)), cos(atan2(normVec.y, normVec.x)), 0};
    printf("normVec.xyz2:(%e, %e, %e) \n", normVec.x, normVec.y, normVec.z);
    printf("strikeVec: (%f, %f, %f). \n", strikeVec.x, strikeVec.y, strikeVec.z);
    Vector3 dipVec = crossProduct(normVec, strikeVec);
    printf("dipVec: (%f, %f, %f). \n", dipVec.x, dipVec.y, dipVec.z);

    double slipComp[3] = {ss, ds, ts};
    Vector3 slipVec = {
        strikeVec.x * slipComp[0] + dipVec.x * slipComp[1] + normVec.x * slipComp[2],
        strikeVec.y * slipComp[0] + dipVec.y * slipComp[1] + normVec.y * slipComp[2],
        strikeVec.z * slipComp[0] + dipVec.z * slipComp[1] + normVec.z * slipComp[2]
    };

    // printf("slipVec: %f, %f, %f.\n", slipVec.x, slipVec.y, slipVec.z); 

    /* Solution vectors */
    for (int i = 0; i < 3; i++) {
	*(U + i) = 0.0;
    }

    /* Create temporary arrays with 4 elements */
    double x_temp[4] = {x[0], x[1], x[2], x[0]};
    double y_temp[4] = {y[0], y[1], y[2], y[0]};
    double z_temp[4] = {z[0], z[1], z[2], z[0]};

    for (int iTri = 0; iTri < 3; ++iTri) {
        /* Calculate strike and dip of current leg */ 
        double strike = 180 / M_PI * atan2(y_temp[iTri + 1] - y_temp[iTri], x_temp[iTri + 1] - x_temp[iTri]);
        double segMapLength = sqrt(pow(x_temp[iTri] - x_temp[iTri + 1], 2) + pow(y_temp[iTri] - y_temp[iTri + 1], 2));
        double rx, ry;
        RotateXyVec(x_temp[iTri + 1] - x_temp[iTri], y_temp[iTri + 1] - y_temp[iTri], -strike, &rx, &ry);
        double dip = 180 / M_PI * atan2(z_temp[iTri + 1] - z_temp[iTri], rx);

        double beta;
        if (dip >= 0) {
            beta = M_PI / 180 * (90 - dip);
            if (beta > M_PI / 2) {
                beta = M_PI / 2 - beta;
            }
        } else {
            beta = -M_PI / 180 * (90 + dip);
            if (beta < -M_PI / 2) {
                beta = M_PI / 2 - fabs(beta);
            }
        }

	Vector3 ssVec = {cos(strike / 180 * M_PI), sin(strike / 180 * M_PI), 0};
        Vector3 tsVec = {-sin(strike / 180 * M_PI), cos(strike / 180 * M_PI), 0};
        Vector3 dsVec = crossProduct(ssVec, tsVec);

        double lss = ssVec.x * slipVec.x + ssVec.y * slipVec.y + ssVec.z * slipVec.z;
        double lts = tsVec.x * slipVec.x + tsVec.y * slipVec.y + tsVec.z * slipVec.z;
        double lds = dsVec.x * slipVec.x + dsVec.y * slipVec.y + dsVec.z * slipVec.z;


	if (fabs(beta) > 0.000001 && fabs(beta - M_PI) > 0.000001) {
            /* First angular dislocation */
            double sx1, sy1;
            RotateXyVec(sx - x_temp[iTri], sy - y_temp[iTri], -strike, &sx1, &sy1);
            double ux1, uy1, uz1;
            adv(sx1, sy1, sz - z_temp[iTri], z_temp[iTri], beta, pr, lss, lts, lds, &ux1, &uy1, &uz1);  

            /* Second angular dislocation */
            double sx2, sy2;
            RotateXyVec(sx - x_temp[iTri+1], sy - y_temp[iTri + 1], -strike, &sx2, &sy2);
            double ux2, uy2, uz2;
            adv(sx2, sy2, sz - z_temp[iTri + 1], z_temp[iTri + 1], beta, pr, lss, lts, lds, &ux2, &uy2, &uz2); 

            /* Rotate vectors to correct for strike */
            double uxn, uyn;
            RotateXyVec(ux1 -ux2, uy1 - uy2, strike, &uxn, &uyn);
            double uzn = uz1 - uz2;

            /* Add the displacements from current leg */
	    *U       += uxn;
	    *(U + 1) += uyn;
	    *(U + 2) += uzn;
	}
    }

    Vector3 p1 = {x[0], y[0], z[0]};
    Vector3 p2 = {x[1], y[1], z[1]};
    Vector3 p3 = {x[2], y[2], z[2]};
    if (isPointInTriangle2D(sx, sy, &p1, &p2, &p3)) {
	Vector3 intersection;
	Vector3 linePoint = {sx, sy,  sz};
	Vector3 lineDir   = { 0,  0, -sz};
	linePlaneIntersect(p1, p2, p3, linePoint, lineDir, &intersection);
	if (intersection.z < sz) {
	    printf("Now the intersection.z < sz: %f < %f.\n", intersection.z, sz);
	    *U       = *U       - slipVec.x;
	    *(U + 1) = *(U + 1) - slipVec.y;
	    *(U + 2) = *(U + 2) - slipVec.z;
	}
    }
}
