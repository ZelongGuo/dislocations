#include <stdio.h>

typedef struct {
    double xx, yy, zz, xy, xz, yz;
} Tensor;

typedef struct {
    double I1, I2, I3;
    Tensor stress;
} StressState;

StressState StrainToStress(Tensor Strain, double lambda, double mu) {
    StressState Stress;
    Stress.stress.xx = 2 * mu * Strain.xx + lambda * (Strain.xx + Strain.yy + Strain.zz);
    Stress.stress.yy = 2 * mu * Strain.yy + lambda * (Strain.xx + Strain.yy + Strain.zz);
    Stress.stress.zz = 2 * mu * Strain.zz + lambda * (Strain.xx + Strain.yy + Strain.zz);
    Stress.stress.xy = 2 * mu * Strain.xy;
    Stress.stress.xz = 2 * mu * Strain.xz;
    Stress.stress.yz = 2 * mu * Strain.yz;
    Stress.I1 = Stress.stress.xx + Stress.stress.yy + Stress.stress.zz;
    Stress.I2 = -(Stress.stress.xx * Stress.stress.yy + Stress.stress.yy * Stress.stress.zz + Stress.stress.xx * Stress.stress.zz) + Stress.stress.xy * Stress.stress.xy + Stress.stress.xz * Stress.stress.xz + Stress.stress.yz * Stress.stress.yz;
    Stress.I3 = Stress.stress.xx * Stress.stress.yy * Stress.stress.zz + 2 * Stress.stress.xy * Stress.stress.xz * Stress.stress.yz - (Stress.stress.xx * Stress.stress.yz * Stress.stress.yz + Stress.stress.yy * Stress.stress.xz * Stress.stress.xz + Stress.stress.zz * Stress.stress.xy * Stress.stress.xy);
    return Stress;
}



