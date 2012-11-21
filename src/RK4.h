#ifndef RK4_H
#define RK4_H

#include <cuda.h>
#include "vectorfield.h"

extern __device__ inline float *RK4(float *x, float t0, float T, float *output);

#endif
