#ifndef VECTORFIELD_H
#define VECTORFIELD_H

#include <math.h>
#include <cuda.h>

extern __device__ inline void vec(float *x, float t, float *xout);
extern __device__ inline void vec_mask(float *x, float t, int *xmask);

#endif
