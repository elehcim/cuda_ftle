/* 
 * vectorfield.c
 *
 * This file provides the vector flow function to be integrated
 * to track particles over time. The function vec() should always
 * take the same arguments as those rendered below, and deal
 * with them in a similar manner. 
 *
 * The function vec_mask() acts as a function to prevent calculating
 * useless points, such as the interior of a simulated solid body.
 * You can also use it to clear a path around a boundary where
 * the simulation might blow up. int mask is returned 1 for do compute,
 * 0 for ignore.
 *
 * The use of macros here may seem excessive, but by turning
 * small operations into macros, CUDA allows you to trim a register.
 * The current time-dependent double gyre implementation with 
 * -use-fast-math results in a total of 17 registers used.
 *
 * Also note that GCC processes comments before preprocessor macros,
 * so feel free to comment out prior functions and still use #define
 * without being afraid of crowding the namespace.
 * 
 * Raymond Jimenez
 * <raymondj@caltech.edu>
 *
 */

#include "vectorfield.h"

/* Vector field for a simple point vortex of strength GAM */

__device__ inline void vec(float *x, float t, float *xout)
{

#define GAM 1.0

  xout[0] = (-GAM/(2*M_PI)) * (-1.0/(x[0]*x[0]+x[1]*x[1]))*2*x[1];
  xout[1] = (GAM/(2*M_PI)) * (-1.0/(x[0]*x[0]+x[1]*x[1]))*2*x[0];

  return;
 
}

__device__ inline void vec_mask(float *x, float t, int *xmask)
{
  *xmask = 1;
 
  return;
}
