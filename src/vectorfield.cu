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

/* This function provides the vector field for a time-dependent
 * double gyre. Given these settings, the results should always
 * match the example given on the Caltech CDS LCS tutorial:
 *
 * http://www.cds.caltech.edu/~shawn/LCS-tutorial/examples.html
 * 
 * (Example 7.1)
 *
 */

__device__ inline void vec(float *x, float t, float *xout)
{

#define A 0.1
#define eps 0.25
#define ome ((2.0/10.0)*M_PI)

#define a(t) (eps*sinf(ome*(t)))
#define b(t) (1-(2*eps*sinf(ome*(t))))
#define f(x,t) ((a(t)*(x)*(x))+(b(t)*(x)))
#define df(x,t) ((2*a(t)*(x))+b(t))

  xout[0] = -1.0*M_PI*A*sinf(M_PI*f(x[0],t))*cosf(M_PI*x[1]);
  xout[1] = M_PI*A*cosf(M_PI*f(x[0],t))*sinf(M_PI*x[1])*df(x[0],t);

  return;
}

__device__ inline void vec_mask(float *x, float t, int *xmask)
{
  *xmask = 1;
 
  return;
}
