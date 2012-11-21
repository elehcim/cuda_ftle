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

/* 
 * Vector field for the two-stirrer Stokes flow problem
 *
 * NB: this is pretty convoluted atm. We have a known 
 * streamfunction, so we find polar derivatives via
 * central difference, then convert to x/y.
 * 
 */


__device__ inline void vec_polar(float r, float theta, float t, float *dr, float *dtheta)
{

#define EPS 0.001

#define P 2.0
#define A 1.0
#define B (0.5298833*A)
#define SIGMA (1.17*A*A/P)

#define STREAM_CYL(r,th) (\
   (SIGMA/2.0)*(logf((r*r - 2.0*B*r*cosf(th) + B*B)\
 / (A*A - 2.0*B*r*cosf(th) + B*B*r*r/(A*A)))\
 + (1.0 - r*r/(A*A))*(A*A - B*B*r*r/(A*A))\
 / (A*A - 2.0*B*r*cosf(th) + B*B*r*r/(A*A)))\
)


#define STREAM(r,th,t) (\
( (int)floorf(t)%2  ? -1.0 : 0.0 )*STREAM_CYL(r,th) +\
( (int)floorf(t)%2  ?  0.0 : 1.0 )*STREAM_CYL(r,th+M_PI)\
)


  *dtheta = -(STREAM((r+EPS), theta, t)-STREAM((r-EPS),theta,t))/(2*EPS*r);
  *dr = (STREAM(r, (theta+EPS), t)-STREAM(r,(theta-EPS),t))/(2*EPS*r);

  return;
}


__device__ inline void vec(float *x, float t, float *xout)
{

  float r;
  float theta;
  float dr;
  float dtheta;

  theta = atan2f(x[1],x[0]);
  r = sqrtf(x[0]*x[0]+x[1]*x[1]);

  vec_polar(r,theta,t,&dr,&dtheta);

  xout[0] = dr*cosf(theta)+r*(-sinf(theta))*dtheta;
  xout[1] = dr*sinf(theta)+r*(cosf(theta))*dtheta;

  return;
}


__device__ inline void vec_mask(float *x, float t, int *xmask)
{
  if(sqrtf(x[0]*x[0]+x[1]*x[1])<1.0)
    *xmask = 1;
  else
    *xmask = 0;
  
  return;
}


