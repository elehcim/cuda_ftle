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
 * Vector field for the strain-flow+double vortex problem 
 *  
 * See Rom-Kedar, Leonard, Wiggins (1990), Appendix A
 *
 * (wtf macro hell) 
 *
 */


__device__ inline void vec(float *x, float t, float *xout)
{

#define EPS 0.1
#define GAM 0.5

#define I_P ( x[0]*x[0]+((x[1]+1.0)*(x[1]+1.0)) )
#define I_N ( x[0]*x[0]+((x[1]-1.0)*(x[1]-1.0)) )

#define F1 ( (-(x[1]-1.0)/I_N)+((x[1]+1.0)/I_P)-0.5 )
#define F2 ( x[0] * ((1.0/I_N)-(1.0/I_P)) )

#define G1 (\
  (cosf(t/GAM)-1.0)*( (1.0/I_N) + (1.0/I_P) - ( (2.0*(x[1]-1.0)*(x[1]-1.0))/(I_N*I_N) ) - ( (2.0*(x[1]+1.0)*(x[1]+1.0))/(I_P*I_P) ) )\
+ (x[0]/GAM)*sinf(t/GAM)*( (GAM*GAM)*( ((x[1]-1.0)/(I_N*I_N)) - ((x[1]+1.0)/(I_P*I_P)) ) + 1.0 ) - 0.5 \
)

#define G2 (\
  (2.0*x[0]*(cosf(t/GAM)-1.0))*( ((x[1]-1.0)/(I_N*I_N)) + ((x[1]+1.0)/(I_P*I_P)) ) + (1.0/GAM)*sinf(t/GAM)\
* ( (((GAM*GAM)/(2.0))*((1.0/I_N)-(1.0/I_P))) - (x[0]*x[0]*GAM*GAM*( (1.0/(I_N*I_N)) - (1.0/(I_P*I_P)) )) - x[1] )\
)

   xout[0] = F1+(EPS*G1);
   xout[1] = F2+(EPS*G2);


}


__device__ inline void vec_mask(float *x, float t, int *xmask)
{
  *xmask = 1;
 
  return;
}

