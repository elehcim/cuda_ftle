/*
 * RK4.c
 *
 * RK4 is a simple Runge-Kutta 4 implementation
 * that is optimized for less register usage on
 * CUDA devices. It integrates the function:
 *
 * vec(float *xout, float t, float *xin)
 * 
 * (see vectorflow.c for details). Integration time
 * may be positive or negative.
 *
 * Notes: Precision is somewhat unreliable. For a 
 * given stepsize, the result may be different than
 * what you expect, but should be in the ballpark.
 * Floating-point arithmetic is not associative 
 * on CUDA devices due to rounding (see NVIDIA CUDA
 * Best Practices Manual v2.3, 7.3.2), so the code
 * here returns a different value than if you 
 * calculated k_n individually and summed them afterward.
 *
 */

#include "RK4.h"

__device__ inline float *RK4(float *x, float t0, float T, float *output)
{

  float k[2]; // Runge-Kutta term
  float xpos[2],xn[2]; // temporary x buffers
//  int mask;

  // emperically "good enough" stepize
  float h = 0.0001;
  float t;

  float l;

  // initialize our buffer with our initial state
  xpos[0] = x[0];
  xpos[1] = x[1];

  output[0] = x[0];
  output[1] = x[1];

  t = t0;
  
  if (T<0) 
    h = -h;

// never unroll the following loop, or else there
// is a chance CUDA will eat itself and freeze

#pragma unroll 1 
  for(l=fabsf(T/h);l>0.0;l-=1.0)
  {
    vec(output, t, k);
    xn[0]=(h/6.0)*k[0];
    xn[1]=(h/6.0)*k[1];

    xpos[0] = output[0]+(.5*h*k[0]);
    xpos[1] = output[1]+(.5*h*k[1]);
    vec(xpos, t+(.5*h), k);
    xn[0]+=(h/6.0)*2.0*k[0];
    xn[1]+=(h/6.0)*2.0*k[1];

    xpos[0] = output[0]+(.5*h*k[0]);
    xpos[1] = output[1]+(.5*h*k[1]);
    vec(xpos, t+(.5*h), k);
    xn[0]+=(h/6.0)*2.0*k[0];
    xn[1]+=(h/6.0)*2.0*k[1];

    xpos[0] = output[0]+(h*k[0]);
    xpos[1] = output[1]+(h*k[1]);
    vec(xpos, t+h, k);
    xn[0]+=(h/6.0)*k[0];
    xn[1]+=(h/6.0)*k[1];

    output[0]+=xn[0];
    output[1]+=xn[1];
    
    t+=h;

/*
    vec_mask(output,t,&mask);
    if(!mask)
    {
      output[0] = -1.0/0.0;
      output[1] = -1.0/0.0;
      break;
    }
*/
  }



  return output;
}


