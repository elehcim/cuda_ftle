#include <stdio.h>
#include <math.h>
#include <cuda.h>

// Computing some Darwin drift like setup, 
// by means of a very naive CUDA implementation.
// 
// We should improve this even further (profiler, etc).
//
// For now, this example just illustrates that there is 
// some gain in using the GPU by means of relatively 
// modest coding efforts.




// Rigid body velocity -- just a constant
#define U 1.0


// Right-hand side of the ODE has to be present as a device function
__device__ inline void func(float *x, float t, float *dx)
{
    float X = x[0] - U*t, Y = x[1];
    float norm2 = X*X + Y*Y;

    dx[0] = U*(X*X - Y*Y)/(norm2*norm2);
    dx[1] = 2.0*U*X*Y/(norm2*norm2);
}



// Code from Raymond
// Again, device function
__device__ inline float *RK4(float *x, float t, float T, float *output)
{
  // NB func must be in this form:
  // void func(float *x, float t, float *xout)
  // where *x is a float[2] representing x y
  // and *xout[2] represents x' y'


  float k1[2], k2[2], k3[2], k4[2]; // Runge-Kutta terms
  float xpos[2], xnext[2]; // temporary x buffers 
 

  // Jvk: decreased time step

  // emperically "good enough" stepize 
  float h = 1e-3;

  // int i should be good enough unless you're taking a 
  // crazy number of steps, >65536 min
  int i;

  // initialize our buffer with our initial state
  xpos[0] = x[0];
  xpos[1] = x[1];

  xnext[0] = x[0];
  xnext[1] = x[1];

  for(i=(int)(T/h);i>0;i--)
  {
    func(xnext, t, k1);

    xpos[0] = xnext[0]+(.5*h*k1[0]);
    xpos[1] = xnext[1]+(.5*h*k1[1]);
    func(xpos, t+(.5*h), k2);

    xpos[0] = xnext[0]+(.5*h*k2[0]);
    xpos[1] = xnext[1]+(.5*h*k2[1]);
    func(xpos, t+(.5*h), k3);

    xpos[0] = xnext[0]+(h*k3[0]);
    xpos[1] = xnext[1]+(h*k3[1]);
    func(xpos, t+h, k4);
  
    xpos[0] = xnext[0];
    xpos[1] = xnext[1];

    xnext[0] = xpos[0]+((1.0/6.0)*h*(k1[0]+(2.0*k2[0])+(2.0*k3[0])+k4[0]));
    xnext[1] = xpos[1]+((1.0/6.0)*h*(k1[1]+(2.0*k2[1])+(2.0*k3[1])+k4[1]));

    t=t+h;
  }

  output[0] = xnext[0];
  output[1] = xnext[1];

  return output;
}


__global__ void integrate(float *xposout, float xmax, float dx)
{
    // Setup
    float xin[2], xout[2];
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Obtain starting position from index
    xin[0] = -xmax + idx*dx;
    xin[1] = 0.0;

    // Do not integrate if particle is inside the cylinder
    if (fabs(xin[0]) < 1.0) {
	xposout[idx] = 0.0;
	return;
    }

    // Integrate forward in time 
    RK4(xin, 0.0, 10.0, xout);

    // Store result of computation in global memory
    
    //printf("%f\n", xout[0]);
    xposout[idx] = xout[0];
}


int main()
{
    // Number of particles
    int N = 256*256;
    // Distribution along the x-axis
    float xmax = 10.0, dx = 2.0*xmax/N; 

    // Arrays to hold final position, on device and host
    float *xposout, *xposout_host;

    // Initialize
    xposout_host = (float *)malloc(N*sizeof(float));
    cudaMalloc((void **) &xposout, N*sizeof(float));   

    int blocksize = 512;
    int nBlocks = N/blocksize;

    // Execute kernel
    integrate <<< nBlocks, blocksize >>>(xposout, xmax, dx);

    // Obtain final result
    cudaMemcpy(xposout_host, xposout, sizeof(float)*N, cudaMemcpyDeviceToHost);

    // Block until kernel has completed
    cudaError_t err = cudaThreadSynchronize();
    if (err != cudaSuccess) {
	fprintf(stderr, "ERROR");
	fprintf(stderr, cudaGetErrorString(err));
	return -1;
    }

    // Output some particles
/*    for (int i = 0; i < 64; i++) {
	int n = N/64*i;
	printf("%d: %f\n", n, xposout_host[n]);
    }
*/

    // Cleanup
    free(xposout_host);
    cudaFree(xposout);

    return 0;
}
